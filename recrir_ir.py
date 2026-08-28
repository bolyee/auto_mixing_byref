"""Rec-RIR(블라인드 RIR 식별)로 레퍼런스/반주에서 임펄스 응답을 추정한다.

기존 `legacy/ir_extract.py` 를 대체한다
--------------------------------------
구버전은 감쇠 구간의 STFT 크기를 중앙값으로 합쳐 IR 을 직접 만들었다. 실측 결과
(`mixpractice inst.mp3`):

    · IR 길이가 281 ms 에서 끝나는데 그 지점이 아직 **-9 dB** 밖에 안 떨어진 곳
      → 게이트 리버브처럼 뚝 끊김. 이게 "리버브 성능이 나쁘다"의 주원인이었다.
    · 감쇠 곡선이 단조롭지 않다 (0 → -2.2 → -5.9 → **-3.9** → -6.9 dB).
      세그먼트 중앙값의 잡음이 IR 에 그대로 박힌다.
    · 페이드아웃 없이 -21 dB 지점에서 하드 컷.

Rec-RIR (arXiv 2509.15628, Audio-WestlakeU, MIT) 은 잔향 스펙트럼 재구성으로 CTF
필터를 추정하고 pseudo-intrusive measurement 로 RIR 을 뽑는다. 같은 음원에서:

    · 감쇠가 매끄럽게 단조 하강
    · 대역별 RT60 이 물리적으로 타당한 관계 (고역이 저역보다 빨리 죽음)

한계 (알고 쓴다)
----------------
· **16 kHz 모델**이다 (`config/Rec-RIR.toml` → `sr = 16000`). 8 kHz 위 잔향은
  추정되지 않는다. 44.1 kHz 로 리샘플해도 고역은 비어 있다.
· CTF 길이 L=60, hop 256 → 유효 RIR **0.96 s**. 그보다 긴 테일은 표현 못 한다.
· speech 로 학습됐다. 레퍼런스 보컬은 도메인 안이고, 반주(풀밴드)는 도메인 밖이다.
· 학습 RIR 은 gpuRIR 시뮬레이션(RT60 0.2~1.5 s)이다.
"""

import hashlib
import importlib
import os
import sys
from typing import Dict, Optional

import numpy as np

__all__ = ["estimate_ir", "REC_RIR_SR"]

REC_RIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Rec-RIR")
REC_RIR_SR = 16000
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ir_cache")

_MODEL = None  # (model, TF, pim) 싱글턴


# --------------------------------------------------------------------------- #
# 모델 로딩
# --------------------------------------------------------------------------- #

def _initialize_module(path: str, args: Optional[dict] = None):
    """Rec-RIR 의 `trainer_inferencer.utils.initialize_module` 과 동일한 동작.

    원본을 임포트하지 않는 이유: 그 모듈이 최상단에서 `matplotlib` 을 학습 플롯용으로
    임포트한다. matplotlib 은 컴파일 확장이라 Rec-RIR venv(py3.10)의 것을 이 프로젝트
    venv(py3.9)로 가져올 수 없다. 실제로 필요한 건 이 4줄뿐이다.
    """
    mod_path, _, cls = path.rpartition(".")
    obj = getattr(importlib.import_module(mod_path), cls)
    return obj(**(args or {}))


def _load_model(device: str = "cpu"):
    """모델·STFT 변환·PIM 을 한 번만 로드해 재사용한다."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import toml
    import torch

    if REC_RIR_DIR not in sys.path:
        sys.path.insert(0, REC_RIR_DIR)

    cfg = toml.load(os.path.join(REC_RIR_DIR, "config", "Rec-RIR.toml"))
    TF = _initialize_module(cfg["acoustic"]["path"], cfg["acoustic"]["args"])
    model = _initialize_module(cfg["model"]["path"], cfg["model"]["args"])

    ckpt = torch.load(os.path.join(REC_RIR_DIR, "ckpt", "epoch35.tar"), map_location=device)
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in ckpt["model"].items()
        if not any(x in k for x in ["ops", "params"])
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    pim = _initialize_module(cfg["EM_algo"]["path"], cfg["EM_algo"]["args"])
    _MODEL = (model, TF, pim)
    return _MODEL


# --------------------------------------------------------------------------- #
# 분석 구간 선택
# --------------------------------------------------------------------------- #

def select_window(
    y: np.ndarray,
    sample_rate: int = REC_RIR_SR,
    seconds: float = 5.0,
) -> int:
    """IR 을 추정할 구간의 시작 샘플을 고른다. 빈 구간이 가장 많은 창을 쓴다.

    **가장 시끄러운 구간을 고르면 안 된다.** 같은 음원의 5초 창 4개를 뽑아 추정 IR 의
    대역별 RT60 을 비교한 실측 결과 (`mixpractice inst.mp3`, 고역/저역 RT60 비):

        loudest         빈구간 10%  →  2.20   ← 최악
        max_envvar      빈구간 64%  →  1.38
        max_gaps        빈구간 96%  →  0.71   ← 최선
        median_energy   빈구간 23%  →  0.74

    실제 방의 고역/저역 비는 0.3~0.5 다. 즉 1 을 크게 넘는 값은 추정 실패에 가깝다.
    가장 시끄러운 구간은 소리가 꽉 차서 **잔향 테일이 신호에 마스킹된다** — Rec-RIR 이
    감쇠를 관측할 빈 구간이 없다. 반대로 빈 구간이 많은 창일수록 추정이 좋아진다.

    """
    import librosa

    n = int(seconds * sample_rate)
    if y.size <= n:
        return 0

    hop = int(0.01 * sample_rate)
    db = 20.0 * np.log10(
        librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0] + 1e-8
    )
    fpw = n // hop
    if len(db) <= fpw:
        return 0

    starts, energy, gaps = [], [], []
    for s in range(0, len(db) - fpw, 25):
        w = db[s: s + fpw]
        starts.append(s * hop)
        energy.append(float(w.mean()))
        gaps.append(float(np.mean(w < w.max() - 12.0)))

    starts = np.asarray(starts)
    energy, gaps = np.asarray(energy), np.asarray(gaps)

    # 무음 구간 제외. 이 게이트가 없으면 "빈 구간 100%"인 곡 시작 전 정적이 뽑힌다.
    active = energy > (energy.max() - 20.0)
    if not active.any():
        return 0

    return int(starts[int(np.argmax(np.where(active, gaps, -1e9)))])


# --------------------------------------------------------------------------- #
# IR 후처리
# --------------------------------------------------------------------------- #

def _normalize_direct(ir: np.ndarray, sample_rate: int) -> np.ndarray:
    """직접음 **피크 진폭**을 1 로 맞춘다.

    이렇게 해야 컨볼브했을 때 드라이 성분이 정확히 유니티 게인으로 통과하고, 잔향이
    그 위에 얹힌다. DRR(직접음/잔향 비)은 IR 이 가진 값 그대로 보존된다.

    전체 에너지로 정규화하면 안 된다. 테일이 길수록 직접음이 작아져 **보컬 본체가
    멀어지고 잔향만 커진다**.

    직접음 구간(5 ms)의 *에너지*로 정규화해도 안 된다. 그 창에는 초기반사가 섞여
    있어서 피크 진폭이 1 이 되지 않는다 — 실측에서 5 ms 에너지를 1 로 맞췄더니 피크
    진폭이 0.46 이 되어 드라이 보컬이 6.7 dB 뒤로 밀렸다.

    DRR 이 IR 안에 들어있으므로 별도의 wet/dry 믹싱이 필요 없다. 컨볼루션 한 번이
    드라이와 잔향을 동시에 만들어낸다. 드라이를 따로 더하면 직접음이 두 번 들어가고,
    Rec-RIR 의 IR 은 직접음 피크가 2.5 ms 지점에 있으므로(`method/pim.py` 가
    `argmax - 2.5ms` 부터 자른다) 400 Hz 간격 콤필터가 생긴다.
    """
    peak = float(np.abs(ir).max())
    return ir / peak if peak > 1e-12 else ir


def _edc_db(ir: np.ndarray) -> Optional[np.ndarray]:
    """Schroeder 역적분 에너지 감쇠 곡선(dB, 첫 샘플 기준 0)."""
    edc = np.cumsum(np.asarray(ir, dtype=np.float64)[::-1] ** 2)[::-1]
    if edc[0] <= 0:
        return None
    return 10.0 * np.log10(edc / edc[0] + 1e-20)


def _decay_time(db: np.ndarray, sample_rate: int, lo_db: float, hi_db: float) -> float:
    """EDC 의 [lo_db, hi_db] 구간에 직선을 맞추고 -60 dB 로 외삽한다."""
    i1, i2 = int(np.searchsorted(-db, lo_db)), int(np.searchsorted(-db, hi_db))
    if i2 <= i1 + 2 or i2 >= len(db):
        return 0.0
    slope = np.polyfit(np.arange(i1, i2) / sample_rate, db[i1:i2], 1)[0]
    return float(-60.0 / slope) if slope < 0 else 0.0


def measure_rt60(ir: np.ndarray, sample_rate: int) -> float:
    """Schroeder 역적분 EDC 의 -5 ~ -25 dB 구간 기울기로 T20 → RT60.

    -5 dB 부터 재는 이유는 초반에 직접음이 남아 기울기를 왜곡하기 때문이고,
    -25 dB 에서 끊는 이유는 그 아래가 노이즈 플로어에 묻히기 때문이다(ISO 3382).
    """
    db = _edc_db(ir)
    return 0.0 if db is None else _decay_time(db, sample_rate, 5.0, 25.0)


def measure_edt(ir: np.ndarray, sample_rate: int) -> float:
    """EDT (Early Decay Time): EDC 의 0 ~ -10 dB 기울기를 -60 dB 로 외삽.

    RT60 이 테일 전체의 물리적 길이라면 EDT 는 **체감 잔향감**에 가깝다. 사람이
    잔향을 판단하는 건 대부분 감쇠 초반이기 때문이다. 둘이 크게 다르면
    "꼬리는 긴데 별로 안 울리는" 공간(EDT < RT60)이거나 그 반대다.
    """
    db = _edc_db(ir)
    return 0.0 if db is None else _decay_time(db, sample_rate, 0.0, 10.0)


def measure_drr(ir: np.ndarray, sample_rate: int, direct_ms: float = 5.0) -> float:
    """DRR (Direct-to-Reverberant Ratio, dB). 직접음 대 잔향 에너지 비.

    직접음 구간은 피크 기준 +`direct_ms` 까지로 본다. 값이 클수록 드라이하다.
    보컬 리버브의 통상 범위는 +5 ~ +15 dB 이고, 0 dB 근처면 잔향 에너지가 직접음과
    같다는 뜻이라 매우 젖은 공간이다.
    """
    x = np.asarray(ir, dtype=np.float64)
    k = int(np.argmax(np.abs(x)))
    d = k + int(direct_ms * 1e-3 * sample_rate)
    e_dir = float(np.sum(x[:d] ** 2))
    e_tail = float(np.sum(x[d:] ** 2))
    if e_tail <= 1e-20:
        return 60.0
    return float(10.0 * np.log10(e_dir / e_tail)) if e_dir > 1e-20 else -60.0


def measure_clarity(ir: np.ndarray, sample_rate: int, split_ms: float = 80.0) -> float:
    """C80 (Clarity, dB). 피크 이후 첫 `split_ms` 의 에너지 대 나머지 에너지 비.

    음악용 명료도 지표다(말소리는 C50 을 쓴다). 높으면 또렷하고, 낮으면 잔향이
    직접음을 덮어 뭉개진다. 콘서트홀 기준 대략 -2 ~ +2 dB 가 흔하다.
    """
    x = np.asarray(ir, dtype=np.float64)
    k = int(np.argmax(np.abs(x)))
    d = k + int(split_ms * 1e-3 * sample_rate)
    early = float(np.sum(x[:d] ** 2))
    late = float(np.sum(x[d:] ** 2))
    if late <= 1e-20:
        return 60.0
    return float(10.0 * np.log10(early / late)) if early > 1e-20 else -60.0


def measure_band_rt60(ir: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """대역별 RT60. 공간의 **음색**을 드러낸다.

    실제 방은 고역이 저역보다 빨리 죽는다(공기 흡수·흡음재). 고역/저역 비가 보통
    0.3~0.6 이다. 구버전의 `노이즈 × exp(-at)` 합성 IR 은 모양이 지수함수 하나로
    강제돼 이 비가 1 에 가까웠고, 그게 "화장실 같은 소리"의 원인이었다.

    상한은 Nyquist 를 넘지 않게 잡는다 — IR 이 16 kHz 에서 올라온 경우 8 kHz 위는
    비어 있지만, 44.1 kHz 로 리샘플된 뒤이므로 필터 설계 자체는 유효하다.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = sample_rate / 2.0
    out: Dict[str, float] = {}
    for name, lo, hi in (("low", 80.0, 250.0), ("mid", 250.0, 2000.0), ("high", 2000.0, 7800.0)):
        hi = min(hi, nyq * 0.98)
        if lo >= hi:
            out[name] = 0.0
            continue
        try:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            band = sosfiltfilt(sos, np.asarray(ir, dtype=np.float64))
            db = _edc_db(band)
            out[name] = 0.0 if db is None else _decay_time(db, sample_rate, 5.0, 25.0)
        except Exception:
            out[name] = 0.0
    return out


# --------------------------------------------------------------------------- #
# 엔트리
# --------------------------------------------------------------------------- #

def estimate_ir(
    y: np.ndarray,
    sample_rate: int,
    target_sr: int = 44100,
    analysis_seconds: float = 5.0,
    device: str = "cpu",
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[Dict]:
    """`y` 에서 IR 을 추정해 `target_sr` 로 돌려준다.

    Args:
        y: 모노 파형 [T]. 레퍼런스 보컬이나 반주.
        sample_rate: `y` 의 샘플레이트.

    Returns:
        dict(ir=[N] float32 @target_sr, rt60=float, ir_ms=float,
             analysis_start_s=float, source_sr=int) 또는 실패 시 None
    """
    import librosa
    import torch

    if y is None or y.size == 0:
        return None

    # 버전 태그는 후처리(정규화 등)를 바꿀 때마다 올린다. 안 올리면 예전 규칙으로
    # 만든 IR 이 캐시에서 그대로 돌아온다.
    key = hashlib.sha1(
        np.ascontiguousarray(y, dtype=np.float32).tobytes()
        + f"|v4|{sample_rate}|{target_sr}|{analysis_seconds}".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, f"{key}.npz")

    if use_cache and os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            if verbose:
                print(f"[Rec-RIR] 캐시 적중: {cache_path}")
            out = {k: float(z[k]) for k in z.files if k not in ("ir", "source_sr")}
            out["ir"] = z["ir"]
            out["source_sr"] = int(z["source_sr"])
            return out
        except Exception:
            pass  # 캐시가 깨졌으면 그냥 다시 계산한다

    # Rec-RIR 은 16 kHz 전용이다. 입력을 거기 맞춘다.
    y16 = y if sample_rate == REC_RIR_SR else librosa.resample(
        np.asarray(y, dtype=np.float32), orig_sr=sample_rate, target_sr=REC_RIR_SR
    )
    start = select_window(y16, REC_RIR_SR, analysis_seconds)
    seg = y16[start: start + int(analysis_seconds * REC_RIR_SR)]
    if seg.size < int(1.0 * REC_RIR_SR):
        return None

    model, TF, pim = _load_model(device)
    with torch.no_grad():
        rir = pim.process(
            torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).unsqueeze(0).to(device),
            model, TF, device,
        )
    ir16 = rir.squeeze().detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(ir16).all() or np.abs(ir16).max() < 1e-9:
        return None

    # 16 kHz → 프로젝트 샘플레이트. 8 kHz 위는 비어 있다(모델 한계).
    ir = ir16 if target_sr == REC_RIR_SR else librosa.resample(
        ir16.astype(np.float32), orig_sr=REC_RIR_SR, target_sr=target_sr
    )
    ir = _normalize_direct(np.asarray(ir, dtype=np.float64), target_sr).astype(np.float32)

    # 지표는 모두 최종 IR(정규화·리샘플 후)에서 잰다. 배열 연산뿐이라 비용은 무시할
    # 수준이고, 화면에 보이는 값과 실제로 컨볼브되는 IR 이 항상 일치한다.
    bands = measure_band_rt60(ir, target_sr)
    rt_low, rt_high = bands.get("low", 0.0), bands.get("high", 0.0)

    out = {
        "ir": ir,
        "rt60": measure_rt60(ir, target_sr),
        "edt": measure_edt(ir, target_sr),
        "drr_db": measure_drr(ir, target_sr),
        "c80_db": measure_clarity(ir, target_sr),
        "rt60_low": rt_low,
        "rt60_mid": bands.get("mid", 0.0),
        "rt60_high": rt_high,
        # 고역/저역 감쇠 비. 실제 방은 0.3~0.6, 1 근처면 지수감쇠 합성처럼 부자연스럽다.
        "hf_lf_ratio": float(rt_high / rt_low) if rt_low > 1e-6 else 0.0,
        "ir_ms": float(len(ir) / target_sr * 1000.0),
        "analysis_start_s": float(start / REC_RIR_SR),
        "source_sr": int(sample_rate),
    }
    if verbose:
        print(f"[Rec-RIR] IR {out['ir_ms']:.0f}ms  RT60 {out['rt60']:.2f}s  "
              f"EDT {out['edt']:.2f}s  DRR {out['drr_db']:+.1f}dB  C80 {out['c80_db']:+.1f}dB  "
              f"대역 {rt_low:.2f}/{out['rt60_mid']:.2f}/{rt_high:.2f}s (고/저 {out['hf_lf_ratio']:.2f})  "
              f"분석구간 {out['analysis_start_s']:.1f}s~ ({analysis_seconds:.0f}s)")

    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache_path, **out)
        except Exception:
            pass

    return out
