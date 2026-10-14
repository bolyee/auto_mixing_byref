"""보컬 E2E 매칭 파이프라인: EQ, 컴프레서, 수동 리버브의 처리와 결과 지표를 담당한다."""

import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim

from modules import (
    DifferentiableCompressor,
    DifferentiableEQ,
    DifferentiableReverb,
    compute_spectral_envelope,
    compute_stft_crest_factor,
    compute_stft_rms_variance,
    loudest_window,
)
from losses import DSPMatchingLoss, LoudnessMeter

__all__ = ["match_e2e", "E2EChain"]

TARGET_SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80
_EPS = 1e-8

# ===== 보컬 EQ 대역 — 이 세 상수가 유일한 진실 공급원 =====
#
# 밴드 범위와 손실의 평가 대역 Ω 를 **같은 값에서 파생**시킨다. 둘이 어긋나면
# "고칠 수단은 없는데 오차만 쌓이는" 대역이 생긴다. `match_e2e` 가 손실을 만들 때
# tone_fmin/tone_fmax 로 이 상수를 명시 전달하므로, 여기만 고치면 전부 따라온다.
#
# 양 끝을 잘라내는 기준은 하나다 — **신뢰할 수 없는 데이터는 타겟으로 삼지 않는다.**
#
#   아래쪽 200 Hz: 보컬 분리(source separation) 과정에서 100~150 Hz 대역의 보컬이
#           악기에 마스킹되어, 분리 모델이 그 대역 에너지를 복원하지 못한다. 노이즈가
#           섞인 정도가 아니라 정보 자체가 소실된 구간이다.
#   위쪽 10 kHz: 손실 압축(mp3 등) 레퍼런스는 이 위 정보가 대부분 없다. 실측
#           (Golden Acapella.mp3) 17 kHz 위에서 raw 대비 +43~+50 dB 의 코덱 절벽이
#           있었고, 손실은 그것을 "톤 차이"로 읽어 EQ 에 거대한 컷을 요구했다.
#
# 대역 제한은 이 범위 지정만으로 끝난다 — 별도의 HPF/LPF 는 두지 않는다.
VOCAL_EQ_MIN_FREQ = 200.0
VOCAL_EQ_MAX_FREQ = 10000.0
# 기본 밴드 수. 옥타브 간격을 유지하도록 범위에 맞춰 고른다
# (100~10 kHz J=20 의 Δ 0.3497 → 200~10 kHz J=17 의 Δ 0.3527, +0.9%).
VOCAL_EQ_BANDS = 17

# 컴프(L_dyn)가 보는 구간 길이. None 이면 곡 전체.
#
# **다이내믹 매칭의 목표는 곡 전체가 아니라 '하이라이트 구간'이다.** 실제 믹싱에서
# 컴프를 잡을 때 후렴 같은 가장 큰 대목을 기준으로 잡는 것과 같은 정의다.
#
# 중요한 것은 **학습과 평가가 같은 구간을 본다**는 점이다. 예전에 15초로 학습하고
# 곡 전체로 채점했더니 dynamics_error 가 0.116 → 0.655 로 나빴는데, 그건 15초에
# 최적화된 모델을 다른 기준으로 잰 결과였다. 다이내믹 지표는 구간 통계라 구간을 바꾸면 값
# 자체가 이동하고(실측: raw −1.10 dB, ref −1.63 dB — 이동량도 서로 다르다), 두 기준은
# 애초에 같은 축이 아니다. 지금은 §17 지표도 같은 하이라이트 구간에서 잰다.
#
# 부수 효과로 컴프 밸리스틱 재귀가 O(M) 이라 스텝이 크게 빨라진다
# (실측 곡 전체 128 s 2537 ms → 10 s 180 ms, 약 14배).
COMP_TRAIN_SECONDS: Optional[float] = 10.0

# 학습 손실은 실제 리버브 출력에서 측정하고 각 모듈로 가는 그래디언트는 분리한다.
# 컴프 ratio. **학습하지 않는 설정 상수**다(§7 — 학습시키면 R→1 에서 그래디언트가
# 소멸해 bypass 가 흡인점이 되고, threshold 와 축퇴가 생긴다). 학습 대상은 threshold
# 하나뿐이라는 구조는 그대로 두고, 값만 곡·상황에 맞게 고를 수 있게 열어 둔다.
#   1.0         = bypass (압축 없음)
#   10.0 (기본) = 적극적인 압축
#   1000        = 사실상 리미터 (상한, `DifferentiableCompressor.RATIO_MAX`)
COMP_RATIO: float = 10.0


# --------------------------------------------------------------------------- #
# 수동 리버브 파라미터 적용
# --------------------------------------------------------------------------- #

def set_reverb_params(reverb: nn.Module, rt60: float, wet: float) -> None:
    """스칼라 rt60/wet 을 DifferentiableReverb 의 노브에 그대로 꽂는다.

    모듈이 `rt60 = lo + (hi-lo)·σ(raw)`, `wet = wet_max·σ(raw)` 로 파라미터화돼 있으므로
    역함수(logit)를 취해 raw 값을 세팅한다. 범위 상수는 **모듈에서 읽는다** — 여기에
    복제해 두면 모듈 쪽 범위(특히 수동 모드의 `wet_max=1.0`)를 바꿨을 때 조용히 어긋난다.

    범위 밖 값은 경계로 클립된다(logit 이 ±∞ 로 발산하지 않게 1e-4 여유를 둔다).
    따라서 wet=1.0 을 요청해도 실제로는 wet_max 의 99.99 % 근처가 된다.
    """
    def _logit(p, lo, hi):
        u = float(np.clip((p - lo) / (hi - lo), 1e-4, 1 - 1e-4))
        return float(np.log(u / (1.0 - u)))

    rt_lo = float(getattr(reverb, "RT60_MIN", 0.1))
    rt_hi = float(getattr(reverb, "RT60_MAX", 4.0))
    wet_max = float(getattr(reverb, "wet_max", 0.7))

    with torch.no_grad():
        reverb.raw_rt60.fill_(_logit(rt60, rt_lo, rt_hi))
        reverb.raw_wet.fill_(_logit(wet, 0.0, wet_max))



# --------------------------------------------------------------------------- #
# E2E 체인
# --------------------------------------------------------------------------- #

class E2EChain(nn.Module):
    """STFT → EQ → Compressor → iSTFT → Reverb 를 하나의 미분 가능 그래프로 묶는다.

    각 모듈은 `modules.py` 의 것을 그대로 쓴다. 이 클래스가 하는 일은 **연결**뿐이다:
    중간에 numpy 로 나갔다 오지 않으므로 손실의 그래디언트가 Reverb → iSTFT →
    Compressor → EQ 까지 한 번에 흐른다.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SR,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        num_bands: int = VOCAL_EQ_BANDS,
        gain_scale_db: float = 15.0,
        reverb_seconds: float = 1.5,
        reverb_wet_max: float = 0.7,
        attack_ms: float = 0.5,
        release_ms: float = 120.0,
        comp_ratio: float = COMP_RATIO,
        active_modes: Optional[List[str]] = None,
        max_gain_db: Optional[float] = None,
    ) -> None:
        super().__init__()
        if max_gain_db is not None:
            gain_scale_db = max_gain_db
        self.gain_scale_db = gain_scale_db
        self.max_gain_db = gain_scale_db
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.active = set(active_modes or ["eq", "comp", "reverb"])

        # 밴드 하한이 200 Hz 다. 그 아래는 L_tone 평가 대역(Ω) 밖이라 그래디언트가 거의
        # 없어 EQ 파라미터에서 제외했다 — 남겨 두면 데이터 신호 없이 필터 스커트 겹침이나
        # 컴프 경로를 통해 표류하기만 한다. 저역은 아래 eq_mask 가 0 dB 바이패스로
        # 통과시킨다 — 고정 HPF 는 없다(아래 주석 참조).
        #
        # 하드 상한 없음. 게인 크기는 손실의 제곱 페널티(eq_l2 / eq_smooth)가 제어한다.
        self.eq = DifferentiableEQ(
            sample_rate=sample_rate, n_fft=n_fft, num_bands=num_bands,
            min_freq=VOCAL_EQ_MIN_FREQ, max_freq=VOCAL_EQ_MAX_FREQ,
            gain_scale_db=gain_scale_db, hard_clamp=False,
        )
        self.comp = DifferentiableCompressor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            attack_ms=attack_ms,
            release_ms=release_ms,
            ratio=comp_ratio,
        )
        # wet 상한: 학습 경로는 0.7 로 묶고(과도한 wet 방지), 수동 경로만 1.0 을 받는다.
        self.reverb = DifferentiableReverb(
            sample_rate=sample_rate, duration_seconds=reverb_seconds,
            wet_max=reverb_wet_max,
        )

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        # 대역 밖은 EQ 게인 0 dB 바이패스. 범위는 손실의 평가 대역 Ω 와 같다.
        #
        # **HPF/LPF 는 없다.** 예전에는 80 Hz HPF 와 16 kHz LPF 를 여기에 곱했는데, 이
        # 마스크는 출력에만 걸리고 레퍼런스에는 안 걸린다. L_tone 은 둘의 mel 포락선을
        # 비교하므로, 필터가 깎은 대역을 손실은 "톤이 부족하다"로 읽고 EQ 가 되밀어
        # 올리려 한다 — 필터와 EQ 가 서로 반대로 일한다. 대역 제한은 이 바이패스
        # 마스크가 담당하고, 필터는 두지 않는다.
        self.register_buffer(
            "eq_mask",
            torch.tensor(
                (freq_bins >= VOCAL_EQ_MIN_FREQ) & (freq_bins < VOCAL_EQ_MAX_FREQ),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    # -------------------------------------------------------------- #

    def eq_curve_db(self, amount: float = 1.0) -> torch.Tensor:
        """적용될 최종 EQ 곡선(dB). amount 로 사용자 슬라이더 반영."""
        if "eq" not in self.active:
            return torch.zeros_like(self.eq_mask)
        curve, _ = self.eq.get_eq_curve_db()
        return curve * self.eq_mask * amount

    def _analyze(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def _synthesize(self, S: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length,
        )

    def eq_output(self, x: torch.Tensor, eq_amount: float = 1.0) -> torch.Tensor:
        """STFT → EQ → iSTFT. 컴프 이전 신호 [T].

        EQ 게인을 복소 STFT 에 직접 곱한다 — magnitude 만 건드리고 위상을 따로 붙이는
        방식보다 수치적으로 안전하고, 게인이 실수라 위상이 보존된다.
        """
        length = x.shape[-1]
        S = self._analyze(x)  # [F, N] complex
        gain = torch.pow(10.0, self.eq_curve_db(eq_amount) / 20.0)
        return self._synthesize(S * gain.unsqueeze(-1), length)

    def to_dry(
        self,
        x: torch.Tensor,
        eq_amount: float = 1.0,
        comp_amount: float = 1.0,
        detach_comp_input: bool = False,
        return_stages: bool = False,
        comp_window: Optional[Tuple[int, int]] = None,
    ):
        """리버브 이전까지: STFT → EQ → iSTFT → Comp(시간영역). 반환 [T] (모노).

        컴프가 iSTFT **뒤**에 오는 이유: 다이내믹 타겟이 피크 기반 지표(크레스트 팩터
        = True Peak − RMS)이므로 디텍터가 실제 피크를 봐야 한다. STFT 프레임 RMS 로는
        짧은 트랜지언트를 못 봐서, 실측상 지속 음량만 깎이고 피크는 남아 지표가
        **올라갔다**(PLR 기준 14.2 → 17.8 dB). 시간 영역에서 2.9 ms 피크 디텍터로
        잡아야 다이내믹 지표를 실제로 낮출 수 있다.

        Args:
            detach_comp_input / return_stages: **현재 호출부가 없다.** 학습 루프가
                `to_dry` 대신 `eq_output` → `comp` → `apply_reverb` 를 직접 조립하도록
                바뀌면서(`match_e2e` 의 tone_src/dyn_src 생성부) 남은 스위치다. 지금
                `to_dry` 는 렌더(`forward`) 전용 경로다.
            comp_window: `(start, end)` 를 주면 컴프를 **그 구간에만** 적용한다. 학습에서
                다이내믹 손실을 곡 전체가 아니라 최대 볼륨 구간으로 재기 위한 것이다.
                이때 반환값은 `(곡 전체 y_eq, 구간 길이의 컴프 출력)` 으로 길이가 다르다.
                밸리스틱 재귀는 구간 시작에서 상태 0 으로 출발하는데, 릴리즈 120 ms 대비
                구간이 충분히 길어(15 초) 초기 과도상태의 영향은 무시할 수준이다.
                렌더링에서는 None 이어야 한다 — 곡 전체에 컴프가 걸려야 하기 때문이다.
        """
        y_eq = self.eq_output(x, eq_amount)

        y = y_eq
        if "comp" in self.active:
            comp_in = y_eq.detach() if detach_comp_input else y_eq
            if comp_window is not None:
                s, e = comp_window
                comp_in = comp_in[s:e]
            y_comp = self.comp(comp_in)
            if comp_amount != 1.0:
                y = comp_in + comp_amount * (y_comp - comp_in)  # dry/wet 블렌드
            else:
                y = y_comp

        return (y_eq, y) if return_stages else y

    def apply_reverb(self, y_dry: torch.Tensor) -> torch.Tensor:
        """드라이 모노 [T] → 리버브 통과 스테레오 [2, T]. 리버브 off 면 dry 를 복제한다.

        **그래프를 끊지 않는다.** 렌더(`forward`)와 학습 루프(`리버브 이후`)가 이 함수를 공유해야 "손실이 듣는 소리 = 저장되는 소리"가
        정의상 보장된다. 그래디언트를 끊고 싶은 쪽(`L_decay` 입력)은 호출부에서
        `torch.no_grad()` 로 감싼다 — 여기서 끊으면 안 된다.
        """
        length = y_dry.shape[-1]

        if "reverb" not in self.active:
            return torch.stack([y_dry, y_dry], dim=0)

        return self.reverb(y_dry)[..., :length]

    def forward(
        self,
        x: torch.Tensor,
        eq_amount: float = 1.0,
        comp_amount: float = 1.0,
    ) -> torch.Tensor:
        """전체 체인. 반환 [2, T] 스테레오 (리버브 off 면 dry 를 복제)."""
        y_dry = self.to_dry(x, eq_amount, comp_amount)
        return self.apply_reverb(y_dry)


# --------------------------------------------------------------------------- #
# 메인 엔트리
# --------------------------------------------------------------------------- #

def _soft_limit(y: np.ndarray, thresh: float = 0.85, ceiling: float = 0.96) -> np.ndarray:
    """tanh 소프트 리미터 — 0 dBFS 클리핑 방지 (원본 match_eq STEP 3 과 동일 규칙)."""
    peak = float(np.max(np.abs(y)))
    if peak <= 0.88:
        return y
    a = np.abs(y)
    over = a > thresh
    if np.any(over):
        y = y.copy()
        excess = a[over] - thresh
        y[over] = np.sign(y[over]) * (thresh + (ceiling - thresh) * np.tanh(excess / (ceiling - thresh)))
    return y


def match_e2e(
    raw_path: str,
    ref_path: str,
    out_path: str,
    manual_rt60: float = 1.0,
    manual_wet: float = 0.3,
    inst_path: Optional[str] = None,
    num_bands: int = VOCAL_EQ_BANDS,
    match_amount: float = 1.0,
    match_volume: bool = False,
    gain_scale_db: float = 15.0,
    comp_amount: float = 1.0,
    mode: str = "both",
    # --- E2E 전용 ---
    n_steps: int = 50,
    attack_ms: float = 0.5,
    release_ms: float = 120.0,
    comp_ratio: float = COMP_RATIO,
    lr_eq: float = 0.05,
    lr_comp: float = 0.05,
    eq_l2: float = 0.1,
    eq_smooth: float = 0.1,
    comp_thresh_weight: float = 0.0,
    verbose: bool = True,
    max_gain_db: Optional[float] = None,
) -> Dict:
    """EQ / Comp / Reverb 를 최적화하고 결과를 렌더링해 저장한다.

    손실은 **모듈별로 분리**되어 있다(`losses.DSPMatchingLoss` 참조). 가중합이 아니므로
    w_tone/w_dyn 같은 항 가중치는 없고, 모듈별 학습률(`lr_eq`/`lr_comp`)이 그 역할을 한다.

    반환 dict 는 이전 파이프라인 의 `match_eq` 와 키 호환이므로 `main.py` 에서 바로 교체 가능하다.

    Args:
        n_steps:          조인트 최적화 스텝 수. M1 CPU 기준 약 180 ms/step.
    """
    # ---------------- 모드 파싱 (원본 match_eq 규칙 유지) ----------------
    if mode == "both":
        active_modes = ["eq", "comp"]
    elif mode == "reverb":
        active_modes = ["eq", "comp", "reverb"]
    elif "," in mode:
        active_modes = [m.strip() for m in mode.split(",") if m.strip()]
    else:
        active_modes = [mode]

    if verbose:
        print(f"[E2E] active modes = {active_modes}")

    # ---------------- 오디오 로드 ----------------
    y_raw, _ = librosa.load(raw_path, sr=TARGET_SR, mono=True)
    y_ref, _ = librosa.load(ref_path, sr=TARGET_SR, mono=True)

    # 원본과 동일하게 -6 dBFS 피크 정규화로 헤드룸 확보
    y_raw = (y_raw / (np.max(np.abs(y_raw)) + _EPS)) * 0.501187
    y_ref = (y_ref / (np.max(np.abs(y_ref)) + _EPS)) * 0.501187

    t_raw = torch.from_numpy(y_raw).float()
    t_ref = torch.from_numpy(y_ref).float()

    # ---------------- 학습 구간 (크로스페이드) ----------------
    # 학습 구간 = **곡 전체**.
    # 구간을 잘라 쓰면 EQ 가 그 구간에 과적합되어 곡 전체 톤이 오히려 나빠진다
    # (실측: 10초 학습 시 곡 전체 tonal 2.419 로 무처리 원본 2.369 보다 악화).
    # 속도보다 정확도를 택한다.
    x_train = t_raw
    r_train = t_ref
    if verbose:
        print(f"[E2E] 학습 구간: raw {x_train.shape[0]/TARGET_SR:.1f}s (곡 전체), "
              f"ref {r_train.shape[0]/TARGET_SR:.1f}s")

    # ---------------- 체인 ----------------
    if max_gain_db is not None:
        gain_scale_db = max_gain_db

    chain = E2EChain(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        num_bands=num_bands,
        gain_scale_db=gain_scale_db,
        attack_ms=attack_ms,
        release_ms=release_ms,
        comp_ratio=comp_ratio,
        active_modes=active_modes,
        reverb_wet_max=1.0,
    )

    # 수동 리버브: RT60과 wet은 사용자가 지정하며 추정이나 학습을 하지 않는다.
    if "reverb" in active_modes:
        rt60_use = float(np.clip(manual_rt60, DifferentiableReverb.RT60_MIN, DifferentiableReverb.RT60_MAX))
        wet_use = float(np.clip(manual_wet, 0.0, chain.reverb.wet_max))
        set_reverb_params(chain.reverb, rt60_use, wet_use)
        if verbose:
            print(f"[E2E] 수동 리버브: RT60={rt60_use:.2f}s wet={wet_use*100:.0f}%")
    chain.reverb.raw_rt60.requires_grad_(False)
    chain.reverb.raw_wet.requires_grad_(False)
    # 어떤 손실을 계산할지. 침범 방지는 이제 detach 가 구조적으로 보장하므로, 이 스위치는
    # **순수하게 연산을 아끼기 위한 것**이다 — 움직일 파라미터가 없는 손실은 값도 그래디언트도
    # 쓸 데가 없다. (L_dyn 은 K-weighting IIR 2회 + 4배 리샘플로 곡 전체 기준 스텝당 1.5 초다.)
    train_tone = "eq" in active_modes
    train_dyn = "comp" in active_modes

    # 톤/다이내믹 손실을 리버브 뒤에서 잴지. 리버브가 꺼져 있으면 앞뒤가 같으므로 무의미하다.
    measure_post_reverb = (
        "reverb" in chain.active
    )
    if verbose and measure_post_reverb:
        print("[E2E] 손실 측정 지점: 리버브 **이후** (EQ·컴프 그래디언트가 리버브를 관통)")
    if verbose and not train_dyn:
        print("[E2E] 컴프 비활성 → L_dyn 계산 생략")

    criterion = DSPMatchingLoss(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        # 평가 대역 Ω 를 EQ 밴드 범위와 **같은 상수**에서 받는다. 손실 쪽 기본값에
        # 의존하면 여기만 바꿨을 때 EQ 는 새 대역, 손실은 옛 대역을 보게 된다.
        tone_fmin=VOCAL_EQ_MIN_FREQ,
        tone_fmax=VOCAL_EQ_MAX_FREQ,
        comp_thresh_weight=comp_thresh_weight if train_dyn else 0.0,
    )
    criterion.train()

    # 파라미터 그룹별 lr — 세 모듈의 파라미터 스케일과 곡률이 크게 다르므로
    # 단일 lr 로는 한쪽이 발산하거나 다른 쪽이 멈춘다.
    groups = []
    if "eq" in active_modes:
        groups.append({"params": chain.eq.parameters(), "lr": lr_eq})
    if "comp" in active_modes:
        groups.append({"params": chain.comp.parameters(), "lr": lr_comp})

    target = r_train.view(1, 1, -1)  # [B=1, C=1, T]

    # ---------------- 컴프 학습 구간: 각자의 하이라이트 (기본 10초) ----------------
    # threshold 는 큰 대목에서 얼마나 눌리는지로 결정되고, BS.1770 게이팅이 조용한 구간을
    # 어차피 빼버린다. 그래서 곡 전체를 돌릴 필요가 없다.
    #
    # 두 신호의 구간은 **독립적으로** 고른다. 다른 연주라 같은 타임스탬프가 대응하지
    # 않고, 다이내믹 지표는 통계량이라 시간 정렬이 필요 없다(§ 정렬 주석 참조).
    #
    # 타깃 텐서는 루프 밖에서 한 번만 만든다 — 손실의 타깃 캐시가 텐서 동일성으로
    # 검증하므로, 매 스텝 새로 슬라이스하면 캐시가 매번 무효화된다.
    comp_window = None      # 내 보컬 쪽 하이라이트 (학습 입력 + 지표 측정 구간)
    ref_dyn_window = None   # 레퍼런스 쪽 하이라이트 (학습 타깃 + 지표 측정 구간)
    target_dyn = target
    if "comp" in active_modes and COMP_TRAIN_SECONDS is not None:
        xs, xe, x_lufs = loudest_window(
            x_train.numpy(), TARGET_SR, win_sec=COMP_TRAIN_SECONDS
        )
        rs, re_, r_lufs = loudest_window(
            r_train.numpy(), TARGET_SR, win_sec=COMP_TRAIN_SECONDS
        )
        comp_window = (xs, xe)
        ref_dyn_window = (rs, re_)
        target_dyn = r_train[rs:re_].view(1, 1, -1)
        if verbose:
            print(
                f"[E2E] 다이내믹 기준 구간(하이라이트 {COMP_TRAIN_SECONDS:.0f}s): "
                f"raw {xs/TARGET_SR:.1f}~{xe/TARGET_SR:.1f}s ({x_lufs:.1f} LUFS) · "
                f"ref {rs/TARGET_SR:.1f}~{re_/TARGET_SR:.1f}s ({r_lufs:.1f} LUFS)"
            )

    history: List[Dict] = []
    if groups:
        optimizer = optim.Adam(groups)
        for step in range(n_steps):
            optimizer.zero_grad()

            # 각 손실은 최종 신호를 기준으로 계산하고 그래디언트를 해당 모듈로만 보낸다.
            y_eq = chain.eq_output(x_train)
            has_comp = "comp" in chain.active

            if not has_comp:
                tone_src = dyn_full = y_eq
            else:
                # 톤 손실은 컴프 설정을 고정하고 EQ로, 다이내믹 손실은 EQ 출력을 고정하고 컴프로 보낸다.
                tone_src = chain.comp(y_eq, detach_params=True)
                dyn_full = chain.comp(y_eq.detach())

            # 손실은 리버브 출력에서 측정한다.
            #
            # 여기는 **그래프를 끊지 않는다.** EQ·컴프의 그래디언트가 리버브 컨볼루션을
            # 관통해서 흘러야 "최종 출력 기준으로 EQ·컴프를 고른다"가 성립한다.
            #
            # 손실은 모노에서 계산하고 최종 지표도 `y_processed.mean(axis=0)` 로 재므로
            # 여기서도 채널 평균으로 내린다(측정 IR 은 좌우가 동일해 평균이 곧 원신호).
            if measure_post_reverb:
                shared = tone_src is dyn_full
                tone_src = chain.apply_reverb(tone_src).mean(dim=0)
                dyn_full = (tone_src if shared
                            else chain.apply_reverb(dyn_full).mean(dim=0))

            # 자르기는 **리버브 뒤**다. 구간을 먼저 자르고 리버브를 걸면 구간 시작 앞쪽
            # 신호가 만들어 낸 잔향이 통째로 빠져(테일 유입 없음) 구간 앞부분이 실제
            # 출력보다 드라이해진다 — 컴프의 밸리스틱과 같은 이유다.
            dyn_src = (dyn_full if comp_window is None
                       else dyn_full[comp_window[0]:comp_window[1]])

            # 그래프를 공유하는 것은 "unified" 뿐이다. 나머지는 tone/dyn 이 서로 다른
            # 그래프라 첫 backward 가 그래프를 해제해도 두 번째가 살아 있다.
            retain = False

            d: Dict = {}

            # --- EQ 쪽: 톤 손실 + EQ 자체 정규화 ---
            if train_tone:
                l_tone, d_tone = criterion.tone_loss(
                    tone_src.unsqueeze(0), target, sample_rate=TARGET_SR
                )
                d.update(d_tone)

                # 하드 클램프(tanh ±gain_scale_db)는 상한에 닿기 전까지 저항이 0 이라, 결국
                # "일단 크게 밀고 벽에서 잘리는" 사후 절단과 다를 바 없어진다. 제곱 페널티는
                # 게인이 커질수록 한계비용이 커져서(1 dB 밀 때 비용이 1 dB 지점 대비 14 dB
                # 지점에서 약 10 배), 작은 보정은 거의 공짜로 두고 큰 부스트만 억제한다.
                if eq_l2 > 0.0 or eq_smooth > 0.0:
                    _, band_gains = chain.eq.get_eq_curve_db()
                    reg = eq_l2 * torch.mean(band_gains**2)
                    # 인접 밴드 급변 억제 → 뾰족한 부스트 대신 완만한 곡선을 선호하게 한다
                    if band_gains.numel() > 1:
                        reg = reg + eq_smooth * torch.mean(torch.diff(band_gains) ** 2)
                    l_tone = l_tone + reg
                    d["L_eq_reg"] = float(reg.detach())

                l_tone.backward(retain_graph=retain)

            # --- 컴프 쪽: 다이내믹 손실 (+ threshold 심도 정규화는 dyn_loss 안에) ---
            if train_dyn:
                l_dyn, d_dyn = criterion.dyn_loss(
                    dyn_src.unsqueeze(0), target_dyn, sample_rate=TARGET_SR,
                    compressor=chain.comp,
                )
                d.update(d_dyn)
                l_dyn.backward()

            # 파라미터 집합이 겹치지 않으므로 backward 들의 그래디언트가 섞이지 않는다.
            # 따라서 단일 Adam 에 param group 으로 넣은 것이 옵티마이저 여러 개와 동일하다.
            optimizer.step()
            history.append(d)

            if verbose and (step % 25 == 0 or step == n_steps - 1):
                with torch.no_grad():
                    rt60, wet = chain.reverb.get_params()
                    thr, ratio = chain.comp.get_params()

                def _fmt(key: str) -> str:
                    # 계산을 건너뛴 손실은 값이 없다 — 0 으로 위장하지 않고 '-' 로 표시한다
                    return f"{d[key]:.3f}" if key in d else "  -  "

                print(
                    f"[E2E] step {step:4d}  "
                    f"tone={_fmt('L_tone_raw')} dyn={_fmt('L_dyn_raw')} "
                    f"decay={_fmt('L_decay_raw')}  "
                    f"thr={float(thr):+.1f}dB ratio={float(ratio):.1f}(고정) "
                    f"rt60={float(rt60):.2f}s wet={float(wet):.3f}"
                )
    else:
        print("[E2E] 활성 모듈 없음 — 최적화 건너뜀")

    # ---------------- 전체 오디오 렌더링 (단일 패스) ----------------
    chain.eval()
    with torch.no_grad():
        y_out = chain(
            t_raw,
            eq_amount=match_amount,
            comp_amount=comp_amount,
        )  # [2, T]

        if match_volume:
            # 레퍼런스 RMS 에 맞춰 레벨 정렬
            cur = torch.sqrt(torch.mean(y_out**2) + _EPS)
            tgt = torch.sqrt(torch.mean(t_ref**2) + _EPS)
            y_out = y_out * (tgt / cur)

        y_processed = y_out.cpu().numpy()

        # 리포팅용 지표는 **최종 출력에서 한 번만** 잰다.
        # 학습 루프는 움직일 파라미터가 없는 손실을 계산하지 않으므로(속도) 그 항의 원시값이
        # 히스토리에 남지 않는다. 예를 들어 측정 IR 리버브는 학습 파라미터가 없지만 UI 는
        # 감쇠 오차를 보여줘야 한다. 스텝마다 계산하는 대신 여기서 1회 계산하면 비용이
        # 1/n_steps 로 줄고, 값도 마지막 학습 스텝이 아니라 실제 결과물 기준이 된다.
        criterion.eval()
        report_losses = criterion.report(
            y_out.unsqueeze(0),
            target,
            sample_rate=TARGET_SR,
            compressor=chain.comp if "comp" in active_modes else None,
        )

    y_processed = _soft_limit(y_processed)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, y_processed.T, TARGET_SR)
    if verbose:
        print(f"[E2E] 저장: {out_path}")

    # ---------------- 지표 & 차트 데이터 ----------------
    y_mono = y_processed.mean(axis=0)

    _, raw_mel, _ = compute_spectral_envelope(y_raw, TARGET_SR, N_FFT, HOP_LENGTH, N_MELS)
    _, ref_mel, _ = compute_spectral_envelope(y_ref, TARGET_SR, N_FFT, HOP_LENGTH, N_MELS)
    _, proc_mel, _ = compute_spectral_envelope(
        np.ascontiguousarray(y_mono), TARGET_SR, N_FFT, HOP_LENGTH, N_MELS
    )

    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=TARGET_SR / 2)
    # 손실의 Ω 와 같은 대역이어야 화면 수치와 최적화 대상이 일치한다
    eval_mask = torch.tensor(
        (mel_freqs >= VOCAL_EQ_MIN_FREQ) & (mel_freqs < VOCAL_EQ_MAX_FREQ), dtype=torch.float32
    )

    def _norm_db(mel: torch.Tensor) -> torch.Tensor:
        """레벨 정렬된 mel dB. 손실(`losses._tone_db`)과 **같은 규칙**을 써야 한다.

        총에너지로 나누면 평가 대상이 아닌 200 Hz 미만까지 기준 레벨에 섞여 들어가고,
        차트에 그려지는 곡선과 실제로 최적화되는 양이 어긋난다.
        """
        db = 10.0 * torch.log10(mel + 1e-10)
        return db - torch.sum(db * eval_mask) / (torch.sum(eval_mask) + _EPS)

    raw_db, ref_db, proc_db = _norm_db(raw_mel), _norm_db(ref_mel), _norm_db(proc_mel)
    tonal_mae = float(
        (torch.sum(torch.abs(proc_db - ref_db) * eval_mask) / (torch.sum(eval_mask) + _EPS)).item()
    )

    t_stft_ref = torch.stft(
        t_ref, N_FFT, HOP_LENGTH, window=torch.hann_window(N_FFT), center=True, return_complex=True
    )
    t_stft_proc = torch.stft(
        torch.from_numpy(np.ascontiguousarray(y_mono)).float(),
        N_FFT, HOP_LENGTH, window=torch.hann_window(N_FFT), center=True, return_complex=True,
    )
    t_stft_raw = torch.stft(
        t_raw, N_FFT, HOP_LENGTH, window=torch.hann_window(N_FFT), center=True, return_complex=True
    )
    with torch.no_grad():
        cf_raw = float(compute_stft_crest_factor(t_stft_raw))
        cf_ref = float(compute_stft_crest_factor(t_stft_ref))
        cf_proc = float(compute_stft_crest_factor(t_stft_proc))
        rv_raw = float(compute_stft_rms_variance(t_stft_raw, N_FFT // 2 + 1))
        rv_ref = float(compute_stft_rms_variance(t_stft_ref, N_FFT // 2 + 1))
        rv_proc = float(compute_stft_rms_variance(t_stft_proc, N_FFT // 2 + 1))

    # 히스토리(학습 궤적)는 그대로 두고, 최종 지표만 렌더 결과 기준값으로 덮어쓴다.
    last = dict(history[-1]) if history else {"L_tone_raw": 0.0, "L_dyn_raw": 0.0, "L_decay_raw": 0.0}
    last.update({k: report_losses[k] for k in ("L_tone_raw", "L_dyn_raw", "L_decay_raw")})

    # --- UI 지표: 다이내믹 레인지(기본 크레스트 팩터, 고정된 크레스트 팩터) ---
    #
    # **측정 구간은 학습 구간과 같아야 한다.** 하이라이트로 학습하고 곡 전체로
    # 채점하면 서로 다른 축을 비교하는 것이 된다 — 구간 통계라 구간을 바꾸면
    # 값이 이동하고, 그 이동량은 신호마다 다르다. 그래서 comp_window 가 있으면 지표도
    # 같은 구간에서 잰다. 곡 전체 값은 참고용으로 따로 남긴다.
    t_proc = torch.from_numpy(np.ascontiguousarray(y_mono)).float()

    def _slice(t: torch.Tensor, win: Optional[Tuple[int, int]]) -> torch.Tensor:
        if win is None:
            return t
        s_, e_ = win
        return t[s_: min(e_, t.shape[-1])]

    with torch.no_grad():
        _meter = criterion.meter if criterion.meter is not None else LoudnessMeter(TARGET_SR)

        # 손실이 최적화하는 것과 **같은 지표**로 보고한다(기본 크레스트 팩터).
        _dyn = criterion._dyn_metric

        # 곡 전체 (참고용)
        src_var_full = float(_dyn(t_raw.unsqueeze(0))[0])
        ref_var_full = float(_dyn(t_ref.unsqueeze(0))[0])
        shaped_var_full = float(_dyn(t_proc.unsqueeze(0))[0])

        # 하이라이트 구간 (판정 기준). comp_window 가 없으면 곡 전체와 같다.
        src_var = float(_dyn(_slice(t_raw, comp_window).unsqueeze(0))[0])
        ref_var = float(_dyn(_slice(t_ref, ref_dyn_window).unsqueeze(0))[0])
        shaped_var = float(_dyn(_slice(t_proc, comp_window).unsqueeze(0))[0])

        lufs_raw = float(_meter.integrated_lufs(_slice(t_raw, comp_window).unsqueeze(0))[0])
        lufs_ref = float(_meter.integrated_lufs(_slice(t_ref, ref_dyn_window).unsqueeze(0))[0])
        lufs_proc = float(_meter.integrated_lufs(_slice(t_proc, comp_window).unsqueeze(0))[0])

    # --- UI 지표: 최대 게인 리덕션 ---
    # 학습된 컴프를 실제 신호에 한 번 통과시켜 GR 궤적을 뽑는다.
    gr_max_db = 0.0
    if "comp" in active_modes:
        with torch.no_grad():
            S_probe = chain._analyze(t_raw)
            gain = torch.pow(10.0, chain.eq_curve_db(match_amount) / 20.0)
            y_probe = chain._synthesize(S_probe * gain.unsqueeze(-1), t_raw.shape[-1])
            _, gr_traj = chain.comp(y_probe, return_gain_reduction=True)
            gr_max_db = float(gr_traj.min())  # GR 은 음수 → 최댓값은 최솟값

    # 크레스트 팩터는 항상 0 이상이라(피크 ≥ RMS) 예외적인 음수 값 대체가
    # 필요 없다. 지표가 하나로 통일됐다.
    dyn_mae = abs(shaped_var - ref_var)
    combined_mae = 0.65 * tonal_mae + 0.35 * dyn_mae

    with torch.no_grad():
        rt60, wet = chain.reverb.get_params()
        thr, ratio = chain.comp.get_params()
        curve = chain.eq_curve_db(match_amount)
        _, band_gains = chain.eq.get_eq_curve_db()

    freq_bins = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SR)
    idx = np.unique(np.logspace(0, np.log10(len(freq_bins) - 1), 200).astype(int))

    reverb_active = "reverb" in active_modes
    reverb_err = float(last["L_decay_raw"]) if reverb_active else 0.0

    return {
        "status": "success",
        "output_file": out_path,
        "match_error": combined_mae,
        "tonal_error": tonal_mae,
        "dynamics_error": dyn_mae,
        "reverb_error": reverb_err,
        "chart_data": {
            "frequencies": mel_freqs.tolist(),
            "raw_envelope": raw_db.tolist(),
            "ref_envelope": ref_db.tolist(),
            "proc_envelope": proc_db.tolist(),
            "eq_curve_x": [float(freq_bins[i]) for i in idx],
            "eq_curve_y": [float(curve[i]) for i in idx],
            "bands_x": chain.eq.bands.tolist(),
            "bands_y": band_gains.tolist(),
        },
        "compression_data": {
            "gain_reduction_max": [round(gr_max_db, 2)],
            "cf_raw": cf_raw,
            "cf_ref": cf_ref,
            "cf_proc": cf_proc,
            "cf_error": abs(cf_proc - cf_ref),
            "rms_var_raw": rv_raw,
            "rms_var_ref": rv_ref,
            "rms_var_proc": rv_proc,
            "rms_var_error": abs(rv_proc - rv_ref),
            "dynamics_error": dyn_mae,
            "src_dynamic_range": src_var,
            "ref_dynamic_range": ref_var,
            "shaped_dynamic_range": shaped_var,
            # 프론트가 `${r}:1` 로 그대로 찍으므로 여기서 반올림해 보낸다
            "ratio": round(float(ratio), 2),
            "threshold_db": round(float(thr), 1),
            "attack_ms": attack_ms,
            "release_ms": release_ms,
            # 다이내믹 지표의 기준 구간. 곡 전체가 아니라 하이라이트라는 것을 UI 가 알아야 한다.
            "dyn_scope": "highlight" if comp_window is not None else "full",
            "dyn_window_sec": (
                round(float(COMP_TRAIN_SECONDS), 1) if comp_window is not None else None
            ),
            "dyn_window_raw_s": (
                [round(comp_window[0] / TARGET_SR, 1), round(comp_window[1] / TARGET_SR, 1)]
                if comp_window is not None else None
            ),
            "dyn_window_ref_s": (
                [round(ref_dyn_window[0] / TARGET_SR, 1), round(ref_dyn_window[1] / TARGET_SR, 1)]
                if ref_dyn_window is not None else None
            ),
            # 참고용 곡 전체 다이내믹 지표. 판정 기준이 아니다(구간이 다르면 축이 다르다).
            "src_dynamic_range_full": src_var_full,
            "ref_dynamic_range_full": ref_var_full,
            "shaped_dynamic_range_full": shaped_var_full,
            "dynamics_error_full": abs(shaped_var_full - ref_var_full),
            "lufs_raw": round(lufs_raw, 2),
            "lufs_ref": round(lufs_ref, 2),
            "lufs_proc": round(lufs_proc, 2),
            "metric": "Crest Factor (True Peak dBTP - RMS dB)",
            "dyn_metric": "crest",
        },
        "reverb_data": {
            "rt60": float(rt60) if reverb_active else 0.0,
            # Wet은 사용자가 지정한 dry/wet 혼합 비율이다.
            "wet": float(wet) if reverb_active else 0.0,
            "loss": float(last["L_decay_raw"]),
            "error_db": reverb_err,
            "similarity": float(np.clip(100.0 * np.exp(-0.05 * reverb_err), 50.0, 99.5)),
            "manual": True,
            "active": reverb_active,
        },
        "gate_data": None,
        "e2e_data": {
            "steps": len(history),
            "loss_history": history,
            "final": last,
        },
    }
