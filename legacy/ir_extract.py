"""
레퍼런스 보컬에서 임펄스 응답(IR)을 **측정**해 가져온다.

기존 방식과의 차이
------------------
`pipeline.extract_reverb_params` 는 같은 감쇠 구간에서 기울기 하나(RT60)만 뽑고
나머지를 버린 뒤, `노이즈 × exp(-at)` 로 IR 을 새로 합성한다. 그 결과 실측상

    · 대역별 감쇠가 거의 동일 (100Hz 0.73s vs 8kHz 0.65s)   — 실제 방은 고역이 훨씬 빨리 죽음
    · 0 ms 부터 에코 밀도가 최대                              — 초기반사가 없어 노이즈 워시로 들림

가 되어 "화장실 같은" 소리가 난다. 지수감쇠 노이즈는 물리적으로 완벽히 확산된
이상적 공간의 소리이고, 그런 공간은 실재하지 않는다.

여기서는 요약하지 않고 **파형이 가진 시간-주파수 감쇠 포락선을 그대로** 가져온다.
늘어난 것은 학습 자유도가 아니라 측정 해상도이므로, 자유도만 늘리고 기준이 없어지는
문제를 만들지 않는다. wet 은 지금처럼 학습에 맡긴다.

합치는 방법
-----------
음절마다 음높이·음색이 다르므로 파형을 직접 평균하면 위상이 상쇄돼 뭉개진다.
STFT 크기 도메인에서 bin 별로 **각 구간의 시작 프레임 대비 상대 감쇠**를 구한 뒤
구간들의 중앙값을 취한다. 중앙값이라 한두 구간에 섞인 직접음·반주 누수에 강하다.
위상은 랜덤화한다 — 확산 잔향의 후반부는 정의상 노이즈성이고, 위상을 보존하려 해도
서로 다른 음정의 구간을 합칠 수 없다.
"""

from typing import Dict, Optional

import numpy as np

__all__ = ["extract_reverb_ir"]


def extract_reverb_ir(
    y: np.ndarray,
    sample_rate: int = 44100,
    ir_ms: float = 400.0,
    n_fft: int = 512,
    hop: int = 128,
    min_segments: int = 4,
    seed: int = 0,
) -> Optional[Dict]:
    """레퍼런스에서 IR 을 측정해 돌려준다.

    Returns:
        dict(ir=[N] float32, n_segments=int, ir_ms=float) 또는 감쇠 구간이 없으면 None
    """
    from pipeline import _find_decay_segments  # 순환 임포트 회피용 지연 임포트

    segs = _find_decay_segments(y, sample_rate)
    if segs is None:
        return None
    _, env_hop, offsets = segs
    if len(offsets) < min_segments:
        return None

    x = np.asarray(y, dtype=np.float64)
    ir_len = int(sample_rate * ir_ms * 1e-3)
    n_frames = ir_len // hop
    win = np.hanning(n_fft)

    def _stft_mag(seg: np.ndarray, n_frames: int = n_frames) -> Optional[np.ndarray]:
        """[n_frames, n_fft//2+1] 크기 스펙트로그램."""
        out = []
        for t in range(n_frames):
            s = t * hop
            if s + n_fft > seg.size:
                return None
            out.append(np.abs(np.fft.rfft(seg[s:s + n_fft] * win)))
        return np.asarray(out)

    rows = []
    for off_f, end_f in offsets:
        start = off_f * env_hop
        # 감쇠 구간의 **실제 길이**를 넘겨 자르지 않는다.
        # 넘기면 다음 음절이 IR 안으로 들어와 감쇠가 상쇄된다 — 실측에서 400 ms 고정
        # 길이로 자르자 구간 대부분이 150 ms 뒤 다시 +20 dB 로 치솟았고, 중앙값을
        # 취해도 상승분이 남아 IR 이 임펄스 하나로 붕괴했다(초기 5 ms 에 에너지 99.97%).
        seg_len = (end_f - off_f) * env_hop
        usable = min(ir_len, seg_len)
        if usable < n_fft * 2 or start + usable + n_fft > x.size:
            continue

        mag = _stft_mag(x[start:start + usable + n_fft], n_frames=usable // hop)
        if mag is None or mag.shape[0] < 3:
            continue

        # 구간의 절대 음량 차이만 제거한다 — **스칼라 하나**로 나눈다.
        #
        # bin 별로 mag/mag[0] 하면 안 된다. 첫 프레임에서 값이 거의 0 인 bin
        # (잔향이 실리지 않은 주파수)이 분모가 되면 그 bin 의 상대값이 폭발하고,
        # 소수의 그런 bin 이 합계를 지배해 감쇠가 사라진다 — 실측에서 총에너지는
        # 0 → -21 dB 로 잘 떨어지는데 bin 정규화 값은 0 → +5 dB 로 뒤집혔다.
        rel = mag / (np.linalg.norm(mag[0]) + 1e-12)

        # 단조 감소가 깨진 구간은 버린다(직접음 잔재·반주 누수·다음 음절 시작).
        tot_db = 20.0 * np.log10(mag.sum(axis=1) / (mag[0].sum() + 1e-12) + 1e-12)
        run_min = np.minimum.accumulate(tot_db)
        if np.max(tot_db - run_min) > 6.0:
            continue
        if tot_db[-1] > -6.0:          # 충분히 감쇠하지 않은 구간도 제외
            continue

        # 길이가 제각각이다. 짧은 구간을 0 으로 패딩하면 안 된다 — 뒤쪽 프레임에서
        # 절반 이상이 0 이 되어 중앙값이 0 으로 눌리고 IR 이 임펄스로 붕괴한다
        # (실측: 26개 채택했는데도 초기 5 ms 에 에너지 99.99%).
        # 데이터가 없는 프레임은 NaN 으로 두고 nanmedian 으로 있는 것만 합친다.
        pad = np.full((n_frames, rel.shape[1]), np.nan)
        k = min(n_frames, rel.shape[0])
        pad[:k] = rel[:k]
        rows.append(pad)

    if len(rows) < min_segments:
        return None

    stack = np.asarray(rows)
    # 프레임별로 데이터가 있는 구간이 몇 개인지 — 너무 적으면 신뢰할 수 없으므로 자른다
    valid = np.sum(~np.isnan(stack[:, :, 0]), axis=0)
    keep = int(np.argmax(valid < min_segments)) if np.any(valid < min_segments) else n_frames
    if keep < 4:
        return None

    with np.errstate(invalid="ignore"):
        decay = np.nanmedian(stack[:, :keep], axis=0)     # [keep, F]
    decay = np.nan_to_num(decay, nan=0.0)
    decay = np.maximum(decay, 0.0)
    n_frames = keep
    ir_len = n_frames * hop

    # 주의: 여기서 decay/decay[0] 로 다시 정규화하면 안 된다.
    # rel 단계에서 이미 스칼라로 정규화했고, bin 별로 한 번 더 나누면 첫 프레임의
    # 작은 bin 이 분모가 되어 감쇠가 사라진다 — 실측에서 0→-17 dB 로 잘 떨어지던
    # 포락선이 0→-2 dB 로 평탄해졌다.

    # 랜덤 위상으로 재합성 (확산 잔향 = 노이즈성)
    rng = np.random.default_rng(seed)
    ir = np.zeros(ir_len + n_fft, dtype=np.float64)
    wsum = np.zeros_like(ir)
    for t in range(n_frames):
        phase = rng.uniform(-np.pi, np.pi, decay.shape[1])
        phase[0] = 0.0
        if n_fft % 2 == 0:
            phase[-1] = 0.0
        frame = np.fft.irfft(decay[t] * np.exp(1j * phase), n=n_fft) * win
        s = t * hop
        ir[s:s + n_fft] += frame
        wsum[s:s + n_fft] += win ** 2
    # overlap-add 정규화. 바닥을 1e-8 로 두면 안 된다 — 신호 맨 앞은 창이 하나만
    # 겹쳐 wsum 이 0 에 가깝고, 그걸로 나누면 그 지점이 수천만 배 증폭돼 거대한
    # 스파이크가 생긴다. 실측에서 IR 이 임펄스처럼 보이고 프레임 에너지가
    # 0 → -58 dB 절벽이 된 원인이 이것이다. 내부 정상값 기준으로 바닥을 잡는다.
    w = wsum[:ir_len]
    floor = 0.2 * float(np.median(w[w > 0])) if np.any(w > 0) else 1e-8
    ir = ir[:ir_len] / np.maximum(w, floor)

    # 에너지 정규화 — wet 스케일과 분리되도록
    e = np.sqrt(np.sum(ir ** 2))
    if e < 1e-12:
        return None
    ir = ir / e

    return {
        "ir": ir.astype(np.float32),
        "n_segments": len(rows),
        "ir_ms": ir_ms,
    }
