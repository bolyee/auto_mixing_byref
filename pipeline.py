"""
End-to-End Joint Training 파이프라인.

기존 `modules.py` 의 순차(greedy) 최적화 — EQ 500 epoch → 1176 컴프(numpy) → 리버브 100 epoch —
를 대체한다. EQ / Compressor / Reverb 의 모든 파라미터를 **단일 연산 그래프**에 연결해
모듈별로 분리된 손실(`losses.DSPMatchingLoss`)로 각각 역전파한다.

    forward:   STFT → EQ → iSTFT → Compressor → Reverb        (직렬, numpy 왕복 0회)
    backward:  L_tone ← tone_src = Rev(Comp(y_eq, sg[θ_T])) → θ   (EQ)   (경로 분리)
               L_dyn  ← dyn_src  = Rev(Comp(sg[y_eq], θ_T)) → θ_T (Comp)
               ※ 리버브 통과 여부는 LOSS_MEASURE_POINT ("post_reverb" 기본)

각 손실이 자기 모듈만 움직인다는 성질을 **그래디언트 경로를 끊어서** 보장한다. 예전에는
세 손실을 가중합으로 묶고 침범 방지용 정규화 항으로 통제했는데, 구조로 보장되므로 그
항들은 삭제했다(`losses.py` 상단 주석 참조).

DSP 모듈(DifferentiableEQ / DifferentiableCompressor / DifferentiableReverb)과 유틸은
`modules.py` 것을 그대로 임포트해 재사용한다. 이 파일은 **연결과 학습 루프만** 담당한다.

기존 파이프라인 대비 달라진 점
------------------------------
1. **그래프 절단 제거.** 기존 `match_eq` 는 EQ 학습 후 `.numpy()` / `librosa.istft` 로
   빠져나갔다가 numpy 1176 컴프를 거쳐 `torch.tensor()` 로 그래프를 새로 시작했다.
   전부 `torch.stft` / `torch.istft` 체인으로 대체해 numpy 왕복이 0회다.
2. **미분 가능 컴프 사용.** `match_compression_1176`(이진탐색 50회, 미분 불가) 대신
   `DifferentiableCompressor`. 샘플 단위 피크 밸리스틱을 잃는 대신 그래디언트를 얻는다.
3. **학습 구간 크로스페이드.** `legacy/ddsp.py:69` `extract_active_segments` 는 비연속 5초 블록을
   그냥 `torch.cat` 해서 경계에 인위적 attack/decay 를 만든다. 이는 EDC 기울기 통계를
   직접 오염시키므로 여기서는 크로스페이드 버전을 쓴다.

알려진 한계
-----------
· rt60 / wet 축퇴: 전대역 RT60 이 1개뿐이라 "rt60↑ + wet↓" 와 "rt60↓ + wet↑" 가
  EDC 기울기에 거의 같은 효과를 낸다. 실측에서 wet 이 먼저 수렴하며 rt60 을 끌어내렸다.
  RT60 을 추출값으로 고정해 자유도 하나를 없애서 이 축퇴를 피한다.
· 비용: 10초 학습 구간 · 60 step 기준 약 14초 (M1 CPU, 렌더링 포함).
  기존 순차 방식(1~2초)보다 훨씬 느리다. `n_steps` 로 조절한다.
"""

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
    MeasuredIRReverb,
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
SMOOTH_MS = 10.0
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
# (실측 곡 전체 128 s 2537 ms → 15 s 273 ms, 9.3배).
COMP_TRAIN_SECONDS: Optional[float] = 15.0

# 손실을 **어디서 재고**, 그래디언트를 **어디로 흘릴지**.
#
#   "selective" (기본) — 두 손실 모두 컴프 통과 후 신호에서 재되, 경로는 갈라 둔다.
#         tone_src = Comp(y_eq, sg[θ_T])   ← θ 로만 흐름 (컴프 파라미터 고정)
#         dyn_src  = Comp(sg[y_eq], θ_T)   ← θ_T 로만 흐름 (컴프 입력 고정)
#       (기본값 LOSS_MEASURE_POINT="post_reverb" 에서는 두 신호가 여기서 리버브를 한 번
#        더 통과한 뒤 손실에 들어간다 — 아래 LOSS_MEASURE_POINT 주석 참조.)
#       sg 는 forward 값을 바꾸지 않으므로 두 신호와 실제 출력이 **수치적으로 동일**하다.
#       "손실이 듣는 소리"는 최종 출력과 같으면서, 모듈 간 간섭은 없다.
#
#   "split" — 예전 분리 모드. 톤을 컴프 **이전**(y_eq)에서 잰다. 컴프가 만드는 스펙트럼
#       변화를 EQ 가 보지 못한다.
#
#   "unified" — 완전 결합. sg 없이 둘 다 y_full 에서 잰다. 실제 곡에서 PLR 이 목표와
#       반대로 튀고(13.89 → 17.16 dB) EQ 가 저역을 −25 dB 까지 깎는 실패가 관측됐다.
#       L_dyn 이 θ 를, L_tone 이 θ_T 를 서로 잘못된 방향으로 끌어당긴 결과다. 재현용으로만
#       남겨 둔다.
LOSS_GRAD_MODE: str = "selective"

# 톤/다이내믹 손실을 **리버브 앞에서 재는가 뒤에서 재는가**.
#
#   "post_reverb" (기본) — tone_src·dyn_src 를 리버브까지 통과시킨 뒤 잰다. 손실이 보는
#       신호가 실제로 저장되는 출력과 같아지고(§18 감사 A 계열의 불일치 해소), EQ·컴프의
#       그래디언트가 리버브를 **관통해서** 흐른다. 리버브에 학습 파라미터가 없어도(측정 IR)
#       컨볼루션은 선형이라 backward 가 성립한다.
#   "pre_reverb" — 예전 동작. 컴프 출력에서 바로 잰다. 리버브는 렌더 전용 후처리가 된다.
#
# 대가: 스텝마다 곡 전체 길이 FFT 컨볼루션이 tone·dyn 각각 한 번씩(+backward) 더 붙는다.
# 실측 비용은 README / 아래 벤치마크 주석 참조.
#
# 주의(비대칭): 타깃은 레퍼런스 **원본**(이미 잔향이 걸린 신호)이고, 우리 쪽은 여기서
# 레퍼런스에서 추정한 IR 을 한 번 더 얹는다. 내 보컬에 원래 있던 잔향이 남아 있으므로
# 엄밀히는 "웻+웻 대 웻" 이다. 레퍼런스 디리버브(`ref_dry`)가 들어오기 전까지는 남는 갭이다.
LOSS_MEASURE_POINT: str = "post_reverb"

# 컴프 ratio. **학습하지 않는 설정 상수**다(§7 — 학습시키면 R→1 에서 그래디언트가
# 소멸해 bypass 가 흡인점이 되고, threshold 와 축퇴가 생긴다). 학습 대상은 threshold
# 하나뿐이라는 구조는 그대로 두고, 값만 곡·상황에 맞게 고를 수 있게 열어 둔다.
#   1.0        = bypass (압축 없음)
#   3.0 (기본) = 보컬 표준
#   1000       = 사실상 리미터 (상한, `DifferentiableCompressor.RATIO_MAX`)
COMP_RATIO: float = 3.0


# --------------------------------------------------------------------------- #
# 학습 구간 추출 (크로스페이드 버전)
# --------------------------------------------------------------------------- #

def extract_training_segments(
    y: torch.Tensor,
    sample_rate: int,
    segment_len_sec: float = 5.0,
    num_segments: int = 3,
    crossfade_ms: float = 30.0,
) -> torch.Tensor:
    """RMS 상위 N개 블록을 크로스페이드로 이어붙여 대표 학습 구간을 만든다.

    `legacy/ddsp.py` 의 `extract_active_segments` 는 블록을 그대로 concat 하는데, 시간적으로 떨어진
    블록이 맞닿으면 경계에서 **없던 attack/decay 가 생긴다**. 이 파이프라인의 L_decay 는
    바로 그 감쇠 기울기를 측정하므로 경계 아티팩트가 손실을 직접 오염시킨다.
    여기서는 등파워 크로스페이드로 경계를 지운다.
    """
    block = int(segment_len_sec * sample_rate)
    n = y.shape[0]
    if n <= block:
        return y

    n_blocks = n // block
    energies = []
    for i in range(n_blocks):
        seg = y[i * block : (i + 1) * block]
        energies.append((float(torch.sqrt(torch.mean(seg**2) + 1e-8)), i))

    energies.sort(key=lambda t: t[0], reverse=True)
    picked = sorted(idx for _, idx in energies[: min(num_segments, n_blocks)])
    blocks = [y[i * block : (i + 1) * block] for i in picked]

    xf = int(crossfade_ms * 1e-3 * sample_rate)
    if xf <= 0 or len(blocks) == 1:
        return torch.cat(blocks)

    # 등파워(sin/cos) 크로스페이드 — 합산 시 에너지가 보존되어 경계에 딥이 생기지 않는다
    t = torch.linspace(0, 1, xf, device=y.device)
    fade_in = torch.sin(0.5 * torch.pi * t)
    fade_out = torch.cos(0.5 * torch.pi * t)

    out = blocks[0]
    for nxt in blocks[1:]:
        head, tail = out[:-xf], out[-xf:]
        joined = tail * fade_out + nxt[:xf] * fade_in
        out = torch.cat([head, joined, nxt[xf:]])
    return out


# --------------------------------------------------------------------------- #
# 레퍼런스 리버브 파라미터 블라인드 추출
# --------------------------------------------------------------------------- #

def extract_reverb_params(
    y: np.ndarray,
    sample_rate: int = TARGET_SR,
    hop_ms: float = 5.0,
    min_decay_ms: float = 120.0,
    onset_floor_db: float = -25.0,
    smooth_ms: float = SMOOTH_MS,
    rise_db: float = 3.0,
    depth_db: float = 10.0,
    drop_db: float = 6.0,
) -> Dict:
    """레퍼런스 보컬에서 RT60 과 Wet 을 **수치로만** 추출한다.

    두 곡을 스펙트럼으로 비벼 비교하지 않는다. 곡이 다르면 가사·템포·음역이 전부
    달라 신호 대조 자체가 의미가 없기 때문이다. 대신 레퍼런스의 **음절 사이 묵음
    구간(speech pause)** 에서 잔향이 홀로 감쇠하는 구간을 찾아 스칼라 2개만 뽑고,
    그 시점에서 레퍼런스의 역할은 끝난다.

    검출
        20 ms 안에 6 dB 이상 떨어지는 **급락 지점**을 오프셋으로 본다. 전체 피크
        대비 고정 임계값을 쓰면 잔향이 긴 곡에서 테일이 임계 아래로 안 내려가
        오프셋이 하나도 안 잡힌다(실측: RT60 2 s 에서 검출 0개).

    RT60
        감쇠 구간의 에너지 포락선(dB)에 직접 최소자승 회귀를 걸고 −60 dB 로 외삽한다.
        −3 dB 아래부터 재서 직접음 잔재를 피하고, 10 dB 이상 감쇠한 구간만 채택한다.
        구간마다 흔들리므로 **중앙값**으로 집계한다(이상치에 강함).
        포락선은 중앙값 필터(10 ms)로 평활화한다 — 이동평균을 쓰면 오프셋 엣지가
        뭉개져 리버브가 전혀 없는 드라이 신호에 감쇠 구간 22개를 오검출했다.

    Wet
        여기서는 뽑지 않는다. wet 은 `measure_decay_profile` 이 낸 감쇠 곡선을
        타겟으로 `solve_wet_by_decay_profile` 이 내 보컬에서 역산한다.

    검증(정답 알려진 리버브 5종, RT60 0.6~3.0 s): 평균 절대오차 **0.27 s**.
    리버브가 없는 드라이 신호에서는 감쇠 구간 **0개**(오검출 없음).
    실제 보컬 음원에서는 곡당 7개 내외의 감쇠 구간이 잡힌다.

    Returns:
        dict(rt60, n_segments, rt60_all)
    """
    hop = max(1, int(sample_rate * hop_ms * 1e-3))
    win = hop * 2

    x = np.asarray(y, dtype=np.float64)
    if x.size < win * 4:
        return {"rt60": 0.0, "wet": 0.0, "n_segments": 0, "rt60_all": [], "wet_all": []}

    n_frames = (x.size - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    energy = (frames**2).mean(axis=1)
    env_db_raw = 10.0 * np.log10(energy + 1e-12)

    # 검출·회귀 모두 **평활화된 포락선**으로 한다.
    # 실제 보컬은 숨소리·자음·노이즈로 포락선이 프레임 단위로 출렁여서, 원본을 그대로
    # 쓰면 "계속 하강" 조건이 즉시 깨진다(실측: 급락 후보 740개 중 채택 1개).
    # **중앙값** 필터를 쓴다. 이동평균은 오프셋의 급격한 엣지를 뭉개서 없던 감쇠를
    # 만들어내고, 실측에서 리버브가 전혀 없는 드라이 신호에 감쇠 구간 22개를
    # 오검출했다. 중앙값은 스파이크만 제거하고 엣지는 보존한다.
    from scipy.signal import medfilt
    k = max(3, int(smooth_ms / hop_ms) | 1)
    env_db = medfilt(env_db_raw, kernel_size=k)

    peak_db = float(env_db.max())
    min_frames = max(4, int(min_decay_ms / hop_ms))
    drop_span = max(2, int(20.0 / hop_ms))  # 20 ms 안에 급락하면 오프셋으로 본다

    # 오프셋 검출은 **급락 검출**로 한다.
    # 절대 임계값(전체 피크 대비 −25 dB 등)을 쓰면 잔향이 긴 곡에서 테일이 임계 아래로
    # 안 내려가 오프셋이 아예 안 잡힌다(실측: RT60 2 s 에서 검출 구간 0개).
    rt60s, tail_ratios = [], []
    i = drop_span
    while i < n_frames - min_frames:
        if env_db[i] < peak_db - 45.0:      # 노이즈 플로어 근처는 건너뜀
            i += 1
            continue
        if env_db[i - drop_span] - env_db[i] < drop_db:   # 급락 아님
            i += 1
            continue

        # 감쇠 구간: 다시 올라갈 때(=다음 음절 시작)까지
        end = i
        run_min = env_db[i]
        while end < n_frames - 1:
            run_min = min(run_min, env_db[end])
            if env_db[end] > run_min + rise_db:
                break
            end += 1

        if end - i >= min_frames:
            # 감쇠 기울기는 **에너지 포락선에 직접** 회귀한다.
            #
            # Schroeder 역적분은 임펄스응답처럼 잔향이 끝까지 관측될 때만 무편향이다.
            # 노래의 음절 간 묵음은 300~500 ms 뿐이라 적분이 중간에 잘리고, 잘린 끝
            # 부근에서 EDC 가 급락해 기울기가 과대평가된다 → RT60 과소추정.
            # 실측에서 정답 2.0 s 가 0.54 s 로 나왔고 오차가 RT60 에 비례해 커졌다.
            # 포락선 직접 회귀는 이 절단 편향이 없다. 구간별로는 더 시끄럽지만
            # 수십 개 구간의 중앙값을 쓰므로 잡음은 상쇄된다.
            seg_db = env_db[i:end]
            rel = seg_db - seg_db[0]
            band = np.where(rel <= -3.0)[0]           # 직접음 잔재 회피
            if band.size >= 3 and rel[band[-1]] <= -depth_db:
                t = band * hop / sample_rate
                slope = np.polyfit(t, rel[band], 1)[0]  # dB/s
                if slope < -1e-3:
                    rt60 = -60.0 / slope
                    if 0.1 <= rt60 <= 6.0:
                        rt60s.append(float(rt60))

            # 잔향 레벨 지문: 오프셋 30 ms 뒤 테일 − 직전 지속 레벨 (dB)
            tail_i = i + max(1, int(30.0 / hop_ms))
            pre_lo = max(0, i - max(2, int(150.0 / hop_ms)))
            if tail_i < end and i - pre_lo >= 2:
                sustain_db = float(np.median(env_db[pre_lo:i]))
                tail_ratios.append(float(env_db[tail_i] - sustain_db))

        i = max(end, i + 1)

    return {
        "rt60": float(np.median(rt60s)) if rt60s else 0.0,
        # tail_ratio 는 wet 을 직접 주지 않는다. 리버브 모델의 RIR 정규화 방식에
        # 따라 같은 wet 이라도 테일 레벨이 달라지기 때문이다. 이 값은 '지문'으로
        # 넘기고, 실제 wet 은 내 보컬에 모델을 적용해 이 지문을 재현하도록 역산한다.
        "tail_ratio_db": float(np.median(tail_ratios)) if tail_ratios else None,
        "n_segments": len(rt60s),
        "rt60_all": rt60s,
        "tail_all": tail_ratios,
    }


def _envelope_db(y, sample_rate=TARGET_SR, hop_ms=5.0, smooth_ms=SMOOTH_MS):
    """평활화된 프레임 에너지 포락선(dB). 구간 검출과 동일한 규약."""
    from scipy.signal import medfilt
    hop = max(1, int(sample_rate * hop_ms * 1e-3))
    win = hop * 2
    x = np.asarray(y, dtype=np.float64)
    n_frames = max(1, (x.size - win) // hop + 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    e = (x[idx] ** 2).mean(axis=1)
    k = max(3, int(smooth_ms / hop_ms) | 1)
    return medfilt(10.0 * np.log10(e + 1e-12), kernel_size=k)


def _find_decay_segments(
    y: np.ndarray,
    sample_rate: int = TARGET_SR,
    hop_ms: float = 5.0,
    min_decay_ms: float = 120.0,
    smooth_ms: float = SMOOTH_MS,
    rise_db: float = 3.0,
    drop_db: float = 6.0,
    **_ignored,
):
    """음절이 끝나고 잔향만 남는 구간들을 찾는다.

    반환: (env_db, hop, [(start, end), ...]) 또는 구간이 없으면 None.
    RT60 추출과 감쇠 프로파일 측정이 **같은 구간 정의**를 쓰도록 공통화한 함수다.
    """
    hop = max(1, int(sample_rate * hop_ms * 1e-3))
    win = hop * 2
    x = np.asarray(y, dtype=np.float64)
    if x.size < win * 4:
        return None

    n_frames = (x.size - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    energy = (x[idx] ** 2).mean(axis=1)
    env_db_raw = 10.0 * np.log10(energy + 1e-12)

    # 중앙값 필터: 이동평균은 오프셋 엣지를 뭉개서 리버브가 없는 드라이 신호에도
    # 감쇠 구간을 만들어낸다(실측 오검출 22개). 중앙값은 엣지를 보존한다.
    from scipy.signal import medfilt
    k = max(3, int(smooth_ms / hop_ms) | 1)
    env_db = medfilt(env_db_raw, kernel_size=k)

    peak_db = float(env_db.max())
    min_frames = max(4, int(min_decay_ms / hop_ms))
    drop_span = max(2, int(20.0 / hop_ms))

    out = []
    i = drop_span
    while i < n_frames - min_frames:
        if env_db[i] < peak_db - 45.0 or env_db[i - drop_span] - env_db[i] < drop_db:
            i += 1
            continue
        end = i
        run_min = env_db[i]
        while end < n_frames - 1:
            run_min = min(run_min, env_db[end])
            if env_db[end] > run_min + rise_db:
                break
            end += 1
        if end - i >= min_frames:
            out.append((i, end))
        i = max(end, i + 1)

    return (env_db, hop, out) if out else None


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


DECAY_LAGS_MS = (20.0, 40.0, 60.0, 90.0, 120.0, 160.0, 200.0, 260.0)


def measure_decay_profile(
    y: np.ndarray,
    sample_rate: int = TARGET_SR,
    hop_ms: float = 5.0,
    lags_ms=DECAY_LAGS_MS,
    segments=None,
    drop_db: float = 4.0,
    **kw,
) -> Optional[np.ndarray]:
    """음절이 끝나는 지점들의 **감쇠 곡선 프로파일**을 낸다. [len(lags_ms)] (dB)

    각 오프셋마다 직전 지속 레벨을 기준(0 dB)으로 잡고, 오프셋 이후 여러 지연
    시점에서의 상대 레벨을 읽는다. 구간별 곡선을 **중앙값**으로 합쳐 하나의
    대표 곡선을 만든다.

    한 점(예: 30 ms)만 재던 이전 방식은 숨소리·자음에 흔들려 wet 역산이 불안정했다
    (합성 검증에서 정답 0.15 에 0.60 이 나오는 식). 곡선 전체를 쓰면 rt60 이 이미
    맞춰진 상태에서 **곡선의 세로 위치**만 wet 에 따라 움직이므로 훨씬 안정적이다.

    Returns:
        프로파일 배열, 또는 감쇠 구간을 못 찾으면 None.
    """
    # `segments` 를 주면 그 위치에서만 잰다.
    #
    # 자기 자신에서 오프셋을 검출하면 **측정하려는 값이 검출 감도를 좌우한다**:
    # wet 이 커질수록 잔향이 음절 사이를 메워 급락이 사라지고, 오프셋이 아예
    # 안 잡혀 프로파일이 None 이 된다(실측: wet 0.25 이상에서 검출 실패).
    # 내 보컬은 드라이 신호를 갖고 있으므로 **드라이에서 오프셋을 확정**하고
    # 그 자리에서 wet 신호를 재면 이 순환이 끊긴다.
    if segments is not None:
        env_db, hop, offsets = segments
        prof_env = _envelope_db(y, sample_rate, hop_ms, kw.get("smooth_ms", SMOOTH_MS))
        n = min(env_db.size, prof_env.size)
        env_db, prof_env = env_db[:n], prof_env[:n]
        env_db = prof_env  # 위치는 드라이 기준, 레벨은 측정 대상 신호에서 읽는다
    else:
        segs = _find_decay_segments(y, sample_rate, hop_ms, drop_db=drop_db, **kw)
        if not segs:
            return None
        env_db, hop, offsets = segs
    lag_frames = [max(1, int(round(l / hop_ms))) for l in lags_ms]
    pre_frames = max(2, int(round(150.0 / hop_ms)))

    rows = []
    for off, end in offsets:
        lo = max(0, off - pre_frames)
        if off - lo < 2:
            continue
        sustain = float(np.median(env_db[lo:off]))
        row = []
        ok = True
        for lf in lag_frames:
            j = off + lf
            if j >= end or j >= env_db.size:
                ok = False
                break
            row.append(float(env_db[j] - sustain))
        if ok:
            rows.append(row)

    if not rows:
        return None
    return np.median(np.asarray(rows), axis=0)


def solve_wet_for_tail_ratio(
    reverb: nn.Module,
    x: torch.Tensor,
    rt60: float,
    target_tail_db: float,
    sample_rate: int = TARGET_SR,
    iters: int = 18,
) -> float:
    """테일 지문(오프셋 30 ms 뒤 잔향 − 직전 지속 레벨, dB) 하나로 wet 을 역산한다.

    `solve_wet_by_decay_profile` 의 단순 버전. 측정점이 하나뿐이라 잡음에 약하지만,
    감쇠 프로파일 전체를 잡지 못하는 음원에서는 이쪽이 값을 내주기도 한다.
    """
    lo, hi = 0.0, 0.69
    best, best_err = 0.0, float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        set_reverb_params(reverb, rt60, mid)
        with torch.no_grad():
            wet_sig = reverb(x)[..., : x.shape[-1]].mean(0).cpu().numpy()
        meas = extract_reverb_params(wet_sig, sample_rate).get("tail_ratio_db")
        if meas is None:
            break
        err = abs(meas - target_tail_db)
        if err < best_err:
            best, best_err = mid, err
        if meas < target_tail_db:
            lo = mid
        else:
            hi = mid
    return best


def solve_wet_by_decay_profile(
    reverb: nn.Module,
    x: torch.Tensor,
    rt60: float,
    target_profile: np.ndarray,
    sample_rate: int = TARGET_SR,
    coarse: int = 12,
    refine: int = 7,
) -> float:
    """내 보컬의 감쇠 프로파일이 레퍼런스와 일치하도록 wet 을 찾는다.

    RT60 은 이미 이식돼 고정이므로 감쇠 **기울기**는 맞춰져 있다. 남은 자유도는
    오프셋 직후 잔향이 얼마나 높이 남느냐 = 곡선의 세로 위치이고, 그게 곧 wet 이다.
    따라서 레퍼런스에서 잰 프로파일을 타겟으로, 내 보컬에 같은 리버브를 걸었을 때
    같은 프로파일이 나오는 wet 을 찾는다.

    이분 탐색 대신 **거친 격자 → 국소 정밀 격자** 2단으로 간다. 프로파일 오차가
    wet 에 대해 항상 단조롭지는 않아서(측정 잡음) 이분 탐색은 국소해에 빠질 수 있다.
    """
    # 오프셋 위치는 **드라이 신호에서 한 번만** 확정한다(위 주석 참조).
    dry_segments = _find_decay_segments(x.detach().cpu().numpy(), sample_rate)
    if dry_segments is None:
        return 0.0

    if target_profile is None or len(target_profile) == 0:
        return 0.0

    def _err(w: float) -> float:
        set_reverb_params(reverb, rt60, w)
        with torch.no_grad():
            sig = reverb(x)[..., : x.shape[-1]].mean(0).cpu().numpy()
        prof = measure_decay_profile(sig, sample_rate, segments=dry_segments)
        if prof is None:
            return float("inf")
        n = min(len(prof), len(target_profile))
        return float(np.abs(prof[:n] - target_profile[:n]).mean())

    grid = np.linspace(0.0, 0.69, coarse)
    errs = [_err(float(w)) for w in grid]
    k = int(np.argmin(errs))
    if not np.isfinite(errs[k]):
        return 0.0

    step = grid[1] - grid[0]
    lo = max(0.0, grid[k] - step)
    hi = min(0.69, grid[k] + step)
    fine = np.linspace(lo, hi, refine)
    fine_errs = [_err(float(w)) for w in fine]
    return float(fine[int(np.argmin(fine_errs))])


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
        max_gain_db: float = 15.0,
        reverb_seconds: float = 1.5,
        reverb_wet_max: float = 0.7,
        attack_ms: float = 0.5,
        release_ms: float = 120.0,
        comp_ratio: float = COMP_RATIO,
        active_modes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
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
            max_gain_db=max_gain_db, hard_clamp=False,
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

    def apply_reverb(
        self, y_dry: torch.Tensor, reverb_amount: float = 1.0
    ) -> torch.Tensor:
        """드라이 모노 [T] → 리버브 통과 스테레오 [2, T]. 리버브 off 면 dry 를 복제한다.

        **그래프를 끊지 않는다.** 렌더(`forward`)와 학습 루프(`LOSS_MEASURE_POINT
        == "post_reverb"`)가 이 함수를 공유해야 "손실이 듣는 소리 = 저장되는 소리"가
        정의상 보장된다. 그래디언트를 끊고 싶은 쪽(`L_decay` 입력)은 호출부에서
        `torch.no_grad()` 로 감싼다 — 여기서 끊으면 안 된다.
        """
        length = y_dry.shape[-1]

        if "reverb" not in self.active:
            return torch.stack([y_dry, y_dry], dim=0)

        if isinstance(self.reverb, MeasuredIRReverb):
            # 측정 IR 은 센드 구조다. 드라이는 IR 을 통과하지 않고, 잔향만 더해진다.
            # 슬라이더는 그 센드량으로 들어간다 — 출력단에서 dry 를 재블렌드하면
            # 안 된다(직접음 이중 합산 → 2.5 ms 어긋남 → 400 Hz 간격 콤필터).
            return self.reverb(y_dry, reverb_amount)[..., :length]

        y_wet = self.reverb(y_dry)  # [2, T + ir]
        y_wet = y_wet[..., :length]

        if reverb_amount != 1.0:
            # 합성 리버브는 내부에서 wet 을 이미 섞으므로, 사용자 슬라이더는
            # dry 와의 재블렌드로 반영한다 (wet=0 → 완전 dry).
            dry2 = torch.stack([y_dry, y_dry], dim=0)
            y_wet = dry2 + reverb_amount * (y_wet - dry2)
        return y_wet

    def forward(
        self,
        x: torch.Tensor,
        eq_amount: float = 1.0,
        comp_amount: float = 1.0,
        reverb_amount: float = 1.0,
    ) -> torch.Tensor:
        """전체 체인. 반환 [2, T] 스테레오 (리버브 off 면 dry 를 복제)."""
        y_dry = self.to_dry(x, eq_amount, comp_amount)
        return self.apply_reverb(y_dry, reverb_amount)


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
    ir_source: str = "instrumental",
    # "measured"(Rec-RIR 측정 IR) / "synth"(레퍼런스에서 추출 후 wet 학습) / "manual"(사용자 지정)
    reverb_mode: str = "measured",
    manual_rt60: float = 1.0,
    manual_wet: float = 0.3,
    inst_path: Optional[str] = None,
    num_bands: int = VOCAL_EQ_BANDS,
    match_amount: float = 1.0,
    match_volume: bool = False,
    max_gain_db: float = 15.0,
    comp_amount: float = 1.0,
    reverb_amount: float = 1.0,
    mode: str = "both",
    # --- E2E 전용 ---
    n_steps: int = 50,
    attack_ms: float = 0.5,
    release_ms: float = 120.0,
    comp_ratio: float = COMP_RATIO,
    lr_eq: float = 0.05,
    lr_comp: float = 0.02,
    lr_reverb: float = 0.05,
    eq_l2: float = 0.1,
    eq_smooth: float = 0.1,
    comp_thresh_weight: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """EQ / Comp / Reverb 를 최적화하고 결과를 렌더링해 저장한다.

    손실은 **모듈별로 분리**되어 있다(`losses.DSPMatchingLoss` 참조). 가중합이 아니므로
    w_tone/w_dyn 같은 항 가중치는 없고, 모듈별 학습률(`lr_eq`/`lr_comp`)이 그 역할을 한다.

    반환 dict 는 `legacy/ddsp.py` 의 `match_eq` 와 키 호환이므로 `main.py` 에서 바로 교체 가능하다.

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
    manual_reverb = reverb_mode == "manual"
    if manual_reverb:
        # 수동 모드에서는 wet 노브가 곧 사용자 값이므로 렌더 단계의 재블렌드
        # (reverb_amount)를 1.0 으로 고정한다. 그대로 두면 wet 스케일러가 두 개가 되어
        # 화면에 찍히는 wet 과 실제로 들리는 잔향량이 어긋난다.
        if verbose and reverb_amount != 1.0:
            print(f"[E2E] 수동 리버브 → reverb_amount {reverb_amount:.2f} 무시하고 1.0 사용")
        reverb_amount = 1.0

    chain = E2EChain(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        num_bands=num_bands,
        max_gain_db=max_gain_db,
        attack_ms=attack_ms,
        release_ms=release_ms,
        comp_ratio=comp_ratio,
        active_modes=active_modes,
        # 수동 모드만 wet 1.0 까지 허용한다(학습 경로는 0.7 로 묶어 과도한 wet 을 막는다).
        reverb_wet_max=1.0 if manual_reverb else 0.7,
    )

    # ---------------- 리버브: 레퍼런스에서 IR 추정 → 이식 ----------------
    # 곡이 다르면 스펙트럼 대조가 의미 없으므로 리버브는 **학습하지 않는다**.
    # 레퍼런스에서 IR 을 추정하는 시점에 레퍼런스의 역할은 끝난다. 추정한 IR 을 내
    # 보컬에 컨볼브하기만 한다.
    reverb_info = {"rt60": 0.0, "wet": 0.0, "n_segments": 0}
    if "reverb" in active_modes and manual_reverb:
        # ---------------- 수동 모드 ----------------
        # 사용자가 RT60 과 wet 을 직접 지정한다. 레퍼런스에서 아무것도 추정하지 않으므로
        # Rec-RIR 추론(최초 약 60초)도, 감쇠 구간 검출도 돌지 않는다. 학습도 없다.
        rt_lo = DifferentiableReverb.RT60_MIN
        rt_hi = DifferentiableReverb.RT60_MAX
        rt60_use = float(np.clip(manual_rt60, rt_lo, rt_hi))
        wet_use = float(np.clip(manual_wet, 0.0, chain.reverb.wet_max))
        if verbose and (rt60_use != manual_rt60 or wet_use != manual_wet):
            print(f"[E2E] 수동 값 클립: rt60 {manual_rt60}→{rt60_use}, wet {manual_wet}→{wet_use}")
        set_reverb_params(chain.reverb, rt60_use, wet_use)
        with torch.no_grad():
            rt_set, wet_set = chain.reverb.get_params()
        reverb_info = {
            "rt60": float(rt_set),
            "wet": float(wet_set),
            "n_segments": 0,
            "reverb_mode": "manual",
            "ir_source": None,
            "manual": True,
        }
        if verbose:
            print(f"[E2E] 수동 리버브: RT60={float(rt_set):.2f}s  wet={float(wet_set)*100:.0f}%")

    elif "reverb" in active_modes:
        # 잔향을 어디서 잴지 선택한다.
        #
        # "instrumental" — 반주 트랙에서 추정. Demucs 를 안 거쳐 아티팩트가 없고,
        #     음악적으로도 내 보컬이 앉을 공간은 반주의 공간이다.
        # "reference"    — 레퍼런스 보컬에서 추정. Rec-RIR 은 speech 로 학습된
        #     모델이라 도메인상 이쪽이 안쪽이다. 다만 Demucs 분리 잔여물(고역
        #     쉬쉬거림)이 묵음에 남아 잔향으로 오인될 위험은 남는다.
        ir_ref = y_ref
        if ir_source == "instrumental" and inst_path:
            try:
                y_inst, _ = librosa.load(inst_path, sr=TARGET_SR, mono=True)
                ir_ref = y_inst
                if verbose:
                    print(f"[E2E] 잔향 측정 대상: 반주 ({os.path.basename(inst_path)})")
            except Exception as e:
                if verbose:
                    print(f"[E2E] 반주 로드 실패({e}) → 레퍼런스에서 측정")
        elif verbose:
            print("[E2E] 잔향 측정 대상: 레퍼런스 보컬")

        # reverb_mode="measured": Rec-RIR 이 추정한 IR 을 그대로 컨볼브한다.
        # 지수감쇠 합성은 모양이 하나로 강제돼 대역별 감쇠·초기반사를 표현하지 못한다.
        if reverb_mode == "measured":
            # Rec-RIR 은 여기서만 임포트한다. 최상단에서 임포트하면 그 의존성
            # (toml / einops / mamba_ssm)이 없을 때 pipeline 모듈 자체가 로드되지
            # 않아, 리버브를 안 쓰는 EQ·컴프 작업까지 통째로 죽는다.
            from recrir_ir import estimate_ir

            # 분석 구간 5초. 16 kHz 추론이 M1 CPU 에서 약 60초 걸린다.
            est = estimate_ir(
                ir_ref, TARGET_SR, target_sr=TARGET_SR,
                analysis_seconds=5.0, verbose=verbose,
            )
            if est is not None:
                chain.reverb = MeasuredIRReverb(est["ir"], sample_rate=TARGET_SR)
                chain.reverb._rt60_info = est["rt60"]
                reverb_info = {
                    "rt60": est["rt60"],
                    "wet": 1.0,          # IR 의 DRR 이 잔향량을 결정한다. 믹싱 없음.
                    "n_segments": 0,     # Rec-RIR 은 감쇠구간 검출을 쓰지 않는다
                    "ir_source": ir_source,
                    "reverb_mode": "measured",
                    # 추정된 IR 자체의 음향 지표. 전부 IR 배열에서 바로 계산한 값이다.
                    "ir_engine": "rec-rir",
                    "ir_ms": est["ir_ms"],
                    "analysis_start_s": est["analysis_start_s"],
                    "edt": est["edt"],
                    "drr_db": est["drr_db"],
                    "c80_db": est["c80_db"],
                    "rt60_low": est["rt60_low"],
                    "rt60_mid": est["rt60_mid"],
                    "rt60_high": est["rt60_high"],
                    "hf_lf_ratio": est["hf_lf_ratio"],
                }
                if verbose:
                    print(f"[E2E] Rec-RIR IR 사용: {est['ir_ms']:.0f}ms, "
                          f"RT60 {est['rt60']:.2f}s (분석 {est['analysis_start_s']:.1f}s~)")
            else:
                active_modes = [m for m in active_modes if m != "reverb"]
                chain.active.discard("reverb")
                if verbose:
                    print("[E2E] Rec-RIR IR 추정 실패 → 리버브 비활성")

        elif (ext := extract_reverb_params(ir_ref, TARGET_SR))["n_segments"] > 0 and ext["rt60"] > 0.0:
            wet_solved = 0.0
            probe = x_train if x_train.shape[0] > 0 else t_raw
            # 1순위: 테일 지문 역산. 실제 음원에서 안정적으로 값을 낸다.
            if ext.get("tail_ratio_db") is not None:
                wet_solved = solve_wet_for_tail_ratio(
                    chain.reverb, probe, ext["rt60"], ext["tail_ratio_db"], TARGET_SR
                )
            # 2순위: 감쇠 프로파일 역산. 합성 검증 정확도는 더 높지만(MAE 0.061),
            # 실제 음원에서 0 으로 떨어지는 경우가 있어 폴백으로만 쓴다.
            if wet_solved <= 0.0:
                target_profile = measure_decay_profile(ir_ref, TARGET_SR)
                if target_profile is not None:
                    wet_solved = solve_wet_by_decay_profile(
                        chain.reverb, probe, ext["rt60"], target_profile, TARGET_SR
                    )
            set_reverb_params(chain.reverb, ext["rt60"], wet_solved)
            reverb_info = {"rt60": ext["rt60"], "wet": wet_solved,
                           "n_segments": ext["n_segments"], "ir_source": ir_source,
                           "tail_ratio_db": ext.get("tail_ratio_db")}
            if verbose:
                print(f"[E2E] 레퍼런스 리버브 추출: RT60={ext['rt60']:.2f}s  "
                      f"wet={wet_solved:.3f}  (감쇠구간 {ext['n_segments']}개)")
        elif reverb_mode != "measured":
            active_modes = [m for m in active_modes if m != "reverb"]
            chain.active.discard("reverb")
            if verbose:
                print("[E2E] 감쇠 구간을 못 찾음 → 리버브 비활성")
    # 리버브에는 **학습 파라미터가 없다** (Rec-RIR IR 을 쓰는 measured 모드).
    #
    # IR 이 직접음·초기반사·테일을 모두 담고 있고 그 비율(DRR)이 곧 레퍼런스 공간이다.
    # 조절할 스칼라가 없으니 rt60 도 wet 도 학습 대상이 아니다. 부수 효과로 rt60 ↔ wet
    # 축퇴("rt60↑+wet↓" 와 "rt60↓+wet↑" 가 EDC 기울기에 거의 같은 효과)가 사라진다.
    #
    # 합성 리버브(reverb_mode == "synth")에서는 종전대로 wet 만 학습한다.
    # 수동 모드는 사용자가 값을 정했으므로 학습하지 않는다 — 학습하면 지정한 값이 덮인다.
    reverb_trainable = (
        "reverb" in active_modes
        and not manual_reverb
        and not isinstance(chain.reverb, MeasuredIRReverb)
    )
    chain.reverb.raw_rt60.requires_grad_(False)      # 추출값/지정값 고정
    chain.reverb.raw_wet.requires_grad_(reverb_trainable)

    # 어떤 손실을 계산할지. 침범 방지는 이제 detach 가 구조적으로 보장하므로, 이 스위치는
    # **순수하게 연산을 아끼기 위한 것**이다 — 움직일 파라미터가 없는 손실은 값도 그래디언트도
    # 쓸 데가 없다. (L_dyn 은 K-weighting IIR 2회 + 4배 리샘플로 곡 전체 기준 스텝당 1.5 초다.)
    train_tone = "eq" in active_modes
    train_dyn = "comp" in active_modes
    train_decay = reverb_trainable

    # 톤/다이내믹 손실을 리버브 뒤에서 잴지. 리버브가 꺼져 있으면 앞뒤가 같으므로 무의미하다.
    measure_post_reverb = (
        LOSS_MEASURE_POINT == "post_reverb" and "reverb" in chain.active
    )
    if verbose and measure_post_reverb:
        print("[E2E] 손실 측정 지점: 리버브 **이후** (EQ·컴프 그래디언트가 리버브를 관통)")
    if verbose and "reverb" in active_modes and not reverb_trainable:
        print("[E2E] 리버브 학습 파라미터 없음 → L_decay 계산 생략")
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
    if reverb_trainable:
        # 합성 리버브 폴백에서만. rt60 은 고정이므로 wet 만 옵티마이저에 넣는다.
        # Rec-RIR IR 모드에서는 리버브가 옵티마이저에 아예 들어가지 않는다 — 다만
        # forward 그래프에는 그대로 남아, EQ·컴프의 그래디언트가 리버브를 통과한다.
        # 그래야 EQ 가 "잔향 테일이 더해질 것"을 알고 최적화된다(losses.py 의 p_out 주석).
        groups.append({"params": [chain.reverb.raw_wet], "lr": lr_reverb})

    target = r_train.view(1, 1, -1)  # [B=1, C=1, T]

    # ---------------- 컴프 학습 구간: 각자의 '가장 큰 15초' ----------------
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

            # 측정 지점과 그래디언트 경로는 LOSS_GRAD_MODE 가 정한다(상수 주석 참조).
            #
            # "selective": 두 손실이 **같은 소리**(컴프 통과 후)를 듣되 경로는 갈라 둔다.
            #     sg 는 forward 값을 바꾸지 않으므로 tone_src·dyn_src·최종 출력이 수치적으로
            #     같고, 그래디언트만 각자 자기 모듈로 간다. 컴프를 두 번 통과시키는데,
            #     다이내믹 쪽은 하이라이트 구간만이라 비용이 작다.
            y_eq = chain.eq_output(x_train)
            has_comp = "comp" in chain.active
            dyn_presliced = False   # split 모드만 컴프 **이전**에 자른다

            if not has_comp:
                tone_src = dyn_full = y_eq
            elif LOSS_GRAD_MODE == "split":
                # 톤은 컴프 이전에서 잰다(컴프의 스펙트럼 변화를 EQ 가 보지 못함)
                tone_src = y_eq
                dyn_in = y_eq.detach()
                if comp_window is not None:
                    dyn_in = dyn_in[comp_window[0]:comp_window[1]]
                    dyn_presliced = True
                dyn_full = chain.comp(dyn_in)
            elif LOSS_GRAD_MODE == "unified":
                # 완전 결합: 하나의 그래프를 두 손실이 공유한다
                tone_src = dyn_full = chain.comp(y_eq)
            else:  # "selective" (기본)
                # 톤: 컴프를 통과시키되 threshold 를 상수로 고정 → θ 로만 흐른다
                tone_src = chain.comp(y_eq, detach_params=True)
                # 다이내믹: 컴프 입력을 고정 → θ_T 로만 흐른다.
                #
                # 컴프는 **곡 전체**에 걸고 출력을 잘라 쓴다. 구간을 먼저 잘라 넣으면
                # 같은 값이 나오지 않는다 — 밸리스틱 재귀가 구간 시작에서 상태 0 으로
                # 리셋되고, 오토 메이크업이 구간 평균 GR 로 계산돼 상수 오프셋이 달라진다
                # (실측 차이 6.7e-2). 그러면 "두 손실이 같은 소리를 듣는다"가 깨진다.
                dyn_full = chain.comp(y_eq.detach())

            # --- 측정 지점: 리버브 뒤로 옮긴다 (LOSS_MEASURE_POINT) ---
            #
            # 여기는 **그래프를 끊지 않는다.** EQ·컴프의 그래디언트가 리버브 컨볼루션을
            # 관통해서 흘러야 "최종 출력 기준으로 EQ·컴프를 고른다"가 성립한다.
            # 아래 L_decay 의 no_grad 블록과는 목적도 경로도 별개다.
            #
            # 손실은 모노에서 계산하고 최종 지표도 `y_processed.mean(axis=0)` 로 재므로
            # 여기서도 채널 평균으로 내린다(측정 IR 은 좌우가 동일해 평균이 곧 원신호).
            if measure_post_reverb:
                shared = tone_src is dyn_full
                tone_src = chain.apply_reverb(tone_src, reverb_amount).mean(dim=0)
                dyn_full = (tone_src if shared
                            else chain.apply_reverb(dyn_full, reverb_amount).mean(dim=0))

            # 자르기는 **리버브 뒤**다. 구간을 먼저 자르고 리버브를 걸면 구간 시작 앞쪽
            # 신호가 만들어 낸 잔향이 통째로 빠져(테일 유입 없음) 구간 앞부분이 실제
            # 출력보다 드라이해진다 — 컴프의 밸리스틱과 같은 이유다.
            dyn_src = (dyn_full if (comp_window is None or dyn_presliced)
                       else dyn_full[comp_window[0]:comp_window[1]])

            # 그래프를 공유하는 것은 "unified" 뿐이다. 나머지는 tone/dyn 이 서로 다른
            # 그래프라 첫 backward 가 그래프를 해제해도 두 번째가 살아 있다.
            retain = (LOSS_GRAD_MODE == "unified") and has_comp and train_tone and train_dyn

            d: Dict = {}

            # --- EQ 쪽: 톤 손실 + EQ 자체 정규화 ---
            if train_tone:
                l_tone, d_tone = criterion.tone_loss(
                    tone_src.unsqueeze(0), target, sample_rate=TARGET_SR
                )
                d.update(d_tone)

                # 하드 클램프(tanh ±max_gain_db)는 상한에 닿기 전까지 저항이 0 이라, 결국
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

            # --- 리버브 쪽 (합성 리버브 폴백에서만 학습 파라미터가 있다) ---
            # 리버브 입력은 항상 끊는다 → L_decay 의 그래디언트가 EQ·컴프로 새지 않는다.
            # (통합 경로에서도 이 분리는 유지한다 — 리버브는 이번 변경 범위 밖이다.)
            if train_decay:
                # 리버브 입력 = 곡 전체 드라이 체인 출력. 어차피 끊어 쓰므로 no_grad 로 만든다
                # (모드마다 tone_src/dyn_src 의 길이·경로가 달라 재사용하지 않는다).
                with torch.no_grad():
                    y_dry_full = chain.comp(y_eq.detach()) if has_comp else y_eq.detach()
                y_rev = chain.reverb(y_dry_full)[..., : y_dry_full.shape[-1]]
                l_decay, d_decay = criterion.decay_loss(
                    y_rev.unsqueeze(0), target, sample_rate=TARGET_SR
                )
                d.update(d_decay)
                l_decay.backward()

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
            reverb_amount=reverb_amount,
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

    # --- UI 지표: 다이내믹 레인지(기본 크레스트 팩터, `DYN_METRIC`) ---
    # 손실이 실제로 최적화하는 양을 그대로 표시한다. 이전에는 legacy 엔진의
    # '음절 피크 편차'를 표시했는데, L_dyn 이 재는 양과 달라서 컴프가 제대로
    # 동작해도 화면 수치가 안 움직이는(혹은 반대로 가는) 혼란이 있었다.
    #
    # **측정 구간은 학습 구간과 같아야 한다.** 하이라이트 15초로 학습하고 곡 전체로
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

    # 크레스트 팩터는 항상 0 이상이라(피크 ≥ RMS) PLR 시절의 "ref ≤ 0 이면 대체" 예외가
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
            "metric": ("Crest Factor (True Peak dBTP - RMS dB)"
                       if criterion.DYN_METRIC == "crest"
                       else "PLR (True Peak dBTP - Integrated LUFS)"),
            "dyn_metric": criterion.DYN_METRIC,
        },
        "reverb_data": {
            "rt60": float(rt60) if reverb_active else 0.0,
            # 측정 IR 은 센드량, 합성 리버브는 dry/wet 블렌드 — 어느 쪽이든
            # 슬라이더가 실제 잔향량에 곱해진다.
            "wet": float(wet) * reverb_amount if reverb_active else 0.0,
            # RT60 은 레퍼런스에서 추출해 고정, wet 은 추출값을 초기값으로 학습
            # (수동 모드에서는 둘 다 사용자 지정값이고 학습은 없다)
            "rt60_extracted": reverb_info.get("rt60", 0.0),
            "wet_init": reverb_info.get("wet", 0.0),
            "n_segments": reverb_info.get("n_segments", 0),
            "rt60_fixed": True,
            "manual": bool(reverb_info.get("manual", False)),
            "reverb_mode": reverb_info.get("reverb_mode", reverb_mode),
            # --- 측정 IR(Rec-RIR) 모드 전용. synth 모드에서는 전부 None/0 이다. ---
            "ir_engine": reverb_info.get("ir_engine"),
            "ir_source": reverb_info.get("ir_source"),
            "ir_ms": reverb_info.get("ir_ms", 0.0),
            "analysis_start_s": reverb_info.get("analysis_start_s"),
            "edt": reverb_info.get("edt", 0.0),
            "drr_db": reverb_info.get("drr_db"),
            "c80_db": reverb_info.get("c80_db"),
            "rt60_low": reverb_info.get("rt60_low", 0.0),
            "rt60_mid": reverb_info.get("rt60_mid", 0.0),
            "rt60_high": reverb_info.get("rt60_high", 0.0),
            "hf_lf_ratio": reverb_info.get("hf_lf_ratio", 0.0),
            "loss": float(last["L_decay_raw"]),
            "error_db": reverb_err,
            "similarity": float(np.clip(100.0 * np.exp(-0.05 * reverb_err), 50.0, 99.5)),
            "active": reverb_active,
        },
        "gate_data": None,
        "e2e_data": {
            "steps": len(history),
            "loss_history": history,
            "final": last,
        },
    }
