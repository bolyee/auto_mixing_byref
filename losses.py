"""
DSPMatchingLoss — EQ / Compressor / Reverb 매칭 손실.

구조: 손실을 하나로 합치지 않는다
--------------------------------
예전에는 세 손실을 가중합으로 묶고(`Total = w1·L_tone + w2·L_dyn + w3·L_decay`),
"어떤 손실이 어떤 모듈을 움직이는가"를 **손실 설계의 불변성**으로만 통제했다. 세 손실이
모두 출력 한 점에서 체인 전체로 흘렀기 때문에, EQ 가 다른 모듈의 손실을 우회적으로
낮추는 것을 막으려면 별도의 정규화 항(anchor)이 필요했다.

지금은 **그래디언트 경로 자체를 끊는다**(`pipeline.py` 의 학습 루프 `match_e2e` 참조.
기본값 `LOSS_GRAD_MODE="selective"` + `LOSS_MEASURE_POINT="post_reverb"`):

    y_eq     = EQ(x, θ)
    tone_src = Rev(Comp(y_eq, sg[θ_T]))       ← L_tone 이 여기서 측정된다
    dyn_src  = Rev(Comp(sg[y_eq], θ_T))       ← L_dyn 이 여기서 측정된다

    L_tone_total = tone_loss(tone_src) + eq_l2 + eq_smooth   → θ 로만 흐름
    L_dyn_total  = dyn_loss(dyn_src)   + comp_thresh         → θ_T 로만 흐름

`sg[·]` 는 stop-gradient(detach)다. **신호 값은 바꾸지 않는다** — 두 손실이 듣는 소리는
실제 최종 출력과 수치적으로 동일하고, 그래디언트만 각자 자기 모듈로 간다. L_tone 의 입력은
tone_src — EQ→컴프→리버브를 모두 통과한 최종 출력이며, 컴프 파라미터만 detach 로
상수화되어 그래디언트는 EQ(θ)로만 흐른다.

각 손실이 특정 모듈만 낮출 수 있다는 성질이 **구조적으로** 보장되므로, 침범 방지 목적의
정규화 항은 더 이상 필요 없다(`comp_anchor` 삭제). 남은 정규화는 모두 모듈 자신의
파라미터 제약이다.

두 손실은 각각 따로 backward 한다. 파라미터 집합이 서로 겹치지 않아 하나의 Adam 에
param group 으로 넣어도 두 옵티마이저와 수학적으로 동일하다. 가중치 w1/w2 는 삭제했다 —
모듈별 학습률(EQ 0.05 / Comp 0.02)이 같은 역할을 이미 하고 있었다.

각 손실이 재는 양
----------------
    L_tone   곡 전체 평균 Mel 포락선 (평가 대역 평균 dB 로 레벨 정렬)
    L_dyn    크레스트 팩터 = True Peak − RMS  (`DYN_METRIC="crest"`, 기본)
             — PLR(= TP − Integrated LUFS)은 `DYN_METRIC="plr"` 로 되돌릴 수 있다.
             σ_ST 보조항은 제거됨.
    L_decay  대역별 EDC 기울기 (Schroeder 역적분) — 리버브 경로용

정렬
----
raw 보컬과 레퍼런스는 **서로 다른 연주**다. 프레임 단위 직접 비교가 불가능하므로 세 손실
모두 통계 도메인으로 계산한다 — 시간평균 포락선, 크레스트 팩터, 감쇠 기울기 분포.

`L_decay` 의 EQ 불변성(EDC 를 윈도 시작값으로 정규화하면 EQ 의 상수 게인이 소거된다)은
그래디언트 분리와 무관하게 여전히 유효하며, 리버브 경로에서 쓰인다.
"""

import weakref
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks

__all__ = ["DSPMatchingLoss", "LoudnessMeter"]

_EPS = 1e-10


def _to_mono(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T] → [B, T]. 채널 다운믹스.

    톤/다이내믹/감쇠 통계는 모두 채널 합산 기준으로 본다. Reverb는 스테레오 [2, T]를
    출력하고 레퍼런스는 모노일 수 있으므로 여기서 표현을 통일한다.
    (스테레오 폭/디코릴레이션 매칭은 이 손실의 범위 밖이다.)
    """
    if x.dim() == 2:  # [B, T] — 이미 모노
        return x
    if x.dim() == 3:  # [B, C, T]
        return x.mean(dim=1)
    raise ValueError(f"오디오 텐서는 [B, T] 또는 [B, C, T] 여야 함. 받은 shape: {tuple(x.shape)}")


def _octave_band_matrix(
    n_freqs: int,
    sample_rate: int,
    n_bands: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    """rfft bin을 옥타브(로그등간격) 대역으로 묶는 [n_bands, n_freqs] 행렬.

    삼각 필터가 아니라 rectangular 그룹핑을 쓴다. EDC는 대역 내 총 에너지의
    시간적 감쇠만 보므로 대역 모양이 정밀할 필요가 없고, 그룹핑이 훨씬 싸다.
    """
    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    edges = torch.logspace(
        torch.log10(torch.tensor(f_min)),
        torch.log10(torch.tensor(f_max)),
        n_bands + 1,
    )
    mat = torch.zeros(n_bands, n_freqs)
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not bool(mask.any()):
            # 대역이 너무 좁아 bin이 안 잡히면 가장 가까운 bin 하나를 할당
            idx = int(torch.argmin(torch.abs(freqs - 0.5 * (lo + hi))))
            mask = torch.zeros_like(freqs, dtype=torch.bool)
            mask[idx] = True
        mat[b] = mask.float()
    return mat


def _weighted_mean_std(
    x: torch.Tensor, w: torch.Tensor, dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """가중 평균/표준편차. w는 음이 아닌 가중치(정규화 불필요)."""
    w_sum = w.sum(dim=dim, keepdim=True).clamp_min(_EPS)
    mean = (x * w).sum(dim=dim, keepdim=True) / w_sum
    var = (w * (x - mean) ** 2).sum(dim=dim, keepdim=True) / w_sum
    return mean.squeeze(dim), var.clamp_min(0.0).sqrt().squeeze(dim)


class LoudnessMeter(nn.Module):
    """ITU-R BS.1770-4 Integrated Loudness(LUFS) + True Peak(dBTP), 미분 가능 구현.

    `pyloudnorm` 은 numpy 라 그래디언트가 흐르지 않는다. 여기서는 **필터 설계만**
    pyloudnorm 에서 가져오고(동일 계수 보장), 적용은 전부 torch 로 한다.

    다이내믹 레인지(DR) 지표 두 가지를 제공한다 — 크레스트 팩터(TP − RMS, 기본)와
    PLR(Peak to Loudness Ratio = TP − Integrated LUFS). 둘 다 순간 최대 피크와 지속
    평균 음압의 격차이며 **전체 게인에 불변**(분자·분모가 같은 dB 만큼 이동)이라
    다이내믹 손실의 앵커로 적합하다. 어느 쪽을 쓸지는 `DSPMatchingLoss.DYN_METRIC`.
    """

    def __init__(self, sample_rate: int = 44100, oversample: int = 4, topk: int = 64):
        super().__init__()
        import pyloudnorm as pyln

        self.sample_rate = sample_rate
        self.oversample = oversample
        self.topk = topk

        meter = pyln.Meter(sample_rate)
        hs = meter._filters["high_shelf"]
        hp = meter._filters["high_pass"]
        # K-weighting: high-shelf → high-pass 2단 biquad
        self.register_buffer("hs_b", torch.tensor(hs.b, dtype=torch.float32), persistent=False)
        self.register_buffer("hs_a", torch.tensor(hs.a, dtype=torch.float32), persistent=False)
        self.register_buffer("hp_b", torch.tensor(hp.b, dtype=torch.float32), persistent=False)
        self.register_buffer("hp_a", torch.tensor(hp.a, dtype=torch.float32), persistent=False)

        # 게이팅 블록: 400 ms, 75% 오버랩 → 홉 100 ms
        self.block_len = int(round(0.400 * sample_rate))
        self.block_hop = int(round(0.100 * sample_rate))
        # 단기 라우드니스: 3 s 블록, 1 s 홉
        self.st_len = int(round(3.0 * sample_rate))
        self.st_hop = int(round(1.0 * sample_rate))

    # ------------------------------------------------------------------ #

    def k_weight(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] → K-weighted [B, T]. `lfilter` 는 미분 가능하다."""
        from torchaudio.functional import lfilter

        y = lfilter(x, self.hs_a, self.hs_b, clamp=False)
        y = lfilter(y, self.hp_a, self.hp_b, clamp=False)
        return y

    def _block_powers(self, y_k: torch.Tensor, length: int, hop: int) -> torch.Tensor:
        """K-weighted 신호를 블록으로 잘라 블록별 평균제곱 z_i 를 낸다. [B, M]"""
        if y_k.shape[-1] < length:
            return (y_k**2).mean(dim=-1, keepdim=True)
        blocks = y_k.unfold(dimension=-1, size=length, step=hop)  # [B, M, L]
        return (blocks**2).mean(dim=-1)

    def integrated_lufs(self, x: torch.Tensor) -> torch.Tensor:
        """BS.1770-4 통합 라우드니스 [B]. 2단 게이팅 포함."""
        y_k = self.k_weight(x)
        z = self._block_powers(y_k, self.block_len, self.block_hop)  # [B, M]
        l = -0.691 + 10.0 * torch.log10(z + _EPS)                    # 블록 라우드니스

        # 1차: 절대 게이트 −70 LUFS.  하드 마스크는 스텝마다 튀어 그래디언트가
        # 불연속이 되므로 sigmoid 소프트 게이트를 쓴다(폭 1 dB, 사실상 계단).
        g_abs = torch.sigmoid((l - (-70.0)) / 1.0)

        # 2차: 상대 게이트 Γ = (1차 게이트 평균 라우드니스) − 10 LU.
        # Γ 자체를 미분하면 이차 효과로 불안정해지므로 detach 한다.
        z_avg = (z * g_abs).sum(-1) / g_abs.sum(-1).clamp_min(_EPS)
        gamma = (-0.691 + 10.0 * torch.log10(z_avg + _EPS) - 10.0).detach()
        g_rel = torch.sigmoid((l - gamma.unsqueeze(-1)) / 1.0)

        g = g_abs * g_rel
        z_gated = (z * g).sum(-1) / g.sum(-1).clamp_min(_EPS)
        return -0.691 + 10.0 * torch.log10(z_gated + _EPS)

    def true_peak_db(self, x: torch.Tensor) -> torch.Tensor:
        """BS.1770-4 True Peak [B], dBTP. 4배 오버샘플 후 피크.

        하드 max 는 그래디언트가 표본 하나에만 흘러 최적화가 불안정하다. 반대로
        상위 k개 평균으로 대체하면 그래디언트는 안정되지만 값이 진짜 피크보다 낮아지고,
        그 편차가 신호의 뾰족한 정도에 따라 달라져 측정량 자체를 왜곡한다.

        그래서 **값은 진짜 max, 그래디언트는 상위 k개 평균**을 쓴다(straight-through):
        forward 값은 정의대로 정확하고, backward 는 k개 표본에 분산된다.
        """
        from torchaudio.functional import resample

        up = resample(x, self.sample_rate, self.sample_rate * self.oversample)
        a = up.abs()
        k = min(self.topk, a.shape[-1])
        soft = torch.topk(a, k, dim=-1).values.mean(dim=-1)   # 그래디언트 담당
        hard = a.amax(dim=-1)                                  # 값 담당
        peak = hard.detach() + (soft - soft.detach())
        return 20.0 * torch.log10(peak + _EPS)

    def plr(self, x: torch.Tensor) -> torch.Tensor:
        """PLR = True Peak(dBTP) − Integrated Loudness(LUFS). [B]"""
        return self.true_peak_db(x) - self.integrated_lufs(x)

    def rms_db(self, x: torch.Tensor) -> torch.Tensor:
        """구간 전체의 단순 RMS [B], dB. K-weighting 도 게이팅도 쓰지 않는다."""
        return 20.0 * torch.log10(torch.sqrt((x ** 2).mean(dim=-1)) + _EPS)

    def crest_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Crest Factor = True Peak(dBTP) − RMS(dB). [B]

        PLR 과 같은 "순간 최대 대 지속 레벨" 지표지만 분모가 다르다. PLR 의 LUFS 는
        K-weighting + 2단 게이팅(절대 −70 LUFS, 상대 −10 LU)을 거치는데, 그 게이팅이
        압축이 깊어질수록 TP 보다 빠르게 떨어져 **세게 누를수록 PLR 이 오히려 커지는**
        비단조성을 만들었다(실측: R=30, thr −18→−50 에서 TP −29.7 dB vs LUFS −33.0 dB).
        RMS 는 그런 문턱이 없어 신호 전체를 그대로 반영한다.
        """
        return self.true_peak_db(x) - self.rms_db(x)

    def short_term_lufs_std(self, x: torch.Tensor) -> torch.Tensor:
        """단기 라우드니스(3 s)의 표준편차 [B] — 곡 안에서 음량이 얼마나 출렁이는지."""
        y_k = self.k_weight(x)
        z = self._block_powers(y_k, self.st_len, self.st_hop)
        l = -0.691 + 10.0 * torch.log10(z + _EPS)
        if l.shape[-1] < 2:
            return torch.zeros(l.shape[0], device=l.device, dtype=l.dtype)
        w = torch.sigmoid((l - (l.max(dim=-1, keepdim=True).values.detach() - 30.0)) / 3.0)
        _, std = _weighted_mean_std(l, w, dim=-1)
        return std


class DSPMatchingLoss(nn.Module):
    """EQ / Compressor / Reverb 매칭 손실. **각 손실은 따로 반환되며 합쳐지지 않는다.**

    사용법:
        loss = DSPMatchingLoss(...)
        l_tone, d1 = loss.tone_loss(tone_src, target)   # θ 로만 흐름
        l_dyn,  d2 = loss.dyn_loss(dyn_src,   target, compressor=comp)
        (l_tone + eq_reg).backward()
        (l_dyn).backward()

    모듈 간 침범(예: EQ 가 L_dyn 을 우회적으로 낮추는 것)은 호출부에서 `detach` 로
    끊는다. 따라서 이 클래스에는 침범 방지용 정규화 항이 없다.

    Args:
        sample_rate:      샘플레이트.
        n_fft, hop_length: 분석 STFT 파라미터. torch.stft / torch.istft 와 동일 규약.
        n_mels:           L_tone 의 mel 밴드 수.
        tone_fmin/tone_fmax:
                          L_tone 의 평가 대역 Ω = {m : tone_fmin ≤ f_c(m) < tone_fmax}.
                          **이 범위는 EQ 밴드 범위와 반드시 같아야 한다** — 어긋나면
                          "EQ 가 만질 수 없는데 손실은 평가하는" 대역이 생겨, 고칠 수
                          없는 오차가 레벨 기준까지 흔든다. 그래서 `pipeline.match_e2e`
                          는 여기에 `VOCAL_EQ_MIN_FREQ` / `VOCAL_EQ_MAX_FREQ` 를 **명시로
                          넘긴다**. 아래 기본값은 이 클래스를 단독으로 쓸 때만 쓰인다.
        tone_norm:        'l1' 또는 'l2'.
        n_octave_bands:   L_decay 의 집계 대역 수.
        decay_f_min/max:  L_decay 대역 범위.
        decay_win_sec:    Schroeder 적분 윈도 길이(초).
        decay_hop_sec:    Schroeder 윈도 홉(초).
        decay_fit_frac:   윈도 중 기울기 회귀에 쓸 앞부분 비율(꼬리는 노이즈).
        comp_thresh_weight:   Comp threshold 가 신호 아래로 내려가는 것을 막는 정규화
                          항 가중치(0이면 비활성). `dyn_loss` 안에 포함된다.
                          ratio 가 3:1 상수로 고정된 뒤로는 threshold 혼자 압축량을
                          감당하므로 바닥에 붙을 위험이 커져 이 항의 역할이 커졌다.
        auto_balance / balance_beta:
                          **현재 미사용.** 세 손실을 가중합으로 묶던 시절, 단위가 다른
                          항들의 스케일을 EMA 로 맞추던 장치다. 손실을 분리한 뒤로는
                          더할 일이 없어 목적이 사라졌다(Adam 은 손실의 상수배에 거의
                          불변이라 어차피 no-op 에 가깝다). 리버브 경로를 다시 합산
                          구조로 되돌릴 가능성이 있어 코드는 남겨 두고 기본값을
                          False 로 두었다. `_balance()` 참조.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 80,
        tone_fmin: float = 100.0,
        tone_fmax: float = 10000.0,
        tone_norm: str = "l1",
        n_octave_bands: int = 8,
        decay_f_min: float = 63.0,
        decay_f_max: float = 16000.0,
        decay_win_sec: float = 0.35,
        decay_hop_sec: float = 0.12,
        decay_fit_frac: float = 0.75,
        comp_thresh_weight: float = 0.0,
        auto_balance: bool = False,   # 미사용(위 주석 참조)
        balance_beta: float = 0.98,
    ) -> None:
        super().__init__()

        if tone_norm not in ("l1", "l2"):
            raise ValueError("tone_norm 은 'l1' 또는 'l2' 여야 함")

        self.sample_rate = sample_rate
        self.auto_balance = auto_balance
        self.balance_beta = balance_beta
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tone_norm = tone_norm
        self.decay_fit_frac = decay_fit_frac
        self.comp_thresh_weight = comp_thresh_weight

        n_freqs = n_fft // 2 + 1

        # --- 상수 버퍼: 학습 파라미터가 아니므로 register_buffer 로 device 따라가게 한다 ---
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        # Mel 필터뱅크 [n_freqs, n_mels] → transpose 해서 [n_mels, n_freqs] 로 보관
        fb = melscale_fbanks(
            n_freqs=n_freqs,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )  # [n_freqs, n_mels]
        self.register_buffer("mel_fb", fb.T.contiguous(), persistent=False)

        # L_tone 마스크: [tone_fmin, tone_fmax) 밖의 mel 밴드 제외.
        # mel 밴드 중심 주파수를 필터뱅크 무게중심으로 근사한다.
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
        centers = (fb * freqs.unsqueeze(1)).sum(0) / fb.sum(0).clamp_min(_EPS)  # [n_mels]
        self.register_buffer(
            "tone_mask",
            ((centers >= tone_fmin) & (centers < tone_fmax)).float(),
            persistent=False,
        )

        # L_decay 옥타브 그룹핑 행렬
        self.register_buffer(
            "band_mat",
            _octave_band_matrix(n_freqs, sample_rate, n_octave_bands, decay_f_min, decay_f_max),
            persistent=False,
        )

        # Schroeder 윈도를 프레임 단위로 환산
        frame_sec = hop_length / sample_rate
        self.decay_win_frames = max(4, int(round(decay_win_sec / frame_sec)))
        self.decay_hop_frames = max(1, int(round(decay_hop_sec / frame_sec)))
        self.frame_sec = frame_sec

        # 기울기 최소자승 회귀용 시간축 (윈도 내 앞 fit_frac 구간만 사용)
        n_fit = max(3, int(round(self.decay_win_frames * decay_fit_frac)))
        self.n_fit = n_fit
        t_fit = torch.arange(n_fit, dtype=torch.float32) * frame_sec  # [n_fit], 단위 초
        self.register_buffer("t_fit", t_fit, persistent=False)
        t_centered = t_fit - t_fit.mean()
        # closed-form LSQ 기울기 = Σ(t−t̄)·y / Σ(t−t̄)²  → 분모를 미리 접어둔 계수
        self.register_buffer(
            "lsq_coef", (t_centered / (t_centered.pow(2).sum() + _EPS)), persistent=False
        )

        # auto_balance 용 EMA 버퍼. **현재 경로에서는 쓰이지 않는다**(손실 분리로 목적 소멸).
        # 되살릴 때를 대비해 남겨 둔다. persistent=True 로 체크포인트에 함께 저장한다
        # (재개 시 스케일이 리셋되면 손실 지형이 튀기 때문).
        self.register_buffer("ema", torch.ones(3), persistent=True)
        self.register_buffer("ema_ready", torch.zeros(1, dtype=torch.bool), persistent=True)

        # 다이내믹 손실용 미터. True Peak / RMS / LUFS 를 모두 제공한다
        # (기본 지표는 크레스트 팩터 = TP − RMS, `DYN_METRIC` 참조).
        self.meter = LoudnessMeter(sample_rate)

        # 타깃 통계 캐시. 레퍼런스는 학습 내내 바뀌지 않는데 매 스텝 다시 계산하면
        # 곡 전체 기준으로 STFT 1회 + K-weighting IIR 2회 + 4배 리샘플이 통째로
        # 반복된다(실측 128s/193s 조합에서 스텝당 약 0.4 초). 같은 텐서로 다시
        # 들어오면 계산 결과를 그대로 돌려준다.
        # weakref 로 동일성을 검증한다 — data_ptr 만 보면 해제된 메모리가 재사용될 때
        # 다른 오디오를 같은 타깃으로 오인할 수 있다.
        self._tgt_cache: Optional[Tuple[Any, Dict[str, Any]]] = None

    def _balance(self, raw: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """[미사용] 세 손실을 각자의 EMA 크기로 나눠 O(1) 스케일로 맞춘다.

        손실을 가중합으로 묶던 시절의 장치다. 지금은 각 손실을 따로 backward 하므로
        단위를 맞출 이유가 없고(Adam 은 손실의 상수배에 거의 불변), 호출되지 않는다.
        리버브 경로를 합산 구조로 되돌릴 경우를 대비해 남겨 둔다.

        원래 이유: L_tone 은 dB, L_decay 는 dB/s 라 원시 크기가 수십 배 차이 난다.
        임의 상수로 나누면(초기 구현에서 /100 을 썼다) 특정 항의 그래디언트가 뭉개져
        앵커가 무력화된다 — 실측에서 L_decay 가 Reverb 에 주는 그래디언트가 L_dyn 의
        1/30 로 밀려 리버브가 '컴프 대용'으로 학습되는 실패가 확인됐다.
        """
        vals = torch.stack([r.detach() for r in raw]).abs().clamp_min(_EPS)
        if not bool(self.ema_ready):
            self.ema.copy_(vals)
            self.ema_ready.fill_(True)
        elif self.training:
            self.ema.mul_(self.balance_beta).add_(vals, alpha=1.0 - self.balance_beta)
        denom = self.ema.clamp_min(_EPS)
        return tuple(r / denom[i] for i, r in enumerate(raw))

    # ------------------------------------------------------------------ #
    # 내부 헬퍼
    # ------------------------------------------------------------------ #

    def _target_stats(self, target_audio: torch.Tensor) -> Dict[str, Any]:
        """이 타깃 텐서에 대한 캐시 딕셔너리를 돌려준다(없으면 새로 만든다).

        키를 채우는 것은 호출부의 몫이다 — 가중치가 0인 항은 아예 계산하지 않으므로
        필요한 값만 lazy 하게 들어간다.
        """
        if self._tgt_cache is not None:
            ref, stats = self._tgt_cache
            if ref() is target_audio:
                return stats
        stats: Dict[str, Any] = {}
        self._tgt_cache = (weakref.ref(target_audio), stats)
        return stats

    def _power_spec(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] → 파워 스펙트로그램 [B, n_freqs, N].

        torch.stft 사용 → 전 구간 미분가능. torch.istft 와 동일한 window/hop 규약이므로
        DSP 체인(STFT→EQ→Comp→iSTFT→Reverb)과 그대로 호환된다.
        center=True 로 두어 librosa 기본 동작과 프레임 정렬을 맞춘다.
        """
        S = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # |S|² — abs() 는 complex 입력에서 미분가능 (0에서만 특이, eps로 회피)
        return S.real.pow(2) + S.imag.pow(2)

    # -------------------------- L_tone -------------------------------- #

    def _loss_tone_from_avg(self, avg_out: torch.Tensor, avg_tgt: torch.Tensor) -> torch.Tensor:
        """이미 시간평균된 파워 스펙트럼 [B, n_freqs] 로 L_tone 을 계산한다.

        왜 별도 경로가 필요한가:
        톤은 **장기 평균 통계**다. 학습 구간(예: 10초)에서만 재면 EQ 가 그 구간에
        과적합되어 곡 전체에서는 오히려 톤이 나빠진다 — 실측에서 10초 학습 시
        전체 곡 L_tone 이 2.419 로, 아무 처리도 안 한 raw(2.369)보다 나빴다.

        EQ 는 모든 프레임에 동일한 게인을 곱하므로 장기 평균에 대한 효과가 정확히 닫힌
        형태로 계산된다:  mean_t |EQ(f)·S(f,t)|² = EQ(f)²·mean_t|S(f,t)|².
        따라서 곡 전체 평균 스펙트럼을 한 번만 구해두고 EQ 게인을 해석적으로 곱하면,
        학습 구간이 짧아도 톤은 곡 전체 기준으로 정렬된다.
        """
        out_db = self._tone_db_from_avg(avg_out)
        tgt_db = self._tone_db_from_avg(avg_tgt)
        mask = self.tone_mask.view(1, -1)
        diff = (out_db - tgt_db) * mask
        denom = mask.sum().clamp_min(_EPS)
        if self.tone_norm == "l1":
            return diff.abs().sum() / (denom * out_db.shape[0])
        return diff.pow(2).sum() / (denom * out_db.shape[0])

    def _tone_db(self, p: torch.Tensor) -> torch.Tensor:
        """파워 스펙트로그램 → mel 포락선 dB [B, n_mels]. 레벨은 평가 대역 기준으로 정렬.

        **총에너지로 나누면 안 된다.** dB 로 펴 보면 그 정규화는 모든 밴드에서 상수
        `C = 10log10(Σ_all S)` 를 빼는 것과 같은데, 이 C 에는

          · 평가에서 제외되는 200 Hz 미만 대역(EQ 가 만지지도 못한다)이 들어가고,
          · 무엇보다 **EQ 가 C 를 조종할 수 있다.**

        한 밴드를 크게 부스트하면 Σ 가 커져 나머지 모든 밴드의 값이 함께 내려간다.
        그래서 "이 대역이 과하다"를 그 대역을 깎아서가 아니라 **다른 대역을 폭발시켜
        상대 비중을 낮추는 것**으로 풀 수 있게 된다. 실측에서 200~400 Hz 가 레퍼런스보다
        4.9 dB 과한데 그 대역 컷은 0 dB 이고 100~200 Hz 에 +21 dB 가 걸리는 해가 나왔다.

        여기서는 **평가 대역(tone_mask) 안의 평균 dB** 를 빼서 레벨을 맞춘다. 전체 게인은
        여전히 공짜(레벨 불변)지만, 한 밴드를 키워도 다른 밴드가 따라 내려가지 않으므로
        각 밴드는 자기 오차를 자기가 고쳐야 한다.
        """
        return self._tone_db_from_avg(p.mean(dim=-1))

    def _tone_db_from_avg(self, avg_power: torch.Tensor) -> torch.Tensor:
        """시간평균된 파워 스펙트럼 [B, n_freqs] → 레벨 정렬된 mel dB [B, n_mels]."""
        mel = torch.matmul(avg_power, self.mel_fb.T)
        db = 10.0 * torch.log10(mel + _EPS)
        mask = self.tone_mask.view(1, -1)
        denom = mask.sum().clamp_min(_EPS)
        return db - (db * mask).sum(dim=-1, keepdim=True) / denom

    def _loss_tone(
        self,
        p_out: torch.Tensor,
        p_tgt: Optional[torch.Tensor] = None,
        tgt_db: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """EQ 앵커: Mel 포락선(= 톤 밸런스) 차이. **체인 출력에서 직접 계산한다.**

        p_out 은 EQ·Compressor·Reverb 를 모두 통과한 신호의 파워 스펙트로그램이다.
        따라서 컴프가 피크를 눌러 생기는 음색 변화, 리버브 테일이 더하는 색깔이
        전부 이 손실에 반영되고, 그 그래디언트가 EQ 로 흘러가 사전 보정된다.
        (EQ 커브를 원본 평균 스펙트럼에 해석적으로 곱하는 방식은 빠르지만 EQ 단독
        효과만 보게 되어 E2E 가 성립하지 않는다.)

        raw 와 레퍼런스는 서로 다른 연주이므로 시간축을 평균으로 접어 **장기 평균
        스펙트럼**끼리 비교한다. 레벨 차이는 평가 대역 평균 dB 로 정렬해 제거하고
        (`_tone_db` 주석 참조) 순수 '밸런스'만 남긴다.

        `tgt_db` 를 주면 타깃 포락선을 다시 계산하지 않는다(레퍼런스는 학습 중 불변).
        """
        out_db = self._tone_db(p_out)
        if tgt_db is None:
            if p_tgt is None:
                raise ValueError("p_tgt 또는 tgt_db 중 하나는 필요하다")
            tgt_db = self._tone_db(p_tgt)
        mask = self.tone_mask.view(1, -1)
        diff = (out_db - tgt_db) * mask
        denom = mask.sum().clamp_min(_EPS)
        if self.tone_norm == "l1":
            return diff.abs().sum() / (denom * out_db.shape[0])
        return diff.pow(2).sum() / (denom * out_db.shape[0])

    def _frame_rms_db(self, p: torch.Tensor) -> torch.Tensor:
        """파워 스펙 → 프레임 RMS(dB) [B, N]. Parseval 기준 상수배는 dB 오프셋이라 무관."""
        frame_power = p.mean(dim=1)  # [B, N]
        return 10.0 * torch.log10(frame_power + _EPS)

    # σ_ST(단기 라우드니스 표준편차) 보조항 사용 여부. **기본 False — 제거 상태.**
    #
    # 원래는 "PLR 은 스칼라 1개라 파라미터를 구속하기 부족하다"는 이유로 넣었는데,
    # 실측에서 이 항이 손실을 지배해 컴프를 망가뜨렸다. raw σ_ST=8.22 vs ref σ_ST=0.71
    # (12배 격차)인 소재에서 보조항이 3.3~5.7, PLR 항이 2.6~3.1 이라 옵티마이저가
    # threshold 를 −38.9 dB 까지 밀어넣었고, 그 지점에서는 전 구간이 threshold 위라
    # 게인이 −29 dB 로 포화돼 컴프가 **감쇠기**로 변질됐다. 그 결과 LUFS 가 게이팅
    # 특성 때문에 True Peak 보다 2.64 dB 더 떨어지며 PLR 이 오히려 **올라갔다**
    # (13.89 → 16.54, 목표 10.75 와 반대 방향).
    #
    # 계산 코드는 남겨 둔다 — 표본 수를 늘리거나 분위수 곡선으로 바꾸는 재도입안이
    # 후속 과제로 남아 있다(§22-B).
    USE_ST_TERM: bool = False
    ST_WEIGHT: float = 0.5

    # 다이내믹 지표. "crest"(기본) 또는 "plr".
    #
    # PLR = TP − Integrated LUFS 를 쓰다가 크레스트 팩터(TP − RMS)로 바꿨다. LUFS 의
    # 게이팅 로직이 압축 깊이에 따라 TP 와 다른 속도로 움직여, threshold 를 최적점보다
    # 깊게 내리면 PLR 이 되레 커지는 비단조성을 만들었기 때문이다. 메이크업 게인은
    # 원인이 아니었다(끄고 재현해도 재상승 그대로).
    DYN_METRIC: str = "crest"

    def _dyn_metric(self, x: torch.Tensor) -> torch.Tensor:
        return (self.meter.crest_factor(x) if self.DYN_METRIC == "crest"
                else self.meter.plr(x))

    def target_dyn_stats(self, y_tgt: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """레퍼런스의 (다이내믹 지표, 단기 LUFS 표준편차). 학습 중 불변이라 캐시 대상이다.

        첫 값은 `DYN_METRIC` 이 고른 지표다 — 기본은 크레스트 팩터(TP − RMS).

        `USE_ST_TERM=False` 면 두 번째 값은 None 이다(계산도 건너뛴다).
        """
        with torch.no_grad():
            st = self.meter.short_term_lufs_std(y_tgt) if self.USE_ST_TERM else None
            return self._dyn_metric(y_tgt), st

    def _loss_dyn_plr(
        self,
        y_out: torch.Tensor,
        y_tgt: Optional[torch.Tensor] = None,
        tgt_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compressor 앵커 (기본): 음원 다이내믹 레인지 오차. 지표는 `DYN_METRIC`.

            crest (기본) = True Peak (dBTP) − RMS (dB)
            plr          = True Peak (dBTP) − Integrated Loudness (LUFS)

        곡의 DR 을 말할 때 쓰는 계열의 지표다(TT Dynamic Range Meter 계열).
        순간 최대 피크와 지속 평균 음압의 격차이며, True Peak 는 두 경우 모두
        ITU-R BS.1770-4 정의(4배 오버샘플)를 따른다.

        기본이 크레스트 팩터인 이유는 `DYN_METRIC` 주석 참조 — PLR 의 분모인 LUFS 가
        K-weighting + 2단 게이팅(절대 −70 LUFS, 상대 −10 LU)을 거치는데, 그 게이팅이
        압축 깊이에 따라 TP 와 다른 속도로 움직여 **깊게 누를수록 PLR 이 되레 커지는**
        비단조성이 나온다. RMS 에는 그 문턱이 없어 단조 감소한다.

        왜 이게 좋은 앵커인가:
        · **레벨 불변** — 전체 게인을 곱하면 True Peak 와 RMS(또는 LUFS)가 같은 dB 만큼
          이동해 차이가 보존된다. 따라서 EQ 나 메이크업 게인이 이 손실을 대신 낮출 수
          없고, 그래디언트가 Compressor 로 간다.
        · **압축이 직접 줄이는 양** — 컴프는 피크를 눌러 평균 대비 격차를 좁힌다.
          threshold / ratio 가 이 값을 가장 직접적으로 움직이는 파라미터다.

        **지표 단독이다.** 예전에는 단기 라우드니스 표준편차 σ_ST 를 보조항으로 더했는데,
        그 항이 손실을 지배해 컴프를 감쇠기로 변질시키는 실패가 확인돼 제거했다
        (`USE_ST_TERM` 주석 참조). 자유도는 ratio 를 3:1 상수로 고정해 threshold 하나로
        줄였으므로, 스칼라 목표 하나로도 구속된다.

        `tgt_stats` 를 주면 타깃 쪽 측정을 건너뛴다. 타깃 측정은 K-weighting IIR 2회와
        4배 리샘플이라 곡 전체 기준으로 스텝당 수백 ms 다.
        """
        if tgt_stats is None:
            if y_tgt is None:
                raise ValueError("y_tgt 또는 tgt_stats 중 하나는 필요하다")
            tgt_stats = self.target_dyn_stats(y_tgt)
        dyn_tgt, st_tgt = tgt_stats

        loss = (self._dyn_metric(y_out) - dyn_tgt).abs().mean()
        if self.USE_ST_TERM and st_tgt is not None:
            st_out = self.meter.short_term_lufs_std(y_out)
            loss = loss + self.ST_WEIGHT * (st_out - st_tgt).abs().mean()
        return loss

    def _band_decay_slopes(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """대역별 EDC 기울기(dB/s)와 가중치를 구한다.

        Schroeder 역적분:  EDC(k) = Σ_{j≥k} E(j)
        임펄스응답이 아니라 연속 음원이므로, 신호를 짧은 윈도로 잘라 각 윈도 안에서
        역적분한다. 역적분 결과는 **정의상 단조 비증가**라 어택 구간이 양의 기울기를
        만들지 않는다 — 이것이 Schroeder 를 쓰는 실질적 이유다.

        ▶ EQ 누수를 막는 핵심 (실측으로 확인된 설계):
          `decay_resolution="bin"` 모드는 **STFT bin 단위로** EDC를 뜬 뒤 기울기를
          구하고, 그 다음에야 대역으로 집계한다. 단일 bin 안에서 EQ는 순수 상수배
          g_f² 이므로, EDC를 윈도 시작값으로 정규화하면 그 상수가 **완전히 소거**된다:

              EDC_db(k) − EDC_db(0)
                = 10log₁₀(g_f²·Σ_{j≥k}E_j) − 10log₁₀(g_f²·Σ_{j≥0}E_j)
                = 10log₁₀(Σ_{j≥k}E_j) − 10log₁₀(Σ_{j≥0}E_j)        ← g_f 소거

          따라서 ∂(slope)/∂g_f ≡ 0 이 되어 EQ 앵커 누수가 **수학적으로 정확히 0**이다.

          반면 `"band"` 모드는 먼저 대역 에너지 Σ_f g_f²·E_f[t] 로 합치므로, 대역 내
          EQ 게인이 불균일하면(31밴드 EQ vs 8옥타브 대역) 대역 에너지의 시간 포락선이
          EQ에 따라 변해 누수가 생긴다. 실측 결과 대역 8개일 때 Reverb/EQ 그래디언트
          비가 2.3배에 불과했고, 24개로 늘려도 5.8배에 그쳤다. "bin" 모드를 기본값으로
          쓰는 이유다. ("band"는 메모리가 빠듯할 때의 근사 옵션.)

        Returns:
            slopes: [B, K, M]  (dB/s, 음수). K = n_freqs("bin") 또는 n_bands("band")
            w:      [B, K, M]  에너지 기반 가중치 (detached)
        """
        e = p  # [B, n_freqs, N] — bin 단위

        N = e.shape[-1]
        W, H = self.decay_win_frames, self.decay_hop_frames
        if N < W:
            # 신호가 윈도보다 짧으면 전체를 한 윈도로 취급
            e = torch.nn.functional.pad(e, (0, W - N))
        win = e.unfold(dimension=-1, size=W, step=H)  # [B, K, M, W]

        # 역방향 누적합 = Schroeder 적분
        edc = torch.flip(torch.cumsum(torch.flip(win, dims=[-1]), dim=-1), dims=[-1])

        # 윈도 시작값으로 정규화 → 0 dB 에서 출발하는 감쇠 곡선.
        # 이 정규화가 L_decay 를 **레벨 불변**으로 만든다(위 수식 참조).
        edc_db = 10.0 * torch.log10(edc + _EPS)
        edc_db = edc_db - edc_db[..., :1]

        # 앞 fit_frac 구간만 회귀에 사용(역적분 꼬리는 수치적으로 불안정)
        y = edc_db[..., : self.n_fit]  # [B, K, M, n_fit]

        # closed-form 최소자승 기울기: slope = Σ(t−t̄)·y / Σ(t−t̄)²
        # y 의 평균 성분은 Σ(t−t̄)=0 이라 자동 소거되므로 y 중심화 불필요.
        slopes = (y * self.lsq_coef).sum(dim=-1)  # [B, K, M], dB/s

        # 가중치: 조용한 구간의 기울기는 노이즈이므로 에너지로 가중.
        # detach 로 가중치 자체에는 그래디언트를 주지 않는다
        # (가중치를 통해 손실을 낮추는 우회 경로 = EQ 누수 경로를 차단).
        w_db = 10.0 * torch.log10(win.mean(dim=-1) + _EPS)
        w_ref = w_db.amax(dim=(-1, -2), keepdim=True)
        w = torch.sigmoid((w_db - (w_ref - 35.0)) / 5.0).detach()

        return slopes, w

    def _agg_decay(self, s: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """bin 단위 기울기를 옥타브 대역으로 집계 → (가중평균, 가중표준편차).

        band_mat 는 0/1 그룹핑 행렬이므로 (band_mat @ (s·w)) / (band_mat @ w) 가 곧
        대역별 가중평균이다. 집계를 기울기 계산 **이후**에 하는 것이 EQ 불변성을
        지키는 핵심이다(대역 에너지로 먼저 합치면 소거가 깨진다).
        """
        num = torch.einsum("kf,bfm->bkm", self.band_mat, s * w)
        den = torch.einsum("kf,bfm->bkm", self.band_mat, w).clamp_min(_EPS)
        return _weighted_mean_std(num / den, den, dim=-1)

    def target_decay_stats(self, p_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """레퍼런스의 대역별 (평균 기울기, 기울기 표준편차). 학습 중 불변."""
        with torch.no_grad():
            s_tgt, w_tgt = self._band_decay_slopes(p_tgt)
            return self._agg_decay(s_tgt, w_tgt)

    def _loss_decay(
        self,
        p_out: torch.Tensor,
        p_tgt: Optional[torch.Tensor] = None,
        tgt_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reverb 앵커: 대역별 EDC 기울기 오차.

        EQ는 각 bin 에 상수 게인을 곱할 뿐이고 EDC 정규화가 그 상수를 소거하므로
        ∂L_decay/∂θ_eq ≡ 0 (bin 모드). Comp 도 감쇠 시간을 늘리지 못한다.
        이 손실을 실제로 낮출 수 있는 파라미터는 Reverb 의 rt60/wet 뿐이다.
        → 세 앵커 중 유일하게 물리적으로 배타적인 앵커다.

        기울기의 **가중 평균**(RT60 에 직접 대응)과 **가중 표준편차**(감쇠의 균질성)를
        대역별로 비교한다. 평균만 쓰면 wet 과 rt60 이 서로 상쇄되는 축퇴가 생긴다.

        `tgt_stats` 를 주면 타깃 쪽 기울기 계산을 건너뛴다(레퍼런스는 학습 중 불변).
        """
        s_out, w_out = self._band_decay_slopes(p_out)
        m_out, sd_out = self._agg_decay(s_out, w_out)

        if tgt_stats is None:
            if p_tgt is None:
                raise ValueError("p_tgt 또는 tgt_stats 중 하나는 필요하다")
            tgt_stats = self.target_decay_stats(p_tgt)
        m_tgt, sd_tgt = tgt_stats

        # 단위는 dB/s (수십~수백 스케일). 임의 상수로 나누지 않고 auto_balance 가
        # 세 항의 스케일을 자동 정규화하도록 맡긴다 — 임의 스케일링은 특정 항의
        # 그래디언트를 뭉개서 앵커를 무력화시킨다(실측으로 확인된 실패 모드).
        return (m_out - m_tgt).abs().mean() + 0.5 * (sd_out - sd_tgt).abs().mean()

    # ------------------------------------------------------------------ #
    # 공개 API — 손실은 항목별로 따로 반환된다 (합치지 않는다)
    # ------------------------------------------------------------------ #

    def _check_sr(self, sample_rate: Optional[int]) -> None:
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(
                f"sample_rate 불일치: 생성자 {self.sample_rate} vs 호출 {sample_rate}"
            )

    def tone_loss(
        self,
        output_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        tone_power_out: Optional[torch.Tensor] = None,
        tone_power_tgt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """EQ 손실. 신호 출처를 가리지 않는 **범용 함수** — 무엇을 넣을지는 호출부가 정한다.

        기본 설정(`LOSS_GRAD_MODE="selective"` + `LOSS_MEASURE_POINT="post_reverb"`)에서
        학습 루프가 넣는 것은 `tone_src` — EQ→컴프→리버브를 모두 통과한 최종 출력이며,
        컴프 파라미터만 detach 로 상수화되어 그래디언트는 EQ(θ)로만 흐른다.
        (`pipeline.py` 의 tone_src 생성부와 아래 `_loss_tone` docstring 참조.)

        Args:
            output_audio: [B, T] 또는 [B, C, T]. 그래프에 연결된 텐서.
            target_audio: 레퍼런스. 같은 텐서로 다시 오면 통계를 캐시에서 재사용한다.
            tone_power_*: 곡 전체 평균 파워 스펙트럼을 직접 줄 때(과적합 방지 경로).
        """
        self._check_sr(sample_rate)
        y_out = _to_mono(output_audio)
        y_tgt = _to_mono(target_audio)

        if tone_power_out is not None and tone_power_tgt is not None:
            l_tone = self._loss_tone_from_avg(tone_power_out, tone_power_tgt)
        else:
            tgt = self._target_stats(target_audio)
            if "tone_db" not in tgt:
                with torch.no_grad():
                    tgt["tone_db"] = self._tone_db(self._power_spec(y_tgt))
            l_tone = self._loss_tone(self._power_spec(y_out), tgt_db=tgt["tone_db"])

        return l_tone, {"L_tone_raw": float(l_tone.detach())}

    def dyn_loss(
        self,
        output_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        compressor: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compressor 손실. 신호 출처를 가리지 않는 **범용 함수**다.

        학습 루프가 넣는 것은 `dyn_src` — 컴프(+`post_reverb` 기본값에서는 리버브까지)를
        통과한 신호다. 컴프 입력이 detach 되어 있으므로 그래디언트는 컴프
        파라미터(threshold)로만 간다.

        `comp_thresh_weight > 0` 이면 threshold 심도 정규화가 여기에 포함된다.
        threshold 를 내리는 쪽은 그래디언트 지렛대가 길어서(활성 프레임 수까지 함께
        늘어난다) 목표 PLR 을 threshold 바닥으로 맞춰 버리기 쉽고, 그 지점에서는 전
        구간이 threshold 위라 컴프가 아니라 포락선 스케일러로 동작한다. ratio 가 3:1
        상수로 고정된 뒤로는 threshold 혼자 압축량을 감당하므로 이 위험이 더 커졌다.
        """
        self._check_sr(sample_rate)
        y_out = _to_mono(output_audio)
        y_tgt = _to_mono(target_audio)

        tgt = self._target_stats(target_audio)
        if "dyn" not in tgt:
            tgt["dyn"] = self.target_dyn_stats(y_tgt)

        l_dyn = self._loss_dyn_plr(y_out, tgt_stats=tgt["dyn"])
        out = {"L_dyn_raw": float(l_dyn.detach())}

        if self.comp_thresh_weight > 0.0 and compressor is not None:
            threshold, _ = compressor.get_params()
            reg_t = (threshold / -60.0) ** 2
            l_dyn = l_dyn + self.comp_thresh_weight * reg_t
            out["L_thresh_reg"] = float(reg_t.detach())

        return l_dyn, out

    def decay_loss(
        self,
        output_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Reverb 손실(대역별 EDC 기울기). 리버브 경로 전용."""
        self._check_sr(sample_rate)
        y_out = _to_mono(output_audio)
        y_tgt = _to_mono(target_audio)

        tgt = self._target_stats(target_audio)
        if "decay" not in tgt:
            with torch.no_grad():
                tgt["decay"] = self.target_decay_stats(self._power_spec(y_tgt))

        l_decay = self._loss_decay(self._power_spec(y_out), tgt_stats=tgt["decay"])
        return l_decay, {"L_decay_raw": float(l_decay.detach())}

    def report(
        self,
        output_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        compressor: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """리포팅용: 세 손실의 원시값을 한 번에 잰다(가중치·스킵 규칙 무시).

        학습에는 쓰지 않는다. 렌더가 끝난 뒤 **최종 출력**에서 1회 호출해 UI 지표를
        채우는 용도다. 학습 루프가 계산하지 않은 항(예: 리버브 비활성 시 L_decay)도
        여기서는 값이 나온다.
        """
        with torch.no_grad():
            _, d_tone = self.tone_loss(output_audio, target_audio, sample_rate)
            _, d_dyn = self.dyn_loss(output_audio, target_audio, sample_rate, compressor)
            _, d_decay = self.decay_loss(output_audio, target_audio, sample_rate)
        return {**d_tone, **d_dyn, **d_decay}
