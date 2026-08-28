"""
E2E 파이프라인이 쓰는 미분 가능 DSP 모듈과 믹스·마스터링 단계.

구성
----
· DifferentiableEQ / DifferentiableCompressor / DifferentiableReverb
    `pipeline.E2EChain` 이 단일 그래프로 엮는 세 모듈.
· mix_with_instrumental / master_track
    보컬 매칭 이후 단계. 학습은 없고 결정론적 처리다.

순차(greedy) 방식이던 옛 엔진 — match_eq, 1176 컴프(match_compression_1176),
CrestFactorShaper 등 — 은 이 파일에서 제거했다. 원본 전체는 `legacy/ddsp.py` 에 보관.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import soundfile as sf


class DifferentiableReverb(nn.Module):
    """
    DDSP-inspired differentiable stereo reverb using filtered noise decay.

    Args:
        wet_max: upper bound of the wet ratio, i.e. `wet = wet_max * sigmoid(raw)`.
            The default 0.7 is a guard rail for the *learned* path: matching a
            reference by gradient can push wet towards 1 (fully drowning the dry
            signal) when the decay statistic is noisy, and a vocal that far back is
            never the intended result. The manual path has no such failure mode —
            the user is choosing the value directly — so `pipeline.match_e2e`
            passes 1.0 there to make the full dry..wet range reachable.
    """
    def __init__(self, sample_rate=44100, duration_seconds=1.5, wet_max=0.7):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.num_samples = int(duration_seconds * sample_rate)
        self.wet_max = float(wet_max)

        # Trainable parameters (initialized to moderate reverb)
        self.raw_rt60 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.raw_wet = nn.Parameter(torch.tensor([-1.5], dtype=torch.float32))

        # Static noise buffers for Left and Right channels to ensure stable gradients
        noise_l = torch.randn(self.num_samples)
        noise_r = torch.randn(self.num_samples)
        noise_l = noise_l / (torch.max(torch.abs(noise_l)) + 1e-8)
        noise_r = noise_r / (torch.max(torch.abs(noise_r)) + 1e-8)

        self.register_buffer('noise_l', noise_l)
        self.register_buffer('noise_r', noise_r)

    # rt60 사상 범위. pipeline.set_reverb_params 가 역함수를 취할 때 같은 값을 써야 한다.
    RT60_MIN, RT60_MAX = 0.1, 4.0

    def get_params(self):
        rt60 = self.RT60_MIN + (self.RT60_MAX - self.RT60_MIN) * torch.sigmoid(self.raw_rt60)
        wet = self.wet_max * torch.sigmoid(self.raw_wet)   # Range: [0.0, wet_max]
        return rt60, wet
        
    def forward(self, x, sig_fft=None, n_fft=None):
        rt60, wet = self.get_params()
        
        # decay factor alpha
        alpha = 6.9078 / (rt60 * self.sample_rate)
        
        t = torch.arange(self.num_samples, device=x.device, dtype=torch.float32)
        decay_envelope = torch.exp(-alpha * t)
        
        rir_l = self.noise_l * decay_envelope
        rir_r = self.noise_r * decay_envelope

        # Energy normalize RIR
        rir_l = rir_l / (torch.sqrt(torch.sum(rir_l ** 2)) + 1e-8)
        rir_r = rir_r / (torch.sqrt(torch.sum(rir_r ** 2)) + 1e-8)

        # Pad input to preserve reverb tail
        x_pad = torch.cat([x, torch.zeros(self.num_samples, device=x.device)])
        n_sig = x_pad.shape[-1]
        
        # Perform fast FFT convolution
        if sig_fft is None or n_fft is None:
            n_ker = rir_l.shape[-1]
            n_fft = 2 ** int(np.ceil(np.log2(n_sig + n_ker - 1)))
            sig_fft = torch.fft.rfft(x_pad, n=n_fft)
            
        ker_l_fft = torch.fft.rfft(rir_l, n=n_fft)
        ker_r_fft = torch.fft.rfft(rir_r, n=n_fft)
        
        y_l_fft = sig_fft * ker_l_fft
        y_r_fft = sig_fft * ker_r_fft
        
        y_l = torch.fft.irfft(y_l_fft, n=n_fft)[..., :n_sig]
        y_r = torch.fft.irfft(y_r_fft, n=n_fft)[..., :n_sig]
        
        # Mix dry and wet
        dry_pad = x_pad
        out_l = (1.0 - wet) * dry_pad + wet * y_l
        out_r = (1.0 - wet) * dry_pad + wet * y_r
        
        return torch.stack([out_l, out_r], dim=0)


class DifferentiableEQ(nn.Module):
    """
    DDSP-inspired differentiable graphic equalizer using Gaussian filters.

    Bands are placed at log-uniform spacing over [min_freq, max_freq], so the
    spacing is constant in octaves and the band widths are constant-Q.

    The band range matters more than it looks: it is the parameter domain, so
    anything outside it simply cannot be represented. The vocal chain passes
    200 Hz – 10 kHz (`pipeline.VOCAL_EQ_MIN_FREQ` / `VOCAL_EQ_MAX_FREQ`) to match
    the tone loss evaluation band exactly. The mastering EQ keeps the full
    20 Hz – 20 kHz range, since there the whole spectrum is being matched.
    """
    def __init__(self, sample_rate, n_fft, num_bands=31, min_freq=20.0, max_freq=20000.0, max_gain_db=15.0,
                 hard_clamp=True):
        super().__init__()
        self.hard_clamp = hard_clamp
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_bands = num_bands
        self.max_gain_db = max_gain_db
        
        # 1. Compute frequency bins for STFT
        self.n_freqs = n_fft // 2 + 1
        freq_bins = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
        freq_bins_clean = np.copy(freq_bins)
        if freq_bins_clean[0] == 0:
            freq_bins_clean[0] = freq_bins_clean[1] * 0.25
        
        # Log2 scale of STFT frequencies
        self.x = torch.tensor(np.log2(freq_bins_clean), dtype=torch.float32)
        
        # 2. Log-spaced band center frequencies
        self.bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands)
        self.c = torch.tensor(np.log2(self.bands), dtype=torch.float32)
        
        # 3. Compute bandwidth (spacing in log2 space)
        log_diffs = np.diff(np.log2(self.bands))
        avg_diff = np.mean(log_diffs)
        self.sigma = avg_diff * 0.6
        
        # 4. Precompute the Gaussian filterbank matrix Phi (n_freqs x num_bands)
        Phi = np.zeros((self.n_freqs, num_bands))
        for j in range(num_bands):
            Phi[:, j] = np.exp(-((np.log2(freq_bins_clean) - self.c[j].item()) ** 2) / (2 * (self.sigma ** 2)))
        
        self.register_buffer('Phi', torch.tensor(Phi, dtype=torch.float32))
        self.theta = nn.Parameter(torch.zeros(num_bands, dtype=torch.float32))
        
    def get_eq_curve_db(self):
        # hard_clamp=True: tanh 로 ±max_gain_db 에 가둔다(legacy / 마스터링 EQ).
        # hard_clamp=False: 상한 없이 선형. 게인 크기 제어는 손실의 제곱 페널티가 맡는다.
        #   tanh 는 상한에 닿기 전까지 저항이 0 이라 결국 "일단 밀고 벽에서 잘리는"
        #   사후 절단과 같아진다. 기울기 스케일은 원점에서의 tanh 와 동일하게 맞춰
        #   (d/dθ = max_gain_db) 학습률을 그대로 쓸 수 있게 한다.
        if self.hard_clamp:
            band_gains_db = self.max_gain_db * torch.tanh(self.theta)
        else:
            band_gains_db = self.max_gain_db * self.theta
        eq_curve_db = torch.matmul(self.Phi, band_gains_db)
        return eq_curve_db, band_gains_db

    def forward(self, magnitude_spectrum):
        eq_curve_db, _ = self.get_eq_curve_db()
        gains = torch.pow(10.0, eq_curve_db / 20.0)
        if magnitude_spectrum.dim() == 2:
            gains = gains.unsqueeze(1)
        return magnitude_spectrum * gains


class _OnePoleBallistics(torch.autograd.Function):
    """Attack/release one-pole smoothing with an analytic backward pass.

    The recursion is
        y[t] = a[t]*y[t-1] + (1 - a[t])*g[t],    y[-1] = 0
        a[t] = a_att  if g[t] < y[t-1]  else  a_rel

    Running this as a Python loop over autograd-tracked tensors builds a graph as
    deep as the frame count, which makes both forward and backward blow up: a
    15-second training segment (~1290 frames) at 200 optimizer steps measured at
    34 minutes on CPU, versus 16 seconds for 10 seconds / 60 steps. The cost grew
    super-linearly with graph depth, not with actual arithmetic.

    Here the forward recursion runs outside autograd on plain floats and only the
    chosen coefficients are saved. The branch `g[t] < y[t-1]` is a discrete
    selection with zero gradient almost everywhere, so freezing it for the
    backward pass is exact, not an approximation. The backward is then the
    adjoint linear recursion

        s[t]    = grad_out[t] + a[t+1]*s[t+1]
        grad[t] = (1 - a[t])*s[t]

    which costs O(T) time and O(T) memory instead of a T-deep graph.
    """

    @staticmethod
    def forward(ctx, target_gr, a_att, a_rel):
        g = target_gr.detach().cpu().numpy().astype(np.float64)
        att = float(a_att)
        rel = float(a_rel)

        n = g.shape[0]
        y = np.empty(n, dtype=np.float64)
        a = np.empty(n, dtype=np.float64)

        y_prev = 0.0
        for t in range(n):
            gt = g[t]
            at = att if gt < y_prev else rel
            y_prev = at * y_prev + (1.0 - at) * gt
            y[t] = y_prev
            a[t] = at

        ctx.save_for_backward(torch.from_numpy(a))
        ctx.dtype = target_gr.dtype
        ctx.device = target_gr.device
        return torch.from_numpy(y).to(device=target_gr.device, dtype=target_gr.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (a_t,) = ctx.saved_tensors
        a = a_t.numpy()
        go = grad_output.detach().cpu().numpy().astype(np.float64)

        n = a.shape[0]
        grad_g = np.empty(n, dtype=np.float64)
        s_next = 0.0
        for t in range(n - 1, -1, -1):
            # s[t] = grad_out[t] + a[t+1]*s[t+1]
            s = go[t] + (a[t + 1] * s_next if t + 1 < n else 0.0)
            grad_g[t] = (1.0 - a[t]) * s
            s_next = s

        grad = torch.from_numpy(grad_g).to(device=ctx.device, dtype=ctx.dtype)
        return grad, None, None


class DifferentiableCompressor(nn.Module):
    """
    Differentiable single-band compressor with a **configurable but constant** ratio
    and fixed ballistics.

    **Threshold is the only trained parameter.** Ratio is a constant (3:1 by
    default) and attack/release are constant buffers.

    Why ratio is not trained. With the previous parameterisation
    `ratio = 1 + exp(raw)` the derivative is `d(ratio)/d(raw) = ratio - 1`, which
    vanishes exactly where the compressor stops doing anything. Bypass (ratio 1)
    was therefore an attractor: once the optimiser drifted towards it, no gradient
    could pull it back. That was patched with a penalty term that pushed ratio away
    from 1, but the penalty only masked the geometry problem. Fixing the ratio
    removes the degenerate direction outright, and the penalty is gone with it.

    Not trained does not mean not adjustable: `ratio` is a plain constant the caller
    picks (`pipeline.COMP_RATIO`, exposed through the API/UI). 1.0 is bypass and large
    values approach a limiter — see `RATIO_MAX` for why the cap is finite.

    A fixed ratio also removes the threshold/ratio trade-off: many (threshold,
    ratio) pairs produce nearly the same dynamic-range reduction, so with both free
    the solution was only weakly identified by a single scalar target (the dynamic
    range metric -- crest factor by default, see `losses.DSPMatchingLoss.DYN_METRIC`).

    Two design points differ from a naive frame-wise implementation:

    1. Ballistics. Gain reduction is smoothed with a one-pole filter -- the
       attack coefficient while the gain is being pulled down, the release
       coefficient while it recovers. Without this the compressor has no time
       constants at all and behaves like gain automation rather than a
       compressor.

    2. Peak detection in the time domain (see `detector_hop`), not STFT frame RMS,
       because the matching target (crest factor = true peak - RMS) is peak-based.
    """

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512,
                 attack_ms=0.5, release_ms=120.0, knee_db=4.0, makeup=True,
                 detector_hop=128, ratio=3.0, init_threshold_db=-30.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1
        self.knee_db = knee_db
        self.makeup = makeup

        # Peak detector resolution. 128 samples ~= 2.9 ms at 44.1 kHz.
        #
        # This must be a *peak* detector, not the frame RMS of an STFT. Measured on a
        # full 128 s vocal, an RMS-detector version pulled integrated loudness down by
        # 17.4 dB while true peak fell only 13.8 dB, so the peak-to-loudness ratio went
        # *up* (14.2 -> 17.8 dB) even under heavy gain reduction. An RMS detector at an
        # 11.6 ms STFT hop simply cannot see short transients: it compresses sustained
        # material and leaves the peaks, which is the opposite of what is needed to
        # match a target dynamic range. (The numbers above were measured with the
        # older PLR metric, but the failure mode is the same for crest factor: an
        # RMS detector moves the denominator, not the peak.)
        self.detector_hop = detector_hop
        self.detector_win = detector_hop * 2

        # Trainable parameter: threshold only. The seed is given in real units and
        # inverted into raw space here (see `get_params` for the forward mapping),
        # so the starting point reads as "-30 dB" instead of a magic -1.0.
        # threshold = -60*sigmoid(raw) 의 역함수. 경계(0 / -60 dB)에서 logit 이 ±inf 로
        # 발산하므로 안쪽으로 clip 한다 — clip 이 없으면 init_threshold_db=-60 이 0 나눗셈이었다.
        p = float(np.clip(-float(init_threshold_db) / 60.0, 1e-6, 1.0 - 1e-6))
        raw_threshold = float(np.log(p / (1.0 - p)))
        self.raw_threshold = nn.Parameter(torch.tensor(raw_threshold, dtype=torch.float32))

        # Constant ratio: a plain float, not a parameter and not a buffer. Nothing in
        # the graph depends on it being a tensor, and keeping it a float makes it
        # obvious at a glance that no gradient can reach it.
        #
        # The cap is finite on purpose. The gain formula `-o*(1 - 1/R)` handles R = inf
        # cleanly (1/inf = 0), but the value is echoed back in the JSON response, and
        # `Infinity` is not valid JSON — the frontend's JSON.parse would throw. At
        # R = 1000 the factor is 0.999, which is a limiter for every practical purpose.
        self.ratio = float(np.clip(float(ratio), self.RATIO_MIN, self.RATIO_MAX))

        # Fixed ballistics, expressed at the detector rate
        det_rate = sample_rate / detector_hop  # 44100/128 ~= 344.5 Hz
        self.register_buffer('a_att', torch.tensor(
            float(np.exp(-1.0 / max(attack_ms * 1e-3 * det_rate, 1e-6))), dtype=torch.float32))
        self.register_buffer('a_rel', torch.tensor(
            float(np.exp(-1.0 / max(release_ms * 1e-3 * det_rate, 1e-6))), dtype=torch.float32))

    # 설정 가능한 ratio 의 허용 범위. 1.0 = bypass, 상한은 사실상 리미터.
    RATIO_MIN, RATIO_MAX = 1.0, 1000.0

    def get_params(self):
        """(threshold, ratio). threshold 만 학습되고 ratio 는 설정 상수(float)다."""
        threshold = -60.0 * torch.sigmoid(self.raw_threshold)          # range: [-60dB, 0dB]
        return threshold, self.ratio

    def forward(self, x, return_gain_reduction=False, detach_params=False):
        """Compress a time-domain waveform.

        Args:
            x: waveform, shape [T].
            detach_params: if True the threshold is detached, so this call produces
                **the same numbers** but contributes no gradient to the compressor.
                Used to measure the tone loss on the compressed signal without
                letting that loss steer the compressor (see `pipeline.LOSS_GRAD_MODE`).
                Detaching a leaf changes nothing in the forward pass, so the output
                is bit-identical to the non-detached call.
        Returns:
            Compressed waveform [T], or (waveform, gain_reduction_db [1, n_det]).
        """
        threshold, ratio = self.get_params()
        if detach_params:
            threshold = threshold.detach()
        epsilon = 1e-8
        n = x.shape[-1]

        # Peak envelope at the detector rate: max |x| over short overlapping windows.
        hop, win = self.detector_hop, self.detector_win
        pad = (win - (n % hop)) % hop + win
        x_pad = torch.nn.functional.pad(x, (0, pad))
        env = x_pad.unfold(0, win, hop).abs().amax(dim=-1)
        env_db = 20.0 * torch.log10(env + epsilon)

        # Smooth differentiable soft-knee overshoot
        half_knee = self.knee_db / 2.0
        diff = env_db - threshold
        overshoot = torch.where(
            diff < -half_knee,
            torch.zeros_like(diff),
            torch.where(
                diff > half_knee,
                diff,
                (diff + half_knee) ** 2 / (2.0 * self.knee_db)
            )
        )
        target_gr_db = -overshoot * (1.0 - 1.0 / ratio)  # target gain reduction (dB, <= 0)

        # Ballistics: attack/release one-pole smoothing with an analytic backward
        # (see _OnePoleBallistics -- a plain autograd loop here is orders of
        # magnitude slower because of the graph depth).
        gain_red_db = _OnePoleBallistics.apply(target_gr_db, self.a_att, self.a_rel)

        net_gain_db = gain_red_db
        if self.makeup:
            # Auto makeup gain: give back the mean reduction over active regions.
            # A constant dB offset, so it does not affect level-invariant metrics
            # such as crest factor or PLR; it only keeps the output at a usable level.
            active_mask = env_db > (env_db.max() - 45.0)
            if bool(active_mask.any()):
                mean_gr_active = torch.mean(torch.masked_select(gain_red_db, active_mask))
            else:
                mean_gr_active = torch.mean(gain_red_db)
            net_gain_db = gain_red_db - mean_gr_active

        # Upsample the gain curve back to sample rate (linear interpolation keeps it
        # smooth enough that no zipper noise is introduced at this detector rate).
        gain = torch.pow(10.0, net_gain_db / 20.0)
        gain_up = torch.nn.functional.interpolate(
            gain.view(1, 1, -1), size=x_pad.shape[-1], mode='linear', align_corners=False
        ).view(-1)[:n]

        out = x * gain_up
        if return_gain_reduction:
            return out, gain_red_db.unsqueeze(0)
        return out


def compute_stft_crest_factor(stft_complex):
    """
    Computes Crest Factor (Peak to RMS ratio) in dB directly from STFT in a differentiable manner.
    """
    epsilon = 1e-8
    magnitude = torch.abs(stft_complex)
    env = torch.sqrt(torch.mean(magnitude ** 2, dim=0) + epsilon)
    # L10 norm to approximate peak frame RMS smoothly
    peak_env = torch.norm(env, p=10)
    rms_env = torch.sqrt(torch.mean(env ** 2) + epsilon)
    cf = 20.0 * torch.log10(peak_env / (rms_env + epsilon) + epsilon)
    return cf


def compute_stft_rms_variance(stft_complex, n_freqs=None):
    """
    Computes variance of active frames (> -45dB) directly from STFT in a differentiable manner.
    """
    epsilon = 1e-8
    magnitude = torch.abs(stft_complex)
    env = torch.sqrt(torch.mean(magnitude ** 2, dim=0) + epsilon)
    env_db = 20.0 * torch.log10(env + epsilon)
    mask = (env_db > -45.0)
    active_env = torch.masked_select(env_db, mask)
    if len(active_env) <= 1:
        return torch.tensor(0.0, device=stft_complex.device)
    return torch.var(active_env)


def compute_spectral_envelope(audio, sr, n_fft, hop_length, n_mels=80):
    """
    Computes the time-averaged Mel-spectrogram spectral envelope of an audio signal.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    power_spectrum = np.mean(magnitude ** 2, axis=1)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spectrum = np.matmul(mel_fb, power_spectrum)
    return torch.tensor(magnitude, dtype=torch.float32), torch.tensor(mel_spectrum, dtype=torch.float32), torch.tensor(mel_fb, dtype=torch.float32)


def _to_stereo(x):
    """[T] 또는 [1,T]/[2,T] → [2,T] 로 정규화."""
    if x.ndim == 1:
        return np.stack([x, x], axis=0)
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    return x[:2]


def _peak_rms_db(x, sr, win_sec=0.4, hop_sec=0.1):
    """
    가장 큰 RMS 구간(loudest section)의 RMS(dB). 전체 평균이 아니라 최대 에너지 구간 기준.
    win_sec(기본 0.4s) 길이 창의 RMS 중 최댓값 → 순간 스파이크가 아닌 '가장 큰 지속 구간'을
    잡아 보컬/반주의 피크 에너지 순간을 서로 정렬한다.
    """
    m = np.mean(x, axis=0) if x.ndim > 1 else x
    frame = max(1024, int(win_sec * sr))
    hop = max(256, int(hop_sec * sr))
    if len(m) < frame:
        return float(20.0 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-8))
    r = librosa.feature.rms(y=np.ascontiguousarray(m), frame_length=frame, hop_length=hop)[0]
    return float(20.0 * np.log10(np.max(r) + 1e-8))


def mix_with_instrumental(vocal_path, inst_path, out_path, sr=44100,
                          vocal_gain_db=0.0, vocal_offset_db=3.0,
                          inst_peak_db=-6.0, headroom_db=-1.0,
                          max_vocal_boost_db=24.0):
    """
    처리된 보컬과 반주(instrumental)를 합쳐 최종 풀믹스를 생성한다.

    - 두 트랙 모두 t=0 시작으로 가정(사용자가 반주에 맞춰 가창) → 짧은 쪽을 0 패딩해 정렬.
    - 반주는 inst_peak_db(기본 -6dBFS)로 피크 정규화해 기준 레벨(bed)을 잡는다.
    - **LUFS 기반 밸런스**: 보컬의 '가장 큰 구간 LUFS'를 반주의 '가장 큰 구간 LUFS'
      대비 (vocal_offset_db + vocal_gain_db) 만큼 위로 자동 정렬한다(ITU-R BS.1770 K-weighting).
      기본 offset +3dB 로 슬라이더를 건드리지 않아도 보컬이 반주 위에 자연스럽게 얹힌다.
      (peak/평균 RMS 매칭은 loudness를 정확히 반영 못해 보컬이 묻히는 문제가 있어 LUFS로 설계.)
    - 합산 후 피크가 헤드룸(기본 -1dBFS)을 넘으면 tanh 소프트 리미터로 클리핑 방지.

    Returns: dict(정보)
    """
    v, _ = librosa.load(vocal_path, sr=sr, mono=False)
    inst, _ = librosa.load(inst_path, sr=sr, mono=False)
    v = _to_stereo(np.asarray(v, dtype=np.float32))
    inst = _to_stereo(np.asarray(inst, dtype=np.float32))

    # 길이 정렬 (t=0 기준, 짧은 쪽 0 패딩)
    T = max(v.shape[1], inst.shape[1])
    if v.shape[1] < T:
        v = np.pad(v, ((0, 0), (0, T - v.shape[1])))
    if inst.shape[1] < T:
        inst = np.pad(inst, ((0, 0), (0, T - inst.shape[1])))

    # 반주 피크 정규화 → 기준 레벨(bed)
    inst_peak = float(np.max(np.abs(inst)))
    if inst_peak > 1e-8:
        inst = inst / inst_peak * (10.0 ** (inst_peak_db / 20.0))

    # LUFS 기반 밸런스: 보컬의 '가장 큰 구간 LUFS'를 반주의 '가장 큰 구간 LUFS' + offset 에 맞춤
    # (ITU-R BS.1770 K-weighting. RMS보다 사람이 지각하는 크기에 정확.)
    inst_lufs = _loudest_lufs(inst, sr, win_sec=3.0)
    voc_lufs = _loudest_lufs(v, sr, win_sec=3.0)
    target_voc_lufs = inst_lufs + vocal_offset_db + vocal_gain_db
    voc_scale_db = float(np.clip(target_voc_lufs - voc_lufs,
                                 -max_vocal_boost_db, max_vocal_boost_db))
    v = v * (10.0 ** (voc_scale_db / 20.0))

    mix = v + inst

    # 소프트 리미터 (헤드룸 초과분만 tanh 압축)
    ceiling = 10.0 ** (headroom_db / 20.0)
    peak = float(np.max(np.abs(mix)))
    if peak > ceiling:
        thresh = ceiling * 0.9
        abs_m = np.abs(mix)
        over = abs_m > thresh
        if np.any(over):
            excess = abs_m[over] - thresh
            mix[over] = np.sign(mix[over]) * (thresh + (ceiling - thresh) * np.tanh(excess / (ceiling - thresh)))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, mix.T, sr)

    return {
        "out_path": out_path,
        "duration_sec": T / sr,
        "vocal_gain_db": vocal_gain_db,
        "inst_peak_db": inst_peak_db,
        "inst_lufs": round(inst_lufs, 2),
        "vocal_lufs_before": round(voc_lufs, 2),
        "vocal_lufs_after": round(voc_lufs + voc_scale_db, 2),
        "vocal_offset_db": round((voc_lufs + voc_scale_db) - inst_lufs, 2),
        "vocal_scale_db": round(voc_scale_db, 2),
        "final_peak": float(np.max(np.abs(mix))),
    }


def _overall_rms_db(x):
    """전체 평균 RMS(dB) — 마스터링 라우드니스(적분 라우드니스) 근사용."""
    m = np.mean(x, axis=0) if x.ndim > 1 else x
    return float(20.0 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-8))


def loudest_window(x, sr, win_sec=3.0, hop_sec=1.0):
    """가장 큰 구간을 찾아 `(start, end, lufs)` 를 돌려준다.

    측정은 ITU-R BS.1770(K-weighting + 게이팅)이다. `win_sec` 창을 `hop_sec` 간격으로
    슬라이딩하며 통합 라우드니스를 재고 최댓값 구간을 고른다.

    피크나 단순 RMS 가 아니라 라우드니스로 고르는 이유: 이 구간은 라우드니스 기반 지표
    (마스터링의 LUFS 매칭, 컴프 학습의 다이내믹 지표)를 재거나 최적화하는 데 쓰인다. 고르는 기준과
    재는 기준이 다르면 "가장 큰 대목"의 정의가 단계마다 어긋난다.

    Args:
        x: [T] 또는 [ch, T].
    Returns:
        (start, end, lufs). 신호가 창보다 짧으면 (0, T, 전체 라우드니스).
    """
    import pyloudnorm as pyln
    data = x.T if x.ndim > 1 else x  # pyloudnorm: [samples, channels] 또는 [samples]
    T = data.shape[0]
    meter = pyln.Meter(sr)
    win = int(win_sec * sr)
    hop = max(1, int(hop_sec * sr))
    if T < win:
        try:
            return 0, T, float(meter.integrated_loudness(data))
        except Exception:
            return 0, T, -70.0
    best, best_start = -np.inf, 0
    for start in range(0, T - win + 1, hop):
        seg = data[start:start + win]
        try:
            L = meter.integrated_loudness(seg)
        except Exception:
            continue
        if np.isfinite(L) and L > best:
            best, best_start = L, start
    if not np.isfinite(best):
        return 0, min(win, T), -70.0
    return best_start, best_start + win, float(best)


def _loudest_lufs(x, sr, win_sec=3.0, hop_sec=1.0):
    """가장 큰 구간의 LUFS 값만 필요할 때. `loudest_window` 의 얇은 래퍼."""
    return loudest_window(x, sr, win_sec=win_sec, hop_sec=hop_sec)[2]


_GR_HARD_CAP_DB = 3.5    # 최종 리미터 게인 리덕션 절대 상한


_MIN_MAKEUP_DB  = -12.0  # 메이크업 하한(감쇠 허용). GR 상한을 지키기 위해 필요


def _brickwall_limiter(x, sr, ceiling=0.97, release_ms=80.0, win=256, return_gr=False):
    """
    블록 기반 브릭월 리미터. 각 블록의 피크를 ceiling 이하로 눌러(어택=블록 내 즉시),
    릴리즈는 지수적으로 서서히 게인 복귀 → 펌핑 최소화. 마지막에 하드클립으로 안전 보장.
    x: [ch, T] → return [ch, T]

    return_gr=True 이면 (out, gr) 를 돌려준다. gr 은 리미터가 실제로 얼마나 눌렀는지의
    직접 측정값이다:
        max_db  : 최대 게인 리덕션 (dB, 음수)
        mean_db : 리덕션이 걸린 구간의 평균 (dB, 음수)
        active  : 리덕션이 0.1 dB 이상 걸린 시간 비율

    라우드니스를 맞추려고 메이크업 게인을 무한정 올리면 초과분을 전부 이 리미터가
    흡수하게 되고, 그게 곧 "우악스러운" 소리의 원인이다. 이 값이 그 정도를 재는
    가장 직접적인 지표이므로 마스터링에서 메이크업 상한을 정하는 데 쓴다.
    """
    T = x.shape[1]
    mono_peak = np.max(np.abs(x), axis=0)  # [T]
    n_blocks = int(np.ceil(T / win))
    mp = np.pad(mono_peak, (0, n_blocks * win - T))
    block_peak = mp.reshape(n_blocks, win).max(axis=1)
    g_block = np.minimum(1.0, ceiling / (block_peak + 1e-9))

    # 릴리즈 스무딩: 게인 낮추기(어택)는 즉시, 올리기(릴리즈)는 느리게
    rc = float(np.exp(-(win / sr) / (release_ms / 1000.0)))
    g_s = np.empty_like(g_block)
    cur = 1.0
    for i in range(n_blocks):
        t = g_block[i]
        cur = t if t < cur else rc * cur + (1.0 - rc) * t
        g_s[i] = cur

    centers = np.arange(n_blocks) * win + win // 2
    g_samp = np.interp(np.arange(T), centers, g_s, left=g_s[0], right=g_s[-1])
    out = x * g_samp[None, :]
    np.clip(out, -ceiling, ceiling, out=out)  # 경계 보간 오버슛 대비 안전 클립

    if return_gr:
        gr_db = 20.0 * np.log10(np.maximum(g_samp, 1e-9))
        active = gr_db < -0.1
        gr = {
            "max_db": float(gr_db.min()),
            "mean_db": float(gr_db[active].mean()) if active.any() else 0.0,
            "active": float(active.mean()),
        }
        return out, gr
    return out


def master_track(mix_path, ref_path, out_path, sr=44100, num_bands=15,
                 match_amount=0.8, max_gain_db=8.0, ceiling_db=-0.3, eq_iters=300,
                 max_gr_db=3.0):
    """
    최종 풀믹스 마스터링: 레퍼런스 곡의 톤(EQ)을 따라가고 라우드니스를 맞춘 뒤 브릭월 리미터.

    1) 풀밴드 DDSP EQ 로 풀믹스의 멜 스펙트럼 포락선을 레퍼런스 곡에 매칭(보컬 매칭과 동일 원리,
       단 저역 마스킹 없이 전대역·게인 한계 완화). match_amount 로 적용량 조절.
    2) 레퍼런스의 활성 RMS(라우드니스)에 맞춰 메이크업 게인 적용.
    3) -0.3 dBFS 브릭월 리미터로 피크 제한(빡세게 밀어붙인 뒤 클리핑 방지).

    Returns: dict(정보)
    """
    n_fft, hop, n_mels, eps = 2048, 512, 80, 1e-8

    mix, _ = librosa.load(mix_path, sr=sr, mono=False)
    ref, _ = librosa.load(ref_path, sr=sr, mono=False)
    mix = _to_stereo(np.asarray(mix, dtype=np.float32))
    ref = _to_stereo(np.asarray(ref, dtype=np.float32))

    mix_mag, _, mel_fb = compute_spectral_envelope(mix.mean(0), sr, n_fft, hop, n_mels)
    _, ref_mel, _ = compute_spectral_envelope(ref.mean(0), sr, n_fft, hop, n_mels)

    # --- Full-band EQ match (mix tone → reference tone) ---
    eq = DifferentiableEQ(sample_rate=sr, n_fft=n_fft, num_bands=num_bands, max_gain_db=max_gain_db)
    opt = optim.Adam(eq.parameters(), lr=0.15)
    mix_power = torch.mean(mix_mag ** 2, dim=1)
    ref_mel_norm = ref_mel / (torch.sum(ref_mel) + eps)
    ref_mel_db = 10.0 * torch.log10(ref_mel_norm + eps)

    for _ in range(eq_iters):
        opt.zero_grad()
        curve_db, gains = eq.get_eq_curve_db()
        filt_power = mix_power * torch.pow(10.0, curve_db / 10.0)
        filt_mel = torch.matmul(mel_fb, filt_power)
        filt_mel_norm = filt_mel / (torch.sum(filt_mel) + eps)
        filt_mel_db = 10.0 * torch.log10(filt_mel_norm + eps)
        loss = torch.mean((filt_mel_db - ref_mel_db) ** 2) + 0.01 * torch.mean(torch.diff(gains) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        curve_db, _ = eq.get_eq_curve_db()
        applied_curve_db = (curve_db * match_amount).numpy()

        # --- 레벨 중립화 ---
        # EQ 는 톤 밸런스만 바꿔야 하고 전체 음량은 건드리면 안 된다. 그러지 않으면
        # EQ 부스트가 라우드니스를 올려버려 뒤따르는 메이크업 게인이 할 일이 없어지고
        # (실측: 필요 makeup 4.25 dB 가 0 으로 계산됨), 결과적으로 라우드니스 매칭을
        # 톤 매칭 단계가 대신 하게 된다. 그 부스트가 프레즌스에 몰리면 쏘는 소리가 된다.
        #
        # 곡선 g(f) 를 걸었을 때의 총 파워 변화는 믹스의 평균 파워 스펙트럼 P(f) 로
        # 정확히 계산된다:  ΔdB = 10·log10( Σ g²P / Σ P ).
        # 이 값을 곡선 전체에서 빼면 총 파워 변화가 0 이 된다. 곡선 모양은 그대로고
        # 세로 위치만 이동하므로 톤 매칭 결과는 보존된다.
        g_lin = np.power(10.0, applied_curve_db / 20.0)
        P = mix_power.numpy()
        delta_db = 10.0 * np.log10(
            float(np.sum((g_lin ** 2) * P)) / float(np.sum(P) + eps) + eps
        )
        applied_curve_db = applied_curve_db - delta_db
        eq_level_shift_db = float(delta_db)

        gains_lin = np.power(10.0, applied_curve_db / 20.0)

    # 적용된 마스터 EQ 커브를 로그 간격 ~120점으로 샘플링(UI 차트용)
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    sample_idx = np.unique(np.logspace(0, np.log10(len(freq_bins) - 1), 120).astype(int))
    eq_curve_x = [float(freq_bins[i]) for i in sample_idx]
    eq_curve_y = [float(applied_curve_db[i]) for i in sample_idx]

    out = np.zeros_like(mix)
    for ch in range(mix.shape[0]):
        S = librosa.stft(mix[ch], n_fft=n_fft, hop_length=hop)
        S = S * gains_lin[:, None]
        out[ch] = librosa.istft(S, hop_length=hop, length=mix.shape[1])

    # --- Loudness match to reference (LUFS, ITU-R BS.1770 K-weighting) ---
    # 가장 큰 구간의 LUFS를 라우드니스 지표로 사용. 전체 평균은 레퍼런스의 무음/조용한 구간에
    # 휘둘려 마스터가 부당하게 작아지므로, 후렴 등 가장 큰 대목의 LUFS끼리 정렬한다.
    ref_lufs = _loudest_lufs(ref, sr, win_sec=3.0)
    out_lufs = _loudest_lufs(out, sr, win_sec=3.0)
    # 믹스가 레퍼런스보다 조용하면 키우고, 이미 더 크면 줄이지 않는다(마스터가 더 작아지지 않게).
    makeup_requested = max(0.0, ref_lufs - out_lufs)

    # --- Loudness vs. 리미터 부담의 타협 ---
    # 레퍼런스 LUFS 를 무조건 따라가면 초과분을 전부 브릭월 리미터가 흡수한다.
    # 상용 마스터는 컴프·새추레이션·멀티밴드를 거쳐 그 라우드니스에 도달하는데,
    # 여기서는 리미터 하나가 그 일을 다 하게 되어 뭉개진(우악스러운) 소리가 된다.
    #
    # 리미터의 게인 리덕션을 직접 재서 예산(max_gr_db)을 넘으면 그만큼 메이크업을
    # 되돌린다. 라우드니스를 조금 포기하고 다이내믹을 지키는 쪽을 택한다.
    ceiling = 10.0 ** (ceiling_db / 20.0)
    pre_makeup = out  # 메이크업 적용 전 신호 (반복 시 기준으로 재사용)
    makeup_db = makeup_requested

    def _apply(mk):
        return _brickwall_limiter(
            pre_makeup * (10.0 ** (mk / 20.0)), sr, ceiling=ceiling, return_gr=True
        )

    # 목표는 max_gr_db(기본 3.0 dB). 초과분만큼 메이크업을 되돌리며 수렴시킨다.
    out, gr = _apply(makeup_db)
    for _ in range(12):
        over = (-gr["max_db"]) - max_gr_db
        if over <= 0.05 or makeup_db <= _MIN_MAKEUP_DB:
            break
        # 음수(감쇠)까지 허용한다. 믹스 자체 피크가 이미 실링을 넘는 경우
        # 메이크업 0 으로도 GR 상한을 못 지키기 때문이다(실측 -8.13 dB).
        makeup_db = max(_MIN_MAKEUP_DB, makeup_db - over)
        out, gr = _apply(makeup_db)

    # 하드 캡: 위에서 수렴하지 못했더라도 GR 이 3.5 dB 를 넘기지 않게 강제로 더 깎는다.
    guard = 0
    while (-gr["max_db"]) > _GR_HARD_CAP_DB and makeup_db > _MIN_MAKEUP_DB and guard < 40:
        makeup_db = max(_MIN_MAKEUP_DB, makeup_db - 0.25)
        out, gr = _apply(makeup_db)
        guard += 1

    sf.write(out_path, out.T, sr)

    final_lufs = _loudest_lufs(out, sr, win_sec=3.0)
    final_peak = float(np.max(np.abs(out)))
    return {
        "out_path": out_path,
        "eq_match_amount": match_amount,
        "eq_curve_x": eq_curve_x,
        "eq_curve_y": eq_curve_y,
        "eq_gain_range_db": [round(float(np.min(applied_curve_db)), 1), round(float(np.max(applied_curve_db)), 1)],
        "eq_level_shift_db": round(eq_level_shift_db, 2),   # 중립화로 상쇄한 EQ 자체의 음량 변화
        "makeup_db": round(float(makeup_db), 2),
        "makeup_requested_db": round(float(makeup_requested), 2),
        "makeup_held_back_db": round(float(makeup_requested - makeup_db), 2),
        "limiter_gr_max_db": round(float(gr["max_db"]), 2),
        "limiter_gr_mean_db": round(float(gr["mean_db"]), 2),
        "limiter_active_pct": round(float(gr["active"]) * 100.0, 1),
        "max_gr_budget_db": max_gr_db,
        "ref_lufs": round(float(ref_lufs), 2),
        "final_lufs": round(float(final_lufs), 2),
        "ceiling_db": ceiling_db,
        "final_peak_db": round(float(20.0 * np.log10(final_peak + 1e-8)), 2),
    }


class MeasuredIRReverb(nn.Module):
    """추정된 임펄스 응답을 컨볼브하기만 하는 리버브. **학습 파라미터가 없다.**

    `DifferentiableReverb` 는 `노이즈 × exp(-at)` 로 IR 을 합성한다. 모양이 지수함수
    하나로 강제되므로 대역별 감쇠도, 초기반사도 표현하지 못한다(실측: 100Hz 0.73s vs
    8kHz 0.65s 로 거의 동일, 0 ms 부터 에코 밀도 최대). 물리적으로 완벽히 확산된
    이상적 공간의 소리이고 실재하지 않아 "화장실 같은" 느낌을 준다.

    IR 은 `recrir_ir.estimate_ir` 가 Rec-RIR 로 레퍼런스나 반주에서 추정한 것을 쓴다.
    IR 은 학습하지 않는다 — 레퍼런스가 다른 연주라 수만 개 샘플을 구속할 학습 신호가
    없기 때문이다. 늘어난 것은 자유도가 아니라 측정 해상도다.

    센드(aux) 구조인 이유 — 고역 보존
    ---------------------------------
    IR 전체를 컨볼브해 그것만 출력으로 쓰면 안 된다. Rec-RIR 은 **16 kHz 모델**이라
    IR 에 8 kHz 위 성분이 아예 없다. 컨볼루션은 주파수 영역에서 곱셈이므로
    `출력(f) = 보컬(f) × IR(f)` 이고, `IR(f)` 가 8 kHz 위에서 0 이면 **보컬의 8 kHz
    위가 통째로 0 이 된다.** 잔향만 어두워지는 게 아니라 원본 보컬의 공기감·치찰음이
    사라진다 — 8 kHz 로우패스를 건 것과 같다.

    그래서 드라이는 IR 을 통과시키지 않는다:

        출력 = 보컬 + amount × conv(보컬, IR 테일)

    드라이는 44.1 kHz 전대역 그대로 남고, IR 을 거치는 건 잔향 성분뿐이다.
    실제 리버브 플러그인의 send/return 구성과 같다.

    테일은 IR 에서 직접음 피크 이후 `direct_ms` 를 지난 부분으로 정의한다. 직접음을
    빼는 이유는 그게 드라이 경로와 중복되기 때문이다. 남겨 두면 원본과 2.5 ms 어긋난
    복사본이 더해져(`method/pim.py` 가 `argmax - 2.5ms` 부터 자른다) 200 Hz 부터
    400 Hz 간격의 콤필터가 생긴다 — 보컬 바디 한복판이라 바로 들린다.

    IR 이 모노라 좌/우가 같다(듀얼 모노). Rec-RIR 은 단일 채널 RIR 만 추정한다.
    """

    def __init__(self, ir, sample_rate=44100, direct_ms=5.0):
        super().__init__()
        self.sample_rate = sample_rate

        h = torch.as_tensor(ir, dtype=torch.float32).flatten()

        # 직접음 피크 이후 direct_ms 까지를 0 으로 지워 **테일만** 남긴다.
        # 잘라내지 않고 0 으로 두는 이유: 테일의 시작 시점(프리딜레이)이 원본 IR 의
        # 위치 그대로 보존되어야 공간감이 유지된다.
        tail = h.clone()
        k = int(torch.argmax(h.abs()))
        cut = min(k + int(direct_ms * 1e-3 * sample_rate), tail.numel())
        tail[:cut] = 0.0

        self.num_samples = tail.numel()
        self.register_buffer('ir', h)        # 원본 IR (지표·참고용)
        self.register_buffer('tail', tail)   # 실제로 컨볼브되는 잔향 성분

        # DifferentiableReverb 와 인터페이스를 맞추기 위한 자리표시자.
        # 둘 다 학습하지 않는다 — 이 모듈에는 학습 파라미터가 없다.
        self.raw_rt60 = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.raw_wet = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self._rt60_info = 0.0

    def get_params(self):
        """(rt60, wet). wet 은 호출부가 넘기는 amount 가 결정하므로 여기선 1.0."""
        rt60 = torch.tensor(self._rt60_info, device=self.ir.device)
        return rt60, torch.tensor(1.0, device=self.ir.device)

    def forward(self, x, amount: float = 1.0):
        """`amount` = 잔향 센드량. 0 이면 완전 드라이, 1 이면 IR 테일 그대로."""
        # 테일이 잘리지 않도록 IR 길이만큼 패딩한다.
        x_pad = torch.cat([x, torch.zeros(self.num_samples, device=x.device)])

        if amount <= 0.0:
            return torch.stack([x_pad, x_pad], dim=0)

        n_sig = x_pad.shape[-1]
        n_fft = 2 ** int(np.ceil(np.log2(n_sig + self.num_samples - 1)))

        wet = torch.fft.irfft(
            torch.fft.rfft(x_pad, n=n_fft) * torch.fft.rfft(self.tail, n=n_fft), n=n_fft
        )[..., :n_sig]

        y = x_pad + amount * wet
        return torch.stack([y, y], dim=0)
