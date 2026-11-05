"""미분 가능 EQ, 컴프레서, 수동 리버브와 믹싱·마스터링 DSP 모듈."""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import soundfile as sf
from losses import DSPMatchingLoss


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

    The vocal chain passes 200 Hz – 10 kHz (`pipeline.VOCAL_EQ_MIN_FREQ` /
    `VOCAL_EQ_MAX_FREQ`) to match the tone loss evaluation band exactly.
    The mastering EQ passes 20 Hz – 10 kHz: keeping 20 Hz to preserve the
    full-mix kick/bass envelope, while capping at 10 kHz to avoid high-frequency
    codec cutoff artifacts from lossy reference audio.
    """
    def __init__(self, sample_rate, n_fft, num_bands=31, min_freq=20.0, max_freq=20000.0,
                 gain_scale_db=15.0, hard_clamp=True, max_gain_db=None):
        super().__init__()
        if max_gain_db is not None:
            gain_scale_db = max_gain_db
        self.gain_scale_db = float(gain_scale_db)
        self.max_gain_db = self.gain_scale_db  # 하위 호환성 유지용 별칭
        self.hard_clamp = hard_clamp
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_bands = num_bands

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
        # hard_clamp=True: tanh 로 ±gain_scale_db 에 가둔다(마스터링 EQ).
        # hard_clamp=False: 상한 없이 선형. 게인 크기 제어는 손실의 제곱 페널티가 맡는다.
        #   tanh 는 상한에 닿기 전까지 저항이 0 이라 결국 "일단 밀고 벽에서 잘리는"
        #   사후 절단과 같아진다. 기울기 스케일은 원점에서의 tanh 와 동일하게 맞춰
        #   (d/dθ = gain_scale_db) 학습률을 그대로 쓸 수 있게 한다.
        if self.hard_clamp:
            band_gains_db = self.gain_scale_db * torch.tanh(self.theta)
        else:
            band_gains_db = self.gain_scale_db * self.theta
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
    range metric -- crest factor by default, the crest-factor metric).

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
                letting that loss steer the compressor (see the E2E loss path).
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


class DifferentiableLimiter(nn.Module):
    """
    미분 가능한 마스터링 리미터 (Differentiable Mastering Limiter / Maximizer).

    DifferentiableCompressor의 밸리스틱과 Soft-knee를 기반으로 하되,
    마스터링 환경에 맞춰 설계:
    1. Stereo-linked: [ch, T] 스테레오 신호에서 L/R 최대 피크를 감지하여 중앙 이미지 보존
    2. 리미터 파라미터:
       - Ratio: 50.0:1 (벽으로 작동)
       - Attack: 0.1 ms (즉각적인 피크 억제)
       - Release: 80.0 ms (자연스러운 지수 릴리즈)
       - Knee: 1.0 dB (미세 소프트니로 역전파 미분 가능성 확보)
       - Ceiling: -0.3 dBFS (절대 천장)
    3. 맥시마이저 볼륨 드라이브:
       - 천장(Ceiling)을 고정하고 인풋 게인(input_gain_db)을 올려서 천장에 밀어 넣는 구조
    """
    def __init__(self, sample_rate=44100, ceiling_db=-0.3, ratio=50.0,
                 attack_ms=0.1, release_ms=80.0, knee_db=1.0, detector_hop=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.ceiling_db = ceiling_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.knee_db = knee_db
        self.detector_hop = detector_hop
        self.detector_win = detector_hop * 2

        # 밸리스틱 1차 계수
        dt = detector_hop / sample_rate
        self.register_buffer('a_att', torch.tensor(np.exp(-dt / (attack_ms / 1000.0)), dtype=torch.float32))
        self.register_buffer('a_rel', torch.tensor(np.exp(-dt / (release_ms / 1000.0)), dtype=torch.float32))

    def forward(self, x, input_gain_db=0.0, return_gr=False):
        """
        Args:
            x: [T] 또는 [ch, T] 텐서/numpy 배열.
            input_gain_db: 볼륨을 올려서 리미터로 밀어 넣는 인풋 게인 (dB).
            return_gr: True 이면 (out, gr_stats) 반환.
        """
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x

        # 1. 인풋 게인 (볼륨 밀어올리기) - 텐서일 경우 autograd 그래프 유지
        if isinstance(input_gain_db, torch.Tensor):
            drive_lin = torch.pow(10.0, input_gain_db / 20.0)
            x_driven = x_tensor * drive_lin
        elif input_gain_db != 0.0:
            drive_lin = 10.0 ** (input_gain_db / 20.0)
            x_driven = x_tensor * drive_lin
        else:
            x_driven = x_tensor

        orig_dim = x_driven.dim()
        if orig_dim == 1:
            x_in = x_driven.unsqueeze(0)
        else:
            x_in = x_driven

        ch, n = x_in.shape
        hop, win = self.detector_hop, self.detector_win
        pad = (win - (n % hop)) % hop + win
        x_pad = torch.nn.functional.pad(x_in, (0, pad))

        # 스테레오 링크 피크 검출: 채널별 윈도우 중 최대값 -> L/R 채널 간 최대값
        env = x_pad.unfold(-1, win, hop).abs().amax(dim=-1).amax(dim=0)
        epsilon = 1e-8
        env_db = 20.0 * torch.log10(env + epsilon)

        # 소프트니 기반 오버슛 계산 (Threshold = ceiling_db)
        threshold = self.ceiling_db
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
        target_gr_db = -overshoot * (1.0 - 1.0 / self.ratio)

        # 밸리스틱 평활화 (수반 재귀)
        gain_red_db = _OnePoleBallistics.apply(target_gr_db, self.a_att, self.a_rel)

        # 게인 업샘플링 [n]
        gain = torch.pow(10.0, gain_red_db / 20.0)
        gain_up = torch.nn.functional.interpolate(
            gain.view(1, 1, -1), size=x_pad.shape[-1], mode='linear', align_corners=False
        ).view(-1)[:n]

        # L과 R에 동일한 게인 곡선 적용 (Stereo Linked)
        out = x_in * gain_up.unsqueeze(0)

        # 최종 안전 하드 실링 클램프 (소프트니 오버슛 방어)
        ceiling_lin = 10.0 ** (self.ceiling_db / 20.0)
        out = torch.clamp(out, -ceiling_lin, ceiling_lin)

        if orig_dim == 1:
            out = out.squeeze(0)

        if return_gr:
            gr_np = gain_red_db.detach().cpu().numpy()
            active = gr_np < -0.1
            gr = {
                "max_db": float(gr_np.min()) if gr_np.size > 0 else 0.0,
                "mean_db": float(gr_np[active].mean()) if active.any() else 0.0,
                "active": float(active.mean()) if active.size > 0 else 0.0,
                "gr_tensor": gain_red_db,
            }
            if is_numpy:
                return out.detach().cpu().numpy(), gr
            return out, gr

        if is_numpy:
            return out.detach().cpu().numpy()
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
                          vocal_gain_db=0.0, vocal_offset_db=0.0,
                          inst_peak_db=-6.0, headroom_db=-1.0,
                          max_vocal_boost_db=24.0):
    """
    처리된 보컬과 반주(instrumental)를 합쳐 최종 풀믹스를 생성한다.

    - 두 트랙 모두 t=0 시작으로 가정(사용자가 반주에 맞춰 가창) → 짧은 쪽을 0 패딩해 정렬.
    - 반주는 inst_peak_db(기본 -6dBFS)로 피크 정규화해 기준 레벨(bed)을 잡는다.
    - **LUFS 기반 밸런스**: 보컬의 '가장 큰 구간 LUFS'를 반주의 '가장 큰 구간 LUFS'
      대비 (vocal_offset_db + vocal_gain_db) 만큼 위로 자동 정렬한다(ITU-R BS.1770 K-weighting).
      기본값 0.0dB 에서 보컬과 반주의 하이라이트 LUFS가 동일하게 맞춰진다.
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

# 통합 마스터링 루프의 드라이브 게인 Adam 설정. EQ와 LUFS/리미터 손실을 매 반복
# 함께 계산하지만, 경사 분리로 각 손실은 자기 파라미터에만 영향을 준다.
# Adam 의 스텝 크기는 그래디언트 크기와 거의 무관하게 lr 로 고정되므로,
# 도달 가능한 dB 범위 = steps × lr 이다.
#
# 이 파라미터는 dB 그 자체라 raw→dB 증폭이 없다(야코비안 1). EQ 의 θ 가 gain_scale_db
# 배로, 컴프의 raw_threshold 가 60·σ(1-σ) 배로 증폭되는 것과 달리 lr 이 그대로
# dB/step 이라, 드라이브 학습률은 0.1로 별도 설정한다. 음압/리미터 손실은 선택한
# 10초 하이라이트 구간에 대해 계산한다.
#
# EQ 300회와 동일한 횟수로 갱신하며, 300 × 0.1 = 약 30 dB 이동 폭을 확보한다.
# 최종 리미터 감쇄량은 별도의 결정론적 하드 캡 가드로 확인한다.
_DRIVE_STEPS = 300
_DRIVE_LR    = 0.1
# 소프트 페널티가 수렴하지 못한 경우를 위한 결정론적 하드 캡 가드.
_DRIVE_GUARD_STEPS = 80
_DRIVE_GUARD_DB    = 0.25


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
                 match_amount=1.0, gain_scale_db=8.0, ceiling_db=-0.3, eq_iters=300,
                 max_gr_db=3.0, eq_l2=0.5, eq_smooth=0.5, lr_eq=0.05,
                 limiter_ratio=50.0, limiter_attack_ms=0.1, limiter_release_ms=80.0,
                 limiter_knee_db=1.0, max_gain_db=None):
    """
    최종 풀믹스 마스터링: 전체 곡 톤 매칭과 10초 LUFS/리미터 최적화를 한 루프에서 수행.

    1) 20 Hz ~ 10 kHz DDSP EQ로 풀믹스의 멜 스펙트럼 포락선을 레퍼런스 곡에 매칭(hard_clamp=False, L2/스무딩 페널티 0.5, 코덱 절벽 방지).
    2) 통합 역전파에서 톤 손실은 EQ로만, 10초 LUFS/리미터 감쇄 손실은 드라이브 게인으로만 전달.
    3) DifferentiableLimiter(미분 가능한 스테레오 링크 리미터)로 피크와 게인 감쇄를 제어.

    Returns: dict(정보)
    """
    if max_gain_db is not None:
        gain_scale_db = max_gain_db
    n_fft, hop, n_mels, eps = 2048, 512, 80, 1e-8

    mix, _ = librosa.load(mix_path, sr=sr, mono=False)
    ref, _ = librosa.load(ref_path, sr=sr, mono=False)
    mix = _to_stereo(np.asarray(mix, dtype=np.float32))
    ref = _to_stereo(np.asarray(ref, dtype=np.float32))

    # 모노 평균 파워 스펙트럼 [1, n_freqs]
    mix_mag = np.abs(librosa.stft(mix.mean(0), n_fft=n_fft, hop_length=hop))
    ref_mag = np.abs(librosa.stft(ref.mean(0), n_fft=n_fft, hop_length=hop))
    mix_power = torch.tensor(np.mean(mix_mag ** 2, axis=1), dtype=torch.float32).unsqueeze(0)
    ref_power = torch.tensor(np.mean(ref_mag ** 2, axis=1), dtype=torch.float32).unsqueeze(0)

    # 믹싱과 동일한 DSPMatchingLoss 사용 (마스터링 대역: 20 Hz ~ 10 kHz, tone_norm='l1')
    criterion = DSPMatchingLoss(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        tone_fmin=20.0, tone_fmax=10000.0, tone_norm="l1"
    )

    # --- 통합 마스터링: EQ 톤 + LUFS/리미터를 한 학습 루프에서 최적화 ---
    eq = DifferentiableEQ(sample_rate=sr, n_fft=n_fft, num_bands=num_bands,
                          min_freq=20.0, max_freq=10000.0,
                          gain_scale_db=gain_scale_db, hard_clamp=False)
    eq_opt = optim.Adam(eq.parameters(), lr=lr_eq)

    # 톤은 전체 곡 평균 스펙트럼으로, 음압은 각 트랙의 가장 큰 10초로 평가한다.
    # 창 선택은 최적화 전에 고정해, 반복 중 평가 구간이 바뀌며 손실이 튀지 않게 한다.
    win_sec = 10.0
    mix_start, mix_end, mix_lufs = loudest_window(mix, sr, win_sec=win_sec, hop_sec=1.0)
    ref_start, ref_end, ref_lufs = loudest_window(ref, sr, win_sec=win_sec, hop_sec=1.0)
    makeup_requested = max(0.0, ref_lufs - mix_lufs)

    limiter = DifferentiableLimiter(
        sample_rate=sr,
        ceiling_db=ceiling_db,
        ratio=limiter_ratio,
        attack_ms=limiter_attack_ms,
        release_ms=limiter_release_ms,
        knee_db=limiter_knee_db,
    )
    drive_gain = nn.Parameter(torch.tensor([makeup_requested], dtype=torch.float32))
    drive_opt = optim.Adam([drive_gain], lr=_DRIVE_LR)

    # EQ가 적용될 선택 구간을 미리 STFT로 변환한다. 창 양끝에 n_fft 문맥을 포함해
    # 매 반복에서 10초 구간 자체를 잘라 필터링할 때 생기는 경계 왜곡을 줄인다.
    context_start = max(0, mix_start - n_fft)
    context_end = min(mix.shape[1], mix_end + n_fft)
    context = torch.from_numpy(mix[:, context_start:context_end]).float()
    eval_offset = mix_start - context_start
    eval_samples = mix_end - mix_start
    stft_window = torch.hann_window(n_fft)
    context_stft = torch.stft(
        context, n_fft=n_fft, hop_length=hop, window=stft_window,
        center=True, pad_mode="constant", return_complex=True,
    )

    # LUFS target도 미분 가능한 같은 미터로 계산한다. 채널별 에너지는 BS.1770 방식으로
    # 합산하며, loudness 손실 그래프는 EQ 입력에서 detach해 드라이브 게인에만 전달한다.
    ref_eval = torch.from_numpy(ref[:, ref_start:ref_end]).float().unsqueeze(0)
    ref_target_lufs = criterion.meter.integrated_lufs_channels(ref_eval).detach()[0]
    drive_max_db = max(makeup_requested, 0.0) + 6.0

    # 두 손실을 같은 outer loop에서 더해 한 번 역전파한다.
    # Selective gradient: tone → EQ만, LUFS/limiter → 드라이브 게인만.
    total_steps = max(eq_iters, _DRIVE_STEPS)
    for step in range(total_steps):
        eq_opt.zero_grad()
        drive_opt.zero_grad()

        curve_db, band_gains = eq.get_eq_curve_db()
        tone_curve_db = curve_db * match_amount
        if step < eq_iters:
            filt_power = mix_power * torch.pow(10.0, tone_curve_db.unsqueeze(0) / 10.0)
            loss_tone = criterion._loss_tone_from_avg(filt_power, ref_power)
            if eq_l2 > 0.0:
                loss_tone = loss_tone + eq_l2 * torch.mean(band_gains ** 2)
            if eq_smooth > 0.0 and band_gains.numel() > 1:
                loss_tone = loss_tone + eq_smooth * torch.mean(torch.diff(band_gains) ** 2)
        else:
            loss_tone = curve_db.new_zeros(())

        # EQ의 전체 레벨 변화는 중립화해 톤 보정과 음압 조정을 분리한다.
        gains_amp = torch.pow(10.0, tone_curve_db / 20.0)
        mix_power_flat = mix_power.squeeze(0)
        delta_db = 10.0 * torch.log10(
            ((gains_amp ** 2) * mix_power_flat).sum() / (mix_power_flat.sum() + eps) + eps
        )
        applied_curve_db = tone_curve_db - delta_db
        gains_lin = torch.pow(10.0, applied_curve_db / 20.0)

        filtered_context_stft = context_stft * gains_lin.view(1, -1, 1)
        filtered_context = torch.istft(
            filtered_context_stft, n_fft=n_fft, hop_length=hop, window=stft_window,
            center=True, length=context.shape[-1],
        )
        eq_eval = filtered_context[:, eval_offset:eval_offset + eval_samples]

        # 경사 분리: EQ 모양은 tone loss가 정하고, 아래 손실은 drive_gain만 조정한다.
        out_eval, gr_eval = limiter(eq_eval.detach(), input_gain_db=drive_gain, return_gr=True)
        out_lufs = criterion.meter.integrated_lufs_channels(out_eval.unsqueeze(0))[0]
        loss_loudness = (out_lufs - ref_target_lufs) ** 2
        max_gr_tensor = -torch.min(gr_eval["gr_tensor"])
        loss_gr = torch.relu(max_gr_tensor - max_gr_db) ** 2 * 25.0
        loss = loss_tone + loss_loudness + loss_gr
        loss.backward()

        if step < eq_iters:
            eq_opt.step()
        if step < _DRIVE_STEPS:
            drive_opt.step()
            with torch.no_grad():
                drive_gain.clamp_(min=_MIN_MAKEUP_DB, max=drive_max_db)

    # 최종 EQ 파라미터로 선택 구간을 다시 렌더링한 뒤 잔여 과도 감쇄를 제한한다.
    with torch.no_grad():
        curve_db, _ = eq.get_eq_curve_db()
        tone_curve_db = curve_db * match_amount
        gains_amp = torch.pow(10.0, tone_curve_db / 20.0)
        delta_db = 10.0 * torch.log10(
            ((gains_amp ** 2) * mix_power_flat).sum() / (mix_power_flat.sum() + eps) + eps
        )
        applied_curve_db_t = tone_curve_db - delta_db
        gains_lin_t = torch.pow(10.0, applied_curve_db_t / 20.0)
        filtered_context_stft = context_stft * gains_lin_t.view(1, -1, 1)
        filtered_context = torch.istft(
            filtered_context_stft, n_fft=n_fft, hop_length=hop, window=stft_window,
            center=True, length=context.shape[-1],
        )
        eq_eval_final = filtered_context[:, eval_offset:eval_offset + eval_samples]
        for _ in range(_DRIVE_GUARD_STEPS):
            _, gr_guard = limiter(eq_eval_final, input_gain_db=float(drive_gain.item()), return_gr=True)
            if (-gr_guard["max_db"]) <= _GR_HARD_CAP_DB or float(drive_gain.item()) <= _MIN_MAKEUP_DB:
                break
            drive_gain.sub_(_DRIVE_GUARD_DB).clamp_(min=_MIN_MAKEUP_DB)

    with torch.no_grad():
        applied_curve_db = applied_curve_db_t.numpy()
        eq_level_shift_db = float(delta_db)
        gains_lin = np.power(10.0, applied_curve_db / 20.0)
        final_drive_db = float(drive_gain.item())

    # 적용된 EQ 커브 샘플(UI 차트용).
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    sample_idx = np.unique(np.logspace(0, np.log10(len(freq_bins) - 1), 120).astype(int))
    eq_curve_x = [float(freq_bins[i]) for i in sample_idx]
    eq_curve_y = [float(applied_curve_db[i]) for i in sample_idx]

    # 학습된 EQ와 드라이브/리미터를 전체 곡에 적용한다.
    out = np.zeros_like(mix)
    for ch in range(mix.shape[0]):
        S = librosa.stft(mix[ch], n_fft=n_fft, hop_length=hop)
        S = S * gains_lin[:, None]
        out[ch] = librosa.istft(S, hop_length=hop, length=mix.shape[1])
    pre_makeup_tensor = torch.from_numpy(out).float()
    with torch.no_grad():
        out_tensor, gr = limiter(pre_makeup_tensor, input_gain_db=final_drive_db, return_gr=True)
        out = out_tensor.detach().cpu().numpy()
        makeup_db = final_drive_db

    sf.write(out_path, out.T, sr)

    final_lufs = _loudest_lufs(out, sr, win_sec=win_sec)
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
        "makeup_held_back_db": round(float(max(0.0, makeup_requested - makeup_db)), 2),
        "limiter_gr_max_db": round(float(gr["max_db"]), 2),
        "limiter_gr_mean_db": round(float(gr["mean_db"]), 2),
        "limiter_active_pct": round(float(gr["active"]) * 100.0, 1),
        "max_gr_budget_db": max_gr_db,
        "window_sec": win_sec,
        "window_track_s": [round(float(mix_start) / sr, 1), round(float(mix_end) / sr, 1)],
        "window_ref_s": [round(float(ref_start) / sr, 1), round(float(ref_end) / sr, 1)],
        "ref_lufs": round(float(ref_lufs), 2),
        "final_lufs": round(float(final_lufs), 2),
        "ceiling_db": ceiling_db,
        "final_peak_db": round(float(20.0 * np.log10(final_peak + 1e-8)), 2),
    }
