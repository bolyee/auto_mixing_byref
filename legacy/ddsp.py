import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import soundfile as sf

def fft_convolve(signal, kernel):
    """
    Performs FFT convolution between signal and kernel.
    signal: [L_sig] or [batch, L_sig]
    kernel: [L_ker]
    """
    n_sig = signal.shape[-1]
    n_ker = kernel.shape[-1]
    n_fft = 2 ** int(np.ceil(np.log2(n_sig + n_ker - 1)))
    
    # Do FFT
    sig_fft = torch.fft.rfft(signal, n=n_fft)
    ker_fft = torch.fft.rfft(kernel, n=n_fft)
    
    # Complex multiply
    out_fft = sig_fft * ker_fft
    
    # IFFT
    out = torch.fft.irfft(out_fft, n=n_fft)
    
    # Trim to input signal length to keep audio size consistent
    return out[..., :n_sig]


def interpolate_sorted(x, target_len):
    if len(x) == 0:
        return torch.zeros(target_len, device=x.device, dtype=torch.float32)
    if target_len == 0:
        return torch.zeros(0, device=x.device, dtype=torch.float32)
    if len(x) == target_len:
        return x
    idx = torch.linspace(0, len(x) - 1, target_len, device=x.device)
    idx_floor = torch.floor(idx).long()
    idx_ceil = torch.min(idx_floor + 1, torch.tensor(len(x) - 1, device=x.device))
    weight = idx - idx_floor.float()
    return (1.0 - weight) * x[idx_floor] + weight * x[idx_ceil]


def compute_env_db_pytorch(y, win_len=2048, hop_len=512):
    epsilon = 1e-5
    if len(y) < win_len:
        rms = torch.sqrt(torch.mean(y ** 2) + epsilon).unsqueeze(0)
        return 20.0 * torch.log10(rms)
    frames = y.unfold(0, win_len, hop_len)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + epsilon)
    rms_db = 20.0 * torch.log10(rms)
    return rms_db


def compute_decay_slopes_pytorch(rms_db, hop_len, sample_rate):
    if len(rms_db) <= 1:
        return torch.zeros(0, device=rms_db.device, dtype=torch.float32)
    diff = rms_db[1:] - rms_db[:-1]
    dt = hop_len / sample_rate
    slopes = diff / dt
    mask = (rms_db[:-1] > -50.0) & (slopes < 0.0)
    decay_slopes = torch.masked_select(slopes, mask)
    return decay_slopes


def extract_active_segments(y, sample_rate, segment_len_sec=5.0, num_segments=3):
    """
    Slices the audio into non-overlapping blocks of length segment_len_sec,
    ranks them by mean RMS energy, and concatenates the top num_segments
    to create a representative, highly active vocal training segment (e.g. 15s).
    """
    block_size = int(segment_len_sec * sample_rate)
    n_samples = y.shape[0]
    
    if n_samples <= block_size:
        return y
        
    num_possible_blocks = n_samples // block_size
    blocks = []
    block_energies = []
    
    for i in range(num_possible_blocks):
        start = i * block_size
        end = start + block_size
        block = y[start:end]
        energy = torch.sqrt(torch.mean(block ** 2) + 1e-8).item()
        blocks.append(block)
        block_energies.append((energy, i))
        
    # Sort blocks by energy descending
    block_energies.sort(key=lambda x: x[0], reverse=True)
    
    # Pick the top N blocks
    sel_count = min(num_segments, num_possible_blocks)
    selected_indices = [idx for _, idx in block_energies[:sel_count]]
    selected_indices.sort() # keep chronological order of segments
    
    selected_blocks = [blocks[idx] for idx in selected_indices]
    return torch.cat(selected_blocks)


def compute_crest_factor_pytorch(y):
    """
    Computes Crest Factor (Peak to RMS ratio) in dB in a differentiable manner.
    Uses L10 norm to approximate the maximum peak smoothly.
    """
    # L10 norm to smoothly approximate max absolute value
    peak = torch.norm(y, p=10)
    rms = torch.sqrt(torch.mean(y ** 2) + 1e-8)
    cf = 20.0 * torch.log10(peak / (rms + 1e-8) + 1e-8)
    return cf


def compute_rms_variance_pytorch(y, win_len=2048, hop_len=512):
    """
    Computes variance of the short-term RMS envelope (in dB) for active regions (> -45dB).
    """
    rms_db = compute_env_db_pytorch(y, win_len=win_len, hop_len=hop_len)
    mask = (rms_db > -45.0)
    active_rms = torch.masked_select(rms_db, mask)
    if len(active_rms) <= 1:
        return torch.tensor(0.0, device=y.device)
    return torch.var(active_rms)


def compute_lra_stft_pytorch(stft_complex, sample_rate=44100, hop_length=512):
    # stft_complex shape: [freq_bins, frames]
    frame_power = torch.sum(torch.abs(stft_complex) ** 2, dim=0) # [frames]
    
    # 3.0 second window and 0.1 second hop in frames
    win_frames = int(3.0 * sample_rate / hop_length)
    hop_frames = int(0.1 * sample_rate / hop_length)
    
    if frame_power.shape[0] < win_frames:
        win_frames = frame_power.shape[0]
        hop_frames = max(1, win_frames // 10)
        
    epsilon = 1e-8
    unfolded = frame_power.unfold(0, win_frames, hop_frames) # [num_blocks, win_frames]
    block_power = torch.mean(unfolded, dim=-1) # [num_blocks]
    
    st_loudness = 10.0 * torch.log10(block_power + epsilon)
    
    # Gate 1: Absolute gate (-70 dB relative to peak)
    max_loudness = torch.max(st_loudness).detach()
    abs_mask = st_loudness > (max_loudness - 70.0)
    abs_gated = torch.masked_select(st_loudness, abs_mask)
    if len(abs_gated) == 0:
        return torch.tensor(0.0, device=stft_complex.device)
        
    # Gate 2: Relative gate
    avg_power = torch.mean(torch.pow(10.0, abs_gated / 10.0))
    avg_loudness = 10.0 * torch.log10(avg_power + epsilon).detach()
    
    final_mask = abs_mask & (st_loudness > (avg_loudness - 20.0))
    rel_gated = torch.masked_select(st_loudness, final_mask)
    if len(rel_gated) < 2:
        return torch.tensor(0.0, device=stft_complex.device)
        
    sorted_loudness = torch.sort(rel_gated).values
    n = len(sorted_loudness)
    
    idx_95 = 0.95 * (n - 1)
    idx_95_f = int(idx_95)
    idx_95_c = min(idx_95_f + 1, n - 1)
    w_95 = idx_95 - idx_95_f
    p95 = (1.0 - w_95) * sorted_loudness[idx_95_f] + w_95 * sorted_loudness[idx_95_c]
    
    idx_10 = 0.10 * (n - 1)
    idx_10_f = int(idx_10)
    idx_10_c = min(idx_10_f + 1, n - 1)
    w_10 = idx_10 - idx_10_f
    p10 = (1.0 - w_10) * sorted_loudness[idx_10_f] + w_10 * sorted_loudness[idx_10_c]
    
    return p95 - p10


class DifferentiableReverb(nn.Module):
    """
    DDSP-inspired differentiable stereo reverb using filtered noise decay.
    """
    def __init__(self, sample_rate=44100, duration_seconds=1.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.num_samples = int(duration_seconds * sample_rate)

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

    def get_params(self):
        rt60 = 0.1 + 3.9 * torch.sigmoid(self.raw_rt60) # Range: [0.1s, 4.0s]
        wet = 0.7 * torch.sigmoid(self.raw_wet)        # Range: [0.0, 0.7]
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


def compute_lra_ebu(y, sample_rate=44100):
    """
    Computes EBU R128 standard Loudness Range (LRA) for time-domain audio signal:
    - 3-second sliding window, updated every 100 ms (10 Hz).
    - Absolute threshold gating (-70 dB relative to peak).
    - Relative threshold gating (-20 dB relative to average).
    - 95th to 10th percentile difference.
    """
    win_len = int(3.0 * sample_rate)
    hop_len = int(0.1 * sample_rate)
    
    if len(y) < win_len:
        win_len = max(int(0.5 * len(y)), int(0.5 * sample_rate))
        if len(y) < win_len:
            win_len = len(y)
            
    # Calculate power for each sliding block
    try:
        frames = librosa.util.frame(y, frame_length=win_len, hop_length=hop_len)
        powers = np.mean(frames ** 2, axis=0)
    except Exception:
        powers = np.array([np.mean(y ** 2)])
        
    st_loudness = 10.0 * np.log10(powers + 1e-8)
    
    # Gate 1: Absolute gate relative to peak
    max_loudness = np.max(st_loudness)
    abs_gated = st_loudness[st_loudness > (max_loudness - 70.0)]
    if len(abs_gated) == 0:
        return 0.0
        
    # Gate 2: Relative gate (-20 dB relative to the average loudness of absolute-gated blocks)
    avg_power = np.mean(10.0 ** (abs_gated / 10.0))
    avg_loudness = 10.0 * np.log10(avg_power + 1e-8)
    
    rel_gated = abs_gated[abs_gated > (avg_loudness - 20.0)]
    if len(rel_gated) == 0:
        return 0.0
        
    # LRA is 95th to 10th percentile difference
    lra = np.percentile(rel_gated, 95) - np.percentile(rel_gated, 10)
    return float(lra)


def get_equal_loudness_weights(frequencies_hz):
    """
    Computes perceptual weighting curve based on the human Equal-Loudness Contour (ISO 226 / Terhardt 1979):
    - Models the threshold of hearing.
    - Clamps the minimum weight to -30 dB for stability.
    - Normalizes the weight to 0 dB at 1000 Hz.
    """
    f = np.clip(frequencies_hz, 20.0, 20000.0)
    f_khz = f / 1000.0
    
    # Terhardt (1979) formula for the threshold of hearing in dB
    a_db = 3.64 * (f_khz ** -0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * (f_khz ** 4)
    
    # Reference at 1000 Hz
    a_db_1000 = 3.64 - 6.5 * np.exp(-0.6 * (1.0 - 3.3) ** 2) + 1e-3
    
    # Sensitivity curve: negative threshold normalized to 1kHz
    weight_db = - (a_db - a_db_1000)
    weight_db = np.clip(weight_db, -30.0, 10.0)
    
    # Linear gain multiplier
    weight_linear = 10.0 ** (weight_db / 20.0)
    return weight_linear

class DifferentiableEQ(nn.Module):
    """
    DDSP-inspired differentiable graphic equalizer using Gaussian filters.
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
    Differentiable single-band compressor with fixed attack/release ballistics.

    Trains only two parameters — threshold and ratio — exactly like a hardware
    compressor where the engineer sets those two knobs and leaves the time
    constants alone. Attack/release are stored as constant buffers, not
    parameters: making them trainable entangles them with threshold/ratio and
    only makes the optimization harder.

    Two design points differ from a naive frame-wise implementation:

    1. Ballistics. Gain reduction is smoothed with a one-pole filter -- the
       attack coefficient while the gain is being pulled down, the release
       coefficient while it recovers. Without this the compressor has no time
       constants at all and behaves like gain automation rather than a
       compressor.

    2. Log-scaled ratio. `ratio = 1 + exp(raw)` instead of a sigmoid mapped
       linearly onto [1, 8]. Measured sweeps show the useful range for vocals
       sits around ratio 1.0-2.5, which a linear map squeezes into a very narrow
       slice of the raw parameter and leaves the gradient poorly conditioned.
    """

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512,
                 attack_ms=0.5, release_ms=120.0, knee_db=4.0, makeup=True,
                 detector_hop=128):
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
        # match a target dynamic range (PLR).
        self.detector_hop = detector_hop
        self.detector_win = detector_hop * 2

        # Trainable parameters
        self.raw_threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.raw_ratio = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))  # ratio ~= 1.37

        # Fixed ballistics, expressed at the detector rate
        det_rate = sample_rate / detector_hop  # 44100/128 ~= 344.5 Hz
        self.register_buffer('a_att', torch.tensor(
            float(np.exp(-1.0 / max(attack_ms * 1e-3 * det_rate, 1e-6))), dtype=torch.float32))
        self.register_buffer('a_rel', torch.tensor(
            float(np.exp(-1.0 / max(release_ms * 1e-3 * det_rate, 1e-6))), dtype=torch.float32))

    def get_params(self):
        threshold = -60.0 * torch.sigmoid(self.raw_threshold)          # range: [-60dB, 0dB]
        ratio = 1.0 + torch.exp(torch.clamp(self.raw_ratio, max=4.0))  # range: [1.0, ~55]
        return threshold, ratio

    def forward(self, x, return_gain_reduction=False):
        """Compress a time-domain waveform.

        Args:
            x: waveform, shape [T].
        Returns:
            Compressed waveform [T], or (waveform, gain_reduction_db [1, n_det]).
        """
        threshold, ratio = self.get_params()
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
            # such as PLR; it only keeps the output at a usable level.
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


class CrestFactorShaper:
    """
    Crest Factor 기반 다이나믹스 프로세서.
    레퍼런스의 RMS 엔벨로프 분포(CDF)를 직접 본떠서
    입력 신호의 다이나믹 레인지를 quantile mapping으로 reshape.
    전통적 threshold/ratio 컴프레서를 대체.
    """
    
    SILENCE_THRESHOLD_DB = -45.0
    
    @staticmethod
    def extract_envelope_stats(stft_complex, num_quantiles=100):
        """
        STFT에서 프레임별 RMS 엔벨로프(dB)를 추출하고,
        활성 프레임(> -45dB)의 통계적 분포를 계산.
        
        Returns:
            env_db: 전체 프레임 RMS dB 엔벨로프 [n_frames]
            stats: dict with 'quantiles' (num_quantiles,), 'mean', 'std', 'min', 'max',
                   'crest_factor', 'dynamic_range'
        """
        epsilon = 1e-8
        magnitude = torch.abs(stft_complex)
        # 프레임별 RMS: frequency axis를 평균
        env = torch.sqrt(torch.mean(magnitude ** 2, dim=0) + epsilon)
        env_db = 20.0 * torch.log10(env + epsilon)
        
        # 활성 프레임 마스크
        active_mask = env_db > CrestFactorShaper.SILENCE_THRESHOLD_DB
        active_env_db = torch.masked_select(env_db, active_mask)
        
        if len(active_env_db) < 2:
            # 활성 프레임이 너무 적으면 기본 통계 반환
            stats = {
                'quantiles': torch.linspace(-30.0, -10.0, num_quantiles),
                'mean': torch.tensor(-20.0),
                'std': torch.tensor(5.0),
                'min': torch.tensor(-45.0),
                'max': torch.tensor(0.0),
                'crest_factor': torch.tensor(0.0),
                'dynamic_range': torch.tensor(0.0),
                'num_active': 0,
            }
            return env_db, stats
        
        # Quantile 분포 계산 (CDF의 역함수 역할)
        quantile_positions = torch.linspace(0.0, 1.0, num_quantiles, device=active_env_db.device)
        sorted_env, _ = torch.sort(active_env_db)
        
        # 인터폴레이션 기반 quantile 계산
        indices = quantile_positions * (len(sorted_env) - 1)
        idx_floor = torch.floor(indices).long()
        idx_ceil = torch.min(idx_floor + 1, torch.tensor(len(sorted_env) - 1, device=sorted_env.device))
        weights = indices - idx_floor.float()
        quantiles = (1.0 - weights) * sorted_env[idx_floor] + weights * sorted_env[idx_ceil]
        
        # Crest Factor: peak-to-RMS
        peak_env = torch.norm(env[active_mask], p=10)
        rms_env = torch.sqrt(torch.mean(env[active_mask] ** 2) + epsilon)
        crest_factor = 20.0 * torch.log10(peak_env / (rms_env + epsilon) + epsilon)
        
        # Dynamic Range: 95th - 5th percentile
        idx_95 = int(0.95 * (len(sorted_env) - 1))
        idx_05 = int(0.05 * (len(sorted_env) - 1))
        dynamic_range = sorted_env[idx_95] - sorted_env[idx_05]
        
        stats = {
            'quantiles': quantiles,
            'mean': torch.mean(active_env_db),
            'std': torch.std(active_env_db),
            'min': torch.min(active_env_db),
            'max': torch.max(active_env_db),
            'crest_factor': crest_factor,
            'dynamic_range': dynamic_range,
            'num_active': len(active_env_db),
        }
        
        return env_db, stats
    
    @staticmethod
    def quantile_map(src_env_db, src_stats, ref_stats, amount=1.0):
        """
        입력 엔벨로프의 각 프레임을 레퍼런스의 분포에 맞춰 quantile mapping.
        
        Args:
            src_env_db: 입력 프레임별 RMS dB [n_frames]
            src_stats: 입력의 envelope 통계 (extract_envelope_stats 결과)
            ref_stats: 레퍼런스의 envelope 통계
            amount: 0.0 (bypass) ~ 1.0 (full mapping) 블렌딩
            
        Returns:
            gain_db: 프레임별 적용할 게인 (dB) [n_frames]
            target_env_db: 매핑된 타겟 엔벨로프 [n_frames]
        """
        active_mask = src_env_db > CrestFactorShaper.SILENCE_THRESHOLD_DB
        
        if src_stats['num_active'] < 2 or ref_stats['num_active'] < 2:
            return torch.zeros_like(src_env_db), src_env_db.clone()
        
        src_quantiles = src_stats['quantiles']  # [num_quantiles]
        ref_quantiles = ref_stats['quantiles']  # [num_quantiles]
        
        # 각 활성 프레임에 대해: src CDF에서의 위치를 찾고, ref CDF에서 같은 위치의 값으로 매핑
        target_env_db = src_env_db.clone()
        
        active_values = torch.masked_select(src_env_db, active_mask)
        
        # 입력값이 src_quantiles에서 어느 위치에 해당하는지 인터폴레이션
        # searchsorted로 위치를 찾고, 그 위치에 해당하는 ref_quantiles 값을 매핑
        src_q_min = src_quantiles[0]
        src_q_max = src_quantiles[-1]
        ref_q_min = ref_quantiles[0]
        ref_q_max = ref_quantiles[-1]
        
        num_q = len(src_quantiles)
        
        # 정규화된 위치 계산: 입력값이 src 분포에서 0~1 사이 어디에 있는지
        # clamp하여 범위 밖 값도 안전하게 처리
        normalized_pos = (active_values - src_q_min) / (src_q_max - src_q_min + 1e-8)
        normalized_pos = torch.clamp(normalized_pos, 0.0, 1.0)
        
        # 정규화된 위치에서 ref_quantiles를 인터폴레이션하여 타겟값 산출
        ref_indices = normalized_pos * (num_q - 1)
        ref_idx_floor = torch.floor(ref_indices).long()
        ref_idx_ceil = torch.min(ref_idx_floor + 1, torch.tensor(num_q - 1, device=ref_indices.device))
        ref_weights = ref_indices - ref_idx_floor.float()
        
        mapped_values = (1.0 - ref_weights) * ref_quantiles[ref_idx_floor] + ref_weights * ref_quantiles[ref_idx_ceil]
        
        # amount로 블렌딩: 0 = 원본 유지, 1 = 완전 매핑
        blended_values = active_values + amount * (mapped_values - active_values)
        
        # target_env_db에 활성 프레임만 업데이트
        target_env_db[active_mask] = blended_values
        
        # 게인 계산
        gain_db = target_env_db - src_env_db
        
        return gain_db, target_env_db
    
    @staticmethod
    def apply_to_stft(stft_complex, gain_db, smooth_frames=5):
        """
        프레임별 gain_db를 스무딩한 후 STFT에 적용.
        
        Args:
            stft_complex: 입력 STFT [n_freqs, n_frames]
            gain_db: 프레임별 게인 [n_frames]
            smooth_frames: 스무딩 윈도우 크기 (프레임 단위)
            
        Returns:
            processed_stft: 게인 적용된 STFT
        """
        # 가우시안 스무딩으로 급격한 게인 변화 방지
        if smooth_frames > 1:
            # 1D 가우시안 커널 생성
            sigma = smooth_frames / 4.0
            half_win = smooth_frames // 2
            x = torch.arange(-half_win, half_win + 1, dtype=torch.float32, device=gain_db.device)
            kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            
            # conv1d로 스무딩
            gain_padded = torch.nn.functional.pad(
                gain_db.unsqueeze(0).unsqueeze(0),
                (half_win, half_win),
                mode='replicate'
            )
            gain_db_smooth = torch.nn.functional.conv1d(
                gain_padded,
                kernel.unsqueeze(0).unsqueeze(0)
            ).squeeze(0).squeeze(0)
        else:
            gain_db_smooth = gain_db
        
        # 게인 범위 제한 (-18dB ~ +12dB)
        gain_db_clamped = torch.clamp(gain_db_smooth, -18.0, 12.0)
        
        # 무음 구간은 게인을 0으로 (노이즈 증폭 방지)
        silence_mask = torch.abs(gain_db) < 1e-6  # quantile_map에서 무음은 gain=0
        gain_db_clamped[silence_mask] = 0.0
        
        # dB를 선형 게인으로
        gain_linear = torch.pow(10.0, gain_db_clamped / 20.0)
        
        # STFT에 적용 (frequency 축으로 broadcast)
        processed_stft = stft_complex * gain_linear.unsqueeze(0)
        
        return processed_stft
    
    @staticmethod
    def process(stft_complex, ref_stft_complex, amount=1.0, smooth_frames=7, num_quantiles=100):
        """
        전체 파이프라인: 레퍼런스의 다이나믹 분포를 본떠 입력 STFT를 reshape.
        
        Args:
            stft_complex: 입력 STFT (torch complex64)
            ref_stft_complex: 레퍼런스 STFT (torch complex64)
            amount: 0.0~1.0 적용량
            smooth_frames: 게인 스무딩 윈도우
            num_quantiles: 분포 매핑 해상도
            
        Returns:
            processed_stft: 처리된 STFT
            info: dict with diagnostic data
        """
        # 1. 레퍼런스 통계 추출
        ref_env_db, ref_stats = CrestFactorShaper.extract_envelope_stats(
            ref_stft_complex, num_quantiles=num_quantiles
        )
        
        # 2. 입력 통계 추출
        src_env_db, src_stats = CrestFactorShaper.extract_envelope_stats(
            stft_complex, num_quantiles=num_quantiles
        )
        
        # 3. Quantile Mapping
        gain_db, target_env_db = CrestFactorShaper.quantile_map(
            src_env_db, src_stats, ref_stats, amount=amount
        )
        
        # 4. 스무딩 후 STFT에 적용
        processed_stft = CrestFactorShaper.apply_to_stft(
            stft_complex, gain_db, smooth_frames=smooth_frames
        )
        
        # 5. 결과 통계 수집
        with torch.no_grad():
            proc_env_db, proc_stats = CrestFactorShaper.extract_envelope_stats(
                processed_stft, num_quantiles=num_quantiles
            )
        
        info = {
            'src_stats': src_stats,
            'ref_stats': ref_stats,
            'proc_stats': proc_stats,
            'gain_db': gain_db,
            'src_dynamic_range': float(src_stats['dynamic_range'].item()),
            'ref_dynamic_range': float(ref_stats['dynamic_range'].item()),
            'shaped_dynamic_range': float(proc_stats['dynamic_range'].item()),
        }
        
        return processed_stft, info



# ==============================================================================
# 1176-Style Vocal Compressor  –  Syllable Peak Variance Matching
# ==============================================================================

def _find_loudest_window(audio: np.ndarray, sr: int, window_sec: float = 3.0):
    """Returns (window_array, start_sample) for the highest-RMS segment."""
    window_samples = int(window_sec * sr)
    if len(audio) <= window_samples:
        return audio, 0
    hop = max(window_samples // 8, 1)
    best_rms, best_start = -1.0, 0
    for start in range(0, len(audio) - window_samples + 1, hop):
        rms = float(np.sqrt(np.mean(audio[start:start + window_samples] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, start
    return audio[best_start:best_start + window_samples], best_start


def _peak_follower(audio: np.ndarray, sr: int,
                   attack_ms: float = 1.0, release_ms: float = 100.0) -> np.ndarray:
    """
    Sample-by-sample peak envelope follower.
    attack_ms : 빠른 어택으로 순간 피크를 잡음
    release_ms: 릴리즈 (분석용 고정값; 실제 컴프레서는 auto release 사용)
    """
    ac = float(np.exp(-1.0 / (sr * attack_ms  / 1000.0)))
    rc = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))
    abs_audio = np.abs(audio)
    n = len(abs_audio)
    envelope = np.empty(n, dtype=np.float32)
    peak = float(abs_audio[0]) if n > 0 else 0.0
    for i in range(n):
        s = float(abs_audio[i])
        peak = ac * peak + (1.0 - ac) * s if s > peak else rc * peak
        envelope[i] = peak
    return envelope


def _syllable_peaks_from_envelope(peak_env: np.ndarray, sr: int,
                                   min_distance_ms: float = 80.0):
    """
    피크 엔벨로프에서 음절 피크(로컬 최대값)를 검출.
    Returns (peak_indices, peak_levels_db).
    """
    epsilon = 1e-8
    env_db = 20.0 * np.log10(peak_env + epsilon)
    min_dist = max(1, int(min_distance_ms * sr / 1000.0))
    # 검출 임계를 신호 최대 대비 상대값(-30dB)으로 설정 → 메이크업 게인 등 전역 레벨
    # 이동에 대해 불변. 절대 -60dB 임계는 압축/메이크업 시 검출 피크셋이 흔들려 편차 측정을 왜곡.
    height_db = float(np.max(env_db)) - 30.0
    try:
        from scipy.signal import find_peaks as _sp_find_peaks
        peaks, _ = _sp_find_peaks(env_db, distance=min_dist, height=height_db)
    except ImportError:
        peaks_list, last = [], -min_dist
        for i in range(1, len(env_db) - 1):
            if (env_db[i] >= env_db[i - 1] and env_db[i] > env_db[i + 1]
                    and env_db[i] > height_db and i - last >= min_dist):
                peaks_list.append(i)
                last = i
        peaks = np.array(peaks_list, dtype=int)
    if len(peaks) < 2:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    return peaks, env_db[peaks].astype(np.float32)


def _syllable_peak_variance(audio: np.ndarray, sr: int,
                             window_sec: float = 3.0) -> float:
    """가장 큰 RMS 윈도우에서 음절 피크 레벨(dB)의 표준편차를 반환."""
    window, _ = _find_loudest_window(audio, sr, window_sec)
    peak_env = _peak_follower(window, sr, attack_ms=1.0, release_ms=100.0)
    _, peak_levels_db = _syllable_peaks_from_envelope(peak_env, sr)
    return float(np.std(peak_levels_db)) if len(peak_levels_db) >= 2 else 0.0


def _apply_1176_compressor(audio: np.ndarray, sr: int,
                            threshold_db: float, ratio: float,
                            attack_ms: float = 3.0,
                            max_gr_db: float = -15.0) -> tuple:
    """
    1176-Style FET 컴프레서:
    - 피크 디텍션 (RMS 아님)
    - Auto release: GR이 클수록 릴리즈 빠름  (200ms base / (1 + |GR| / 8))
    - Hard knee
    - max_gr_db: 최대 GR 한계 (기본 -15 dB)
    - Auto makeup gain: 활성 프레임 평균 GR 보상

    Returns: (output_audio: np.ndarray, gain_reduction_per_sample: np.ndarray)
    """
    epsilon = 1e-8
    ac = float(np.exp(-1.0 / (sr * attack_ms / 1000.0)))

    n = len(audio)
    output  = np.empty(n, dtype=np.float64)
    gr_arr  = np.empty(n, dtype=np.float32)

    peak    = 0.0
    gain_db = 0.0

    for i in range(n):
        s     = float(audio[i])
        abs_s = abs(s)

        # Peak follower with auto release
        if abs_s > peak:
            peak = ac * peak + (1.0 - ac) * abs_s
        else:
            rel_ms = 200.0 / (1.0 + abs(gain_db) / 8.0)
            rel_ms = max(30.0, min(rel_ms, 300.0))
            rc = float(np.exp(-1.0 / (sr * rel_ms / 1000.0)))
            peak = rc * peak

        # Target gain reduction (hard knee) + max GR 하드 리밋
        peak_db = 20.0 * np.log10(peak + epsilon)
        if peak_db > threshold_db:
            tgr = -(peak_db - threshold_db) * (1.0 - 1.0 / ratio)
            tgr = max(tgr, max_gr_db)   # 게인 리덕션 한계 (과압축 방지)
        else:
            tgr = 0.0

        # Ballistic gain smoothing
        if tgr < gain_db:          # attack direction
            gain_db += (tgr - gain_db) * (1.0 - ac)
        else:                       # release direction (auto)
            rel_ms_g = 200.0 / (1.0 + abs(gain_db) / 8.0)
            rel_ms_g = max(30.0, min(rel_ms_g, 300.0))
            rc_g = float(np.exp(-1.0 / (sr * rel_ms_g / 1000.0)))
            gain_db = rc_g * gain_db + (1.0 - rc_g) * tgr

        gr_arr[i] = gain_db
        output[i] = s * float(10.0 ** (gain_db / 20.0))

    # Makeup gain은 여기서 적용하지 않는다. 표준 오토게인(= -평균 GR)을 STEP 2.1에서 압축 전후
    # 음량차로 실측·일괄 적용하기 위함. 압축은 순수 GR만 수행.
    # (음절 피크 std는 전역 게인에 불변이므로 makeup 제거는 탐색/측정에 영향 없음.)
    return output.astype(np.float32), gr_arr


def _binary_search_threshold(window_audio: np.ndarray, sr: int, ref_var: float,
                              ratio: float, max_gr_db: float,
                              th_lo: float = -48.0, th_hi: float = 0.0,
                              window_sec: float = 3.0, n_iter: int = 14) -> tuple:
    """
    Closed-loop threshold binary search.

    정적 gain 모델 대신 **실제 1176 컴프레서**(`_apply_1176_compressor`)를 분석 윈도우에
    적용한 뒤 `_syllable_peak_variance`로 결과 편차를 측정한다. 탐색과 최종 평가가 동일한
    연산 경로를 쓰므로, 어택/릴리즈 밸리스틱·오토 메이크업·피크 재검출로 인한
    "예측 var ≠ 실제 var" 괴리가 발생하지 않는다.

    탐색 중 관측된 최소 편차(threshold)를 반환하므로, 압축이 오히려 편차를 키우는
    파라미터는 절대 선택되지 않는다.

    Returns (best_threshold, achieved_var).
    """
    best_threshold = th_hi          # 사실상 무압축 지점에서 출발
    best_var = float('inf')

    for _ in range(n_iter):
        th_mid = (th_lo + th_hi) / 2.0
        comp, _ = _apply_1176_compressor(window_audio, sr, th_mid, ratio,
                                          max_gr_db=max_gr_db)
        c_var = _syllable_peak_variance(comp, sr, window_sec)

        if c_var < best_var:        # 실측 최소 편차 추적
            best_var = c_var
            best_threshold = th_mid

        if c_var <= ref_var:
            th_lo = th_mid          # 목표 달성 → threshold 완화
        else:
            th_hi = th_mid          # 편차 여전히 큼 → 압축 강화(threshold 낮춤)

    return best_threshold, best_var


def match_compression_1176(raw_audio: np.ndarray, ref_audio: np.ndarray,
                            sr: int, ratio: int = 4,
                            comp_amount: float = 1.0,
                            window_sec: float = 3.0) -> dict:
    """
    1176-Style 보컬 컴프레서 – 음절 피크 편차 매칭.

    ratio를 2 → 4 → 8 → 20 → ∞ 순으로 자동 탐색:
    - 각 ratio에서 th_lo = -36 dB 범위 안에 목표 달성 가능하면 그 ratio 사용
    - 달성 불가하면 다음 (더 강한) ratio로 진행
    - max_gr_db도 ratio에 맞춰 점진적으로 강화
    """
    # ratio→max_gr 대응: 레이시오가 높을수록 더 강한 리밋 허용
    RATIO_PLAN = [
        (2,   -8.0),   # gentle
        (4,   -12.0),  # moderate
        (8,   -15.0),  # heavy
        (20,  -20.0),  # brick-wall
        (999, -24.0),  # ∞:1 limiter
    ]
    MAX_PRACTICAL_THRESHOLD = -36.0

    # 측정 윈도우 고정: raw 의 loudest window 를 한 번 선정하고, 이후 src/탐색/최종 편차를
    # 모두 '동일 시간 구간'에서 측정한다. (구간이 매번 바뀌면 편차 비교가 사과-오렌지가 됨)
    raw_win, win_start = _find_loudest_window(raw_audio, sr, window_sec)
    win_len = len(raw_win)

    # Measure syllable peak variances (raw 는 고정 윈도우, ref 는 자체 loudest window)
    ref_var = _syllable_peak_variance(ref_audio, sr, window_sec)
    src_var = _syllable_peak_variance(raw_win, sr, window_sec)
    print(f"[1176] Syllable peak std  —  raw: {src_var:.2f} dB,  ref: {ref_var:.2f} dB")

    # Already uniform or bypass
    if src_var <= ref_var or comp_amount < 0.01:
        print("[1176] Raw already uniform or bypassed.")
        return {'output': raw_audio.copy(),
                'threshold_db': 0.0, 'max_gr_db': 0.0, 'ratio': 1,
                'src_var': src_var, 'ref_var': ref_var, 'final_var': src_var}

    chosen_ratio   = RATIO_PLAN[-1][0]
    chosen_max_gr  = RATIO_PLAN[-1][1]
    best_threshold = MAX_PRACTICAL_THRESHOLD
    best_achieved  = float('inf')

    for r, mgr in RATIO_PLAN:
        th, achieved_var = _binary_search_threshold(
            raw_win, sr, ref_var, r, mgr,
            th_lo=-48.0, th_hi=0.0, window_sec=window_sec
        )
        ratio_label = f"{r}:1" if r < 999 else "∞:1 (limiter)"
        print(f"[1176] Trying ratio {ratio_label}: "
              f"threshold={th:.1f} dB, achieved_var={achieved_var:.2f} dB")
        # 이 ratio가 지금까지보다 편차를 더 줄였을 때만 채택 (악화 방지)
        if achieved_var < best_achieved:
            best_achieved  = achieved_var
            chosen_ratio   = r
            chosen_max_gr  = mgr
            best_threshold = th
        if achieved_var <= ref_var * 1.1:  # 10% 여유로 목표 달성
            break  # 이 ratio로 충분

    r_label = f"{chosen_ratio}:1" if chosen_ratio < 999 else "∞:1 (limiter)"
    print(f"[1176] Selected  ratio: {r_label},  threshold: {best_threshold:.1f} dB,  max_gr: {chosen_max_gr} dB")

    # --- Apply full 1176 compressor to entire audio ---
    compressed_full, gr_arr = _apply_1176_compressor(
        raw_audio, sr, best_threshold, chosen_ratio, max_gr_db=chosen_max_gr
    )

    # Blend dry/wet
    output = (raw_audio + comp_amount * (compressed_full - raw_audio)).astype(np.float32)

    # 최종 편차도 src/탐색과 '동일한 고정 시간 구간'에서 측정 (윈도우 재선정 금지)
    out_win = output[win_start:win_start + win_len]
    final_var = _syllable_peak_variance(out_win, sr, window_sec)
    max_gr    = float(np.min(gr_arr))
    print(f"[1176] Final syllable std: {final_var:.2f} dB,  Max GR: {max_gr:.1f} dB")

    # 안전장치: 전체 음원 적용 결과가 오히려 편차를 키웠다면 압축 무효화 → 원음 보존
    if final_var > src_var:
        print(f"[1176] Compression increased variance ({src_var:.2f} → {final_var:.2f}) — reverting to dry.")
        return {'output': raw_audio.copy(),
                'threshold_db': 0.0, 'max_gr_db': 0.0, 'ratio': 1,
                'src_var': src_var, 'ref_var': ref_var, 'final_var': src_var}

    return {
        'output':       output,
        'threshold_db': best_threshold,
        'max_gr_db':    max_gr,
        'ratio':        chosen_ratio,
        'src_var':      src_var,
        'ref_var':      ref_var,
        'final_var':    final_var,
    }


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


def apply_hpf_lpf(freq_bins, hpf_freq, lpf_freq, order=2):
    """
    Generates high-pass and low-pass filter curves in the frequency domain.
    """
    n_freqs = len(freq_bins)
    hpf_curve = np.ones(n_freqs)
    lpf_curve = np.ones(n_freqs)
    
    freq_bins_clean = np.copy(freq_bins)
    freq_bins_clean[0] = 1e-5
    
    if hpf_freq is not None and hpf_freq > 0:
        hpf_curve = 1.0 / (1.0 + (hpf_freq / freq_bins_clean) ** (2 * order))
        hpf_curve[0] = 0.0
        
    if lpf_freq is not None and lpf_freq < freq_bins[-1]:
        lpf_curve = 1.0 / (1.0 + (freq_bins_clean / lpf_freq) ** (2 * order))
        
    return torch.tensor(hpf_curve * lpf_curve, dtype=torch.float32)


def match_eq(raw_path, ref_path, out_path, num_bands=5, match_amount=1.0, smoothness=1.0,
             hpf_freq=80.0, lpf_freq=16000.0, match_volume=False, max_gain_db=15.0,
             comp_amount=1.0, reverb_amount=1.0, mode="both"):
    """
    DDSP-based advanced mixing matching function:
    HPF -> Multiband Compressor (Optional) -> Auto-EQ (Optional) -> Reverb (Optional)
    Uses a highly stable sequential optimization strategy.
    """
    target_sr = 44100
    n_fft = 2048
    hop_length = 512
    n_mels = 80
    epsilon = 1e-8
    
    # 1. Load raw and reference audios
    y_raw, sr_raw = librosa.load(raw_path, sr=target_sr, mono=True)
    y_ref, sr_ref = librosa.load(ref_path, sr=target_sr, mono=True)
    
    # Peak normalize both audios to -6 dBFS (0.501187) to secure headroom from the start
    y_raw = (y_raw / (np.max(np.abs(y_raw)) + 1e-8)) * 0.501187
    y_ref = (y_ref / (np.max(np.abs(y_ref)) + 1e-8)) * 0.501187


    # Perform STFT
    stft_raw = librosa.stft(y_raw, n_fft=n_fft, hop_length=hop_length)
    stft_ref = librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length)
    
    t_stft_raw = torch.tensor(stft_raw, dtype=torch.complex64)
    t_stft_ref = torch.tensor(stft_ref, dtype=torch.complex64)
    
    # Compute spectral envelopes of raw signal and reference at the beginning
    raw_mag_spec, raw_mel, mel_fb = compute_spectral_envelope(y_raw, target_sr, n_fft, hop_length, n_mels)
    _, ref_mel, _ = compute_spectral_envelope(y_ref, target_sr, n_fft, hop_length, n_mels)
    
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0/target_sr)
    
    gr_max_db = [0.0]
    bands_x_val = np.logspace(np.log10(20.0), np.log10(20000.0), num_bands).tolist()
    bands_y_val = [0.0] * num_bands
    
    reverb_data = {
        "rt60": 0.0,
        "wet": 0.0,
        "active": False
    }
    
    # Parse modes (supports comma-separated active modes and legacy modes)
    if mode == "both":
        active_modes = ["eq", "comp"]
    elif mode == "eq":
        active_modes = ["eq"]
    elif mode == "comp":
        active_modes = ["comp"]
    elif mode == "reverb":
        active_modes = ["eq", "comp", "reverb"]
    else:
        active_modes = [m.strip().lower() for m in mode.split(",") if m.strip()]
    
    # --- STEP 1: Auto-EQ Optimization (On Raw Vocal) ---
    if "eq" in active_modes:
        print("Optimizing Equalizer...")
        raw_mel_norm = raw_mel / (torch.sum(raw_mel) + 1e-8)
        ref_mel_norm = ref_mel / (torch.sum(ref_mel) + 1e-8)
        
        eq = DifferentiableEQ(sample_rate=target_sr, n_fft=n_fft, num_bands=num_bands, max_gain_db=max_gain_db)
        eq_optimizer = optim.Adam(eq.parameters(), lr=0.15)
        
        raw_power = torch.mean(raw_mag_spec ** 2, dim=1)
        
        for epoch in range(500):
            eq_optimizer.zero_grad()
            
            eq_curve_db, band_gains = eq.get_eq_curve_db()
            eq_gain_linear = torch.pow(10.0, eq_curve_db / 10.0)
            filtered_power = raw_power * eq_gain_linear
            
            filtered_mel = torch.matmul(mel_fb, filtered_power)
            filtered_mel_norm = filtered_mel / (torch.sum(filtered_mel) + 1e-8)
            
            filtered_mel_db = 10.0 * torch.log10(filtered_mel_norm + epsilon)
            ref_mel_db = 10.0 * torch.log10(ref_mel_norm + epsilon)
            
            # Mask out frequencies below 100Hz and treat other frequencies equally (flat weight)
            mel_frequencies = librosa.mel_frequencies(n_mels=80, fmin=0, fmax=target_sr/2)
            mel_freqs_tensor = torch.tensor(mel_frequencies, dtype=torch.float32, device=filtered_mel_db.device)
            
            eq_loss_mask = (mel_freqs_tensor >= 100.0).float()
            
            loss_spectral = torch.sum(((filtered_mel_db - ref_mel_db) ** 2) * eq_loss_mask) / (torch.sum(eq_loss_mask) + 1e-8)
            loss_l2 = 0.001 * torch.mean(band_gains ** 2)
            loss_smooth = smoothness * 0.01 * torch.mean(torch.diff(band_gains) ** 2)
            
            loss = loss_spectral + loss_l2 + loss_smooth
            loss.backward()
            eq_optimizer.step()
            
        # Extract final optimized EQ curve
        with torch.no_grad():
            final_eq_db, final_band_gains = eq.get_eq_curve_db()
            blended_eq_db = final_eq_db * match_amount
            
            # Force EQ gain to be exactly 0 dB below 100 Hz
            eq_processing_mask = torch.tensor(freq_bins >= 100.0, dtype=torch.float32, device=blended_eq_db.device)
            blended_eq_db = blended_eq_db * eq_processing_mask
            
            blended_gains_linear = torch.pow(10.0, blended_eq_db / 20.0)
            final_gains_linear = blended_gains_linear
            final_eq_db_with_filters = blended_eq_db
            
            bands_x_val = eq.bands.tolist()
            bands_y_val = final_band_gains.numpy().tolist()
    else:
        print("Skipping Equalizer (Bypass)...")
        with torch.no_grad():
            blended_eq_db = torch.zeros(len(freq_bins))
            blended_gains_linear = torch.ones(len(freq_bins))
            final_gains_linear = blended_gains_linear
            final_eq_db_with_filters = blended_eq_db
            
    # Apply EQ to raw audio to get y_eq
    stft_raw_complex = librosa.stft(y_raw, n_fft=n_fft, hop_length=hop_length)
    gains_tensor = final_gains_linear.unsqueeze(1).numpy()
    stft_eq = stft_raw_complex * gains_tensor
    y_eq = librosa.istft(stft_eq, hop_length=hop_length)
    t_stft_eq = torch.tensor(stft_eq, dtype=torch.complex64)
    
    # --- STEP 2: 1176-Style Vocal Compressor (Syllable Peak Variance Matching) ---
    shaper_info = {}

    if "comp" in active_modes:
        print("Applying 1176-Style Vocal Compressor...")

        # Precompute Crest Factor / RMS Variance for diagnostics (UI 호환 유지)
        with torch.no_grad():
            ref_cf      = compute_stft_crest_factor(t_stft_ref)
            ref_rms_var = compute_stft_rms_variance(t_stft_ref, n_fft // 2 + 1)
            raw_cf      = compute_stft_crest_factor(t_stft_eq)
            raw_rms_var = compute_stft_rms_variance(t_stft_eq, n_fft // 2 + 1)

            cf_raw_val      = float(raw_cf.item())
            cf_ref_val      = float(ref_cf.item())
            rms_var_raw_val = float(raw_rms_var.item())
            rms_var_ref_val = float(ref_rms_var.item())

        print(f"Reference dynamics: Crest Factor = {cf_ref_val:.2f} dB, RMS Variance = {rms_var_ref_val:.2f}")
        print(f"Input dynamics:     Crest Factor = {cf_raw_val:.2f} dB, RMS Variance = {rms_var_raw_val:.2f}")

        # 1176-style compressor: 음절 피크 편차를 레퍼런스에 맞춤
        comp_result = match_compression_1176(
            raw_audio=y_eq,
            ref_audio=y_ref,
            sr=target_sr,
            comp_amount=comp_amount,
            window_sec=3.0
        )

        y_processed   = comp_result['output']
        gr_max_db     = [comp_result['max_gr_db']]
        stft_processed = librosa.stft(y_processed, n_fft=n_fft, hop_length=hop_length)

        shaper_info = {
            'src_dynamic_range':    comp_result['src_var'],
            'ref_dynamic_range':    comp_result['ref_var'],
            'shaped_dynamic_range': comp_result['final_var'],
            'ratio':                comp_result.get('ratio', 4),
        }

        print(f"1176: raw_var={comp_result['src_var']:.1f}dB → "
              f"ref_var={comp_result['ref_var']:.1f}dB → "
              f"final_var={comp_result['final_var']:.1f}dB  "
              f"(threshold={comp_result['threshold_db']:.1f}dB)")
    else:
        print("Skipping Compressor (Bypass)...")
        y_processed     = y_eq
        stft_processed  = stft_eq
        cf_raw_val, cf_ref_val           = 0.0, 0.0
        rms_var_raw_val, rms_var_ref_val = 0.0, 0.0
        
    # --- STEP 2.1: Auto Makeup Gain (표준 오토게인 = -평균 GR, 실측 기반) ---
    # 일반 컴프레서 플러그인처럼 "압축으로 깎인 평균 음량만큼 되돌린다".
    # 컴프 전(y_eq) 대비 후(y_processed)의 RMS 차이가 곧 실현된 평균 게인 리덕션이므로,
    # 그만큼 makeup 게인을 적용. 밸리스틱·dry/wet·comp on-off를 자동으로 반영한다.
    # (boost-only 클램프: 컴프는 음량을 줄이므로 되돌리기만, 과보상/삭감 방지.)
    if "comp" in active_modes:
        pre_rms = np.sqrt(np.mean(y_eq ** 2) + 1e-8)
        post_rms = np.sqrt(np.mean(y_processed ** 2) + 1e-8)
        makeup_db = float(np.clip(20.0 * np.log10(pre_rms / (post_rms + 1e-8)), 0.0, 24.0))
        y_processed = y_processed * (10.0 ** (makeup_db / 20.0))
        print(f"[AutoGain] makeup = {makeup_db:.2f} dB (= -avg GR)")

    # --- STEP 2.5: Stereo Reverb Optimization ---
    if "reverb" in active_modes:
        print("Optimizing Stereo Reverb...")
        t_y_comp = torch.tensor(y_processed, dtype=torch.float32)
        t_y_ref = torch.tensor(y_ref, dtype=torch.float32)
        
        # Optimize Reverb Model
        reverb_model = DifferentiableReverb(sample_rate=target_sr, duration_seconds=1.5)
        reverb_optimizer = optim.Adam(reverb_model.parameters(), lr=0.1)
        
        # Extract active segments (5s blocks * 3 = 15s total) with the highest vocal energy
        t_y_comp_opt = extract_active_segments(t_y_comp, target_sr, segment_len_sec=5.0, num_segments=3)
        t_y_ref_opt = extract_active_segments(t_y_ref, target_sr, segment_len_sec=5.0, num_segments=3)
        
        # Precompute reference envelope and decay slopes on the 15-second segment
        with torch.no_grad():
            ref_rms_db = compute_env_db_pytorch(t_y_ref_opt, win_len=2048, hop_len=512)
            ref_slopes = compute_decay_slopes_pytorch(ref_rms_db, 512, target_sr)
            ref_slopes_sorted = torch.sort(ref_slopes).values
            ref_env_sorted = torch.sort(ref_rms_db).values

            # Precompute input segment FFT for the loop (size is capped to 15 seconds)
            x_pad_opt = torch.cat([t_y_comp_opt, torch.zeros(reverb_model.num_samples, device=t_y_comp_opt.device)])
            n_sig_opt = x_pad_opt.shape[-1]
            n_ker = reverb_model.num_samples
            rev_n_fft = 2 ** int(np.ceil(np.log2(n_sig_opt + n_ker - 1)))
            sig_fft = torch.fft.rfft(x_pad_opt, n=rev_n_fft)

        for epoch in range(100):
            reverb_optimizer.zero_grad()

            # Forward pass on the segment: returns stereo RIR convolved output [2, T_padded_opt] using precomputed FFT
            y_rev_stereo = reverb_model(t_y_comp_opt, sig_fft=sig_fft, n_fft=rev_n_fft)
            y_rev_mono = torch.mean(y_rev_stereo, dim=0)

            # Compute slopes of convolved segment signal
            proc_rms_db = compute_env_db_pytorch(y_rev_mono, win_len=2048, hop_len=512)
            proc_slopes = compute_decay_slopes_pytorch(proc_rms_db, 512, target_sr)

            proc_slopes_sorted = torch.sort(proc_slopes).values
            proc_env_sorted = torch.sort(proc_rms_db).values

            # Interpolate processed to reference segment length
            proc_slopes_sorted_interp = interpolate_sorted(proc_slopes_sorted, len(ref_slopes_sorted))
            proc_env_sorted_interp = interpolate_sorted(proc_env_sorted, len(ref_env_sorted))

            # Reverb loss: optimize pure decay slope similarity (isolated from absolute volume difference)
            loss_decay = torch.mean((proc_slopes_sorted_interp - ref_slopes_sorted) ** 2) if len(ref_slopes_sorted) > 0 else torch.tensor(0.0)

            loss = loss_decay
            loss.backward()
            reverb_optimizer.step()
            
        with torch.no_grad():
            opt_rt60, opt_wet = reverb_model.get_params()
            
            # Blend wet amount by user scaling slider
            scaled_wet = opt_wet * reverb_amount
            
            # Reconstruct stereo convolved output for the ENTIRE audio with final parameters (once at the end)
            alpha = 6.9078 / (opt_rt60 * target_sr)
            t_env = torch.arange(reverb_model.num_samples, device=t_y_comp.device, dtype=torch.float32)
            decay_env = torch.exp(-alpha * t_env)
            
            rir_l = reverb_model.noise_l * decay_env
            rir_r = reverb_model.noise_r * decay_env
            rir_l = rir_l / (torch.sqrt(torch.sum(rir_l ** 2)) + 1e-8)
            rir_r = rir_r / (torch.sqrt(torch.sum(rir_r ** 2)) + 1e-8)
            
            # Convolve the entire signal (single pass)
            x_pad_full = torch.cat([t_y_comp, torch.zeros(reverb_model.num_samples, device=t_y_comp.device)])
            n_sig_full = x_pad_full.shape[-1]
            n_fft_full = 2 ** int(np.ceil(np.log2(n_sig_full + n_ker - 1)))
            
            sig_fft_full = torch.fft.rfft(x_pad_full, n=n_fft_full)
            ker_l_fft = torch.fft.rfft(rir_l, n=n_fft_full)
            ker_r_fft = torch.fft.rfft(rir_r, n=n_fft_full)
            
            y_l = torch.fft.irfft(sig_fft_full * ker_l_fft, n=n_fft_full)[..., :n_sig_full]
            y_r = torch.fft.irfft(sig_fft_full * ker_r_fft, n=n_fft_full)[..., :n_sig_full]
            
            dry_pad = x_pad_full
            final_l = (1.0 - scaled_wet) * dry_pad + scaled_wet * y_l
            final_r = (1.0 - scaled_wet) * dry_pad + scaled_wet * y_r
            
            y_processed = torch.stack([final_l, final_r], dim=0).cpu().numpy()
            
            final_reverb_loss = float(loss.item())
            mae_decay = torch.mean(torch.abs(proc_slopes_sorted_interp - ref_slopes_sorted)).item() if len(ref_slopes_sorted) > 0 else 0.0
            mae_env = torch.mean(torch.abs(proc_env_sorted_interp - ref_env_sorted)).item() if len(ref_env_sorted) > 0 else 0.0
            reverb_mae = mae_decay * 0.05 + mae_env * 0.5
            reverb_sim = float(np.clip(100.0 * np.exp(-0.05 * reverb_mae), 50.0, 99.5))

            reverb_error_db = float(reverb_mae)
            reverb_data = {
                "rt60": float(opt_rt60.item()),
                "wet": float(scaled_wet.item()),
                "loss": final_reverb_loss,
                "error_db": reverb_error_db,
                "similarity": reverb_sim,
                "active": True
            }
            print(f"Optimized RT60: {reverb_data['rt60']:.2f}s, Wet: {reverb_data['wet']:.3f}, Similarity: {reverb_sim:.1f}%")

    # --- STEP 3: Final Safety Limiter & Save ---
    # Soft Peak Limiter: Smoothly shape peaks above 0.88 (-1.1 dBFS) to prevent digital clipping
    peak = np.max(np.abs(y_processed))
    if peak > 0.88:
        thresh = 0.85
        ceiling = 0.96
        abs_y = np.abs(y_processed)
        over_mask = abs_y > thresh
        if np.any(over_mask):
            over = abs_y[over_mask] - thresh
            y_processed[over_mask] = np.sign(y_processed[over_mask]) * (thresh + (ceiling - thresh) * np.tanh(over / (ceiling - thresh)))
        
    # Save processed audio
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # If stereo (2, T), transpose to (T, 2) for soundfile.write
    y_save = y_processed.T if (y_processed.ndim == 2 and y_processed.shape[0] == 2) else y_processed
    sf.write(out_path, y_save, target_sr)
    
    # Calculate processed envelope for visualization
    with torch.no_grad():
        raw_mag_spec_orig, raw_mel_orig, _ = compute_spectral_envelope(y_raw, target_sr, n_fft, hop_length, n_mels)
        raw_env_db = 10 * torch.log10(raw_mel_orig / torch.max(raw_mel_orig) + 1e-5).numpy()
        ref_env_db = 10 * torch.log10(ref_mel / torch.max(ref_mel) + 1e-5).numpy()
        
        # If stereo, downmix to mono for envelope visualization
        y_proc_mono = np.mean(y_processed, axis=0) if (y_processed.ndim == 2 and y_processed.shape[0] == 2) else y_processed
        
        stft_proc_eval = librosa.stft(y_proc_mono, n_fft=n_fft, hop_length=hop_length)
        proc_power_tensor = torch.tensor(np.mean(np.abs(stft_proc_eval) ** 2, axis=1), dtype=torch.float32)
        proc_mel = torch.matmul(mel_fb, proc_power_tensor)
        proc_env_db = 10 * torch.log10(proc_mel / torch.max(proc_mel) + 1e-5).numpy()
        
        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=target_sr/2)
        
        # 1. Calculate Tonal (Spectral) Similarity
        proc_mel_norm = proc_mel / (torch.sum(proc_mel) + 1e-8)
        ref_mel_norm = ref_mel / (torch.sum(ref_mel) + 1e-8)
        proc_mel_db = 10.0 * torch.log10(proc_mel_norm + epsilon)
        ref_mel_db_norm = 10.0 * torch.log10(ref_mel_norm + epsilon)
        
        final_spectral_mse = torch.mean((proc_mel_db - ref_mel_db_norm) ** 2).item()
        spec_similarity = 100.0 * np.exp(-0.015 * final_spectral_mse)
        
        # 2. Calculate Dynamics Deviation (Crest Factor MAE & RMS Variance MAE)
        t_stft_proc_eval = torch.tensor(stft_proc_eval, dtype=torch.complex64)
        proc_cf = float(compute_stft_crest_factor(t_stft_proc_eval).item())
        proc_rms_var = float(compute_stft_rms_variance(t_stft_proc_eval, n_fft // 2 + 1).item())
        
        cf_error = float(abs(proc_cf - cf_ref_val))
        rms_var_error = float(abs(proc_rms_var - rms_var_ref_val))
        
        # Combined Dynamics MAE — 1176 컴프레서가 실제 매칭하는 음절 피크 편차(std) 기준.
        # (기존 Crest Factor / RMS Variance 방식은 컴프레서 목표와 불일치하여 교체.)
        # cf_error / rms_var_error 는 진단용으로 compression_data 에 계속 포함.
        ref_syl_var    = float(shaper_info.get('ref_dynamic_range', 0.0))
        shaped_syl_var = float(shaper_info.get('shaped_dynamic_range', 0.0))
        if "comp" in active_modes and ref_syl_var > 0.0:
            dyn_mae = float(abs(shaped_syl_var - ref_syl_var))
        else:
            # 컴프레서 바이패스: 음절 편차 미측정 → 기존 Crest/RMS 방식 유지
            dyn_mae = float(0.5 * cf_error + 0.5 * rms_var_error)
        
        # 1. Tonal Deviation (Flat MAE in dB, excluding below 100Hz)
        mel_frequencies = librosa.mel_frequencies(n_mels=80, fmin=0, fmax=target_sr/2)
        mel_freqs_tensor_eval = torch.tensor(mel_frequencies, dtype=torch.float32, device=proc_mel_db.device)
        
        eval_mask = (mel_freqs_tensor_eval >= 100.0).float()
        tonal_mae = float((torch.sum(torch.abs(proc_mel_db - ref_mel_db_norm) * eval_mask) / (torch.sum(eval_mask) + 1e-8)).item())
        
        # 3. Combined Deviation (65% EQ, 35% Compressor)
        combined_mae = 0.65 * tonal_mae + 0.35 * dyn_mae
        
        # Sample EQ curve points for visualization
        eq_curve_x = freq_bins.tolist()
        eq_curve_y = blended_eq_db.numpy().tolist()
        sample_indices = np.unique(np.logspace(0, np.log10(len(freq_bins)-1), 200).astype(int))
        eq_curve_x_sampled = [float(eq_curve_x[idx]) for idx in sample_indices]
        eq_curve_y_sampled = [float(eq_curve_y[idx]) for idx in sample_indices]
        
    return {
        "status": "success",
        "output_file": out_path,
        "match_error": combined_mae,
        "tonal_error": tonal_mae,
        "dynamics_error": dyn_mae,
        "reverb_error": reverb_data.get("error_db", 0.0) if reverb_data.get("active") else 0.0,
        "chart_data": {
            "frequencies": mel_frequencies.tolist(),
            "raw_envelope": raw_env_db.tolist(),
            "ref_envelope": ref_env_db.tolist(),
            "proc_envelope": proc_env_db.tolist(),
            "eq_curve_x": eq_curve_x_sampled,
            "eq_curve_y": eq_curve_y_sampled,
            "bands_x": bands_x_val,
            "bands_y": bands_y_val
        },
        "compression_data": {
            "gain_reduction_max": gr_max_db,
            "cf_raw": cf_raw_val,
            "cf_ref": cf_ref_val,
            "cf_proc": proc_cf,
            "cf_error": cf_error,
            "rms_var_raw": rms_var_raw_val,
            "rms_var_ref": rms_var_ref_val,
            "rms_var_proc": proc_rms_var,
            "rms_var_error": rms_var_error,
            "dynamics_error": dyn_mae,
            "src_dynamic_range": shaper_info.get('src_dynamic_range', 0.0),
            "ref_dynamic_range": shaper_info.get('ref_dynamic_range', 0.0),
            "shaped_dynamic_range": shaper_info.get('shaped_dynamic_range', 0.0),
            "ratio": shaper_info.get('ratio', 4)
        },
        "reverb_data": reverb_data,
        "gate_data": None
    }


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


def _loudest_lufs(x, sr, win_sec=3.0, hop_sec=1.0):
    """
    가장 큰 구간의 LUFS(ITU-R BS.1770, K-weighting 적용) 측정.
    win_sec 길이 창을 hop_sec 간격으로 슬라이딩하며 통합 라우드니스를 재고 그 최댓값 반환.
    x: [ch, T] (또는 [T]). 반환 단위 LUFS(대략 dB와 동일 스케일).
    """
    import pyloudnorm as pyln
    data = x.T if x.ndim > 1 else x  # pyloudnorm: [samples, channels] 또는 [samples]
    T = data.shape[0]
    meter = pyln.Meter(sr)
    win = int(win_sec * sr)
    hop = max(1, int(hop_sec * sr))
    if T < win:
        try:
            return float(meter.integrated_loudness(data))
        except Exception:
            return -70.0
    best = -np.inf
    for start in range(0, T - win + 1, hop):
        seg = data[start:start + win]
        try:
            L = meter.integrated_loudness(seg)
        except Exception:
            continue
        if np.isfinite(L) and L > best:
            best = L
    return float(best) if np.isfinite(best) else -70.0


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
