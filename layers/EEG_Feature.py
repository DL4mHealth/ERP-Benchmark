import torch
import torch.fft as fft
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Utilities
# -----------------------------


def _get_window(win_len: int, device, dtype, kind: str = "hann") -> torch.Tensor:
    """Return window tensor."""
    if kind.lower() == "hann":
        w = torch.hann_window(win_len, periodic=True, device=device, dtype=dtype)
    elif kind.lower() == "hamming":
        w = torch.hamming_window(win_len, periodic=True, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported window: {kind}")
    return w


def _periodogram_single_segment(x_seg: torch.Tensor, fs: float, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Periodogram with proper density scaling.
    Args:
        x_seg: (B, L, C) real-valued segment
        fs: sampling rate
        window: (L,) window
    Returns:
        Pxx: (B, F, C) power spectral density [power/Hz]
        freqs: (F,) frequency bins
    """
    B, L, C = x_seg.shape
    win = window.view(1, -1, 1)
    xw = x_seg * win
    X = fft.rfft(xw, dim=1)          # (B, F, C)
    U = (window**2).sum()            # window power
    Pxx = (X.abs() ** 2) / (fs * U)  # density

    # Single-sided factor (except DC/Nyquist)
    if L % 2 == 0:
        Pxx[:, 1:-1, :] *= 2.0
    else:
        Pxx[:, 1:, :] *= 2.0

    freqs = fft.rfftfreq(L, d=1.0/fs).to(device=x_seg.device, dtype=x_seg.dtype)
    return Pxx, freqs


def compute_psd_periodogram(x_enc: torch.Tensor, fs: float, window: str = "hann") -> Tuple[torch.Tensor, torch.Tensor]:
    """PSD via periodogram on full length."""
    B, T, C = x_enc.shape
    w = _get_window(T, x_enc.device, x_enc.dtype, window)
    return _periodogram_single_segment(x_enc, fs, w)


def compute_psd_welch(x_enc: torch.Tensor, fs: float,
                      nperseg: Optional[int] = None, noverlap: Optional[int] = None,
                      window: str = "hann") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Welch PSD with sensible fallbacks for short trials.
    Args:
        x_enc: (B, T, C)
    Returns:
        Pxx: (B, F, C) density
        freqs: (F,)
    """
    B, T, C = x_enc.shape
    if nperseg is None:
        nperseg = min(256, T)  # works for 0.8–2s @ 200Hz -> T=160–400
    nperseg = max(64, min(nperseg, T))
    if nperseg == T:
        return compute_psd_periodogram(x_enc, fs, window)

    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if step <= 0:
        noverlap = nperseg // 2
        step = nperseg - noverlap
    if nperseg > T or step <= 0:
        return compute_psd_periodogram(x_enc, fs, window)

    w = _get_window(nperseg, x_enc.device, x_enc.dtype, window)
    Pxx_acc, n_segs = None, 0

    for start in range(0, T - nperseg + 1, step):
        seg = x_enc[:, start:start + nperseg, :]
        Pxx_seg, freqs = _periodogram_single_segment(seg, fs, w)
        Pxx_acc = Pxx_seg if Pxx_acc is None else (Pxx_acc + Pxx_seg)
        n_segs += 1

    Pxx = Pxx_acc / n_segs
    return Pxx, freqs

# -----------------------------
# Band power (density -> power)
# -----------------------------


def compute_band_power(psd: torch.Tensor, freqs: torch.Tensor, band: Tuple[float, float]) -> torch.Tensor:
    """
    Integrate PSD over frequency band to obtain band power.
    Args:
        psd: (B, F, C) density
        freqs: (F,)
        band: (low, high)
    Returns:
        (B, C) band power
    """
    f_lo, f_hi = band
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() < 2:
        return torch.zeros(psd.shape[0], psd.shape[2], device=psd.device, dtype=psd.dtype)
    P = psd[:, mask, :]         # (B, Fb, C)
    f = freqs[mask]             # (Fb,)
    return torch.trapz(P, f, dim=1)

# -----------------------------
# Statistical features (time)
# -----------------------------


def statistical_feature_extractor(x_enc: torch.Tensor) -> torch.Tensor:
    """
    Time-domain statistics per channel.
    Args:
        x_enc: (B, T, C)
    Returns:
        (B, 10, C): mean, std, var, skew, kurtosis(excess), iqr, median, min, max, rms
    """
    eps = 1e-9
    mean = x_enc.mean(dim=1)
    std  = x_enc.std(dim=1, unbiased=False)
    var  = std**2
    median = x_enc.median(dim=1).values
    minv   = x_enc.min(dim=1).values
    maxv   = x_enc.max(dim=1).values
    rms    = torch.sqrt((x_enc**2).mean(dim=1) + eps)
    q75 = torch.quantile(x_enc, 0.75, dim=1)
    q25 = torch.quantile(x_enc, 0.25, dim=1)
    iqr = q75 - q25
    dev = x_enc - mean.unsqueeze(1)
    m3 = (dev**3).mean(dim=1)
    m4 = (dev**4).mean(dim=1)
    skew = m3 / (var.clamp_min(eps) ** 1.5 + eps)
    kurt_ex = m4 / (var.clamp_min(eps) ** 2 + eps) - 3.0
    out = torch.stack([mean, std, var, skew, kurt_ex, iqr, median, minv, maxv, rms], dim=1)
    return out  # (B, 10, C)

# -----------------------------
# Spectral features from PSD
# -----------------------------


def spectral_features_from_psd(psd: torch.Tensor, freqs: torch.Tensor, rolloff: float = 0.85) -> torch.Tensor:
    """
    Spectral shape features derived from PSD.
    Returns (B, 7, C):
        1) spectral centroid
        2) rolloff frequency (at rolloff * total power)
        3) peak frequency
        4) peak power
        5) mean frequency (same as centroid)
        6) median frequency
        7) spectral flatness (geo/arith)
    """
    eps = 1e-12
    B, F, C = psd.shape
    f = freqs.view(1, F, 1).to(device=psd.device, dtype=psd.dtype)

    total_power = torch.trapz(psd, freqs, dim=1).clamp_min(eps)  # (B, C)
    num = torch.trapz(psd * f, freqs, dim=1)                     # (B, C)
    spec_centroid = num / total_power

    # Cumulative power using trapezoidal rule
    cumsum_power = torch.cumsum(((psd[:, 1:, :] + psd[:, :-1, :]) / 2.0) *
                                (freqs[1:] - freqs[:-1]).view(1, -1, 1), dim=1)  # (B, F-1, C)
    cumsum_power = torch.cat([torch.zeros(B, 1, C, device=psd.device, dtype=psd.dtype), cumsum_power], dim=1)  # (B, F, C)

    # Rolloff frequency
    thresh = rolloff * total_power
    roll_idx = (cumsum_power >= thresh[:, None, :]).float().argmax(dim=1)
    spec_rolloff = freqs[roll_idx]

    # Peak frequency and power
    peak_idx = psd.argmax(dim=1)
    peak_freq = freqs[peak_idx]
    peak_pow  = psd.gather(1, peak_idx.unsqueeze(1)).squeeze(1)

    mean_freq = spec_centroid

    # Median frequency
    half = 0.5 * total_power
    med_idx = (cumsum_power >= half[:, None, :]).float().argmax(dim=1)
    med_freq = freqs[med_idx]

    # Spectral flatness
    geo = torch.exp(torch.mean(torch.log(psd.clamp_min(eps)), dim=1))
    ari = torch.mean(psd, dim=1).clamp_min(eps)
    spec_flat = (geo / ari).clamp(0, 1)

    out = torch.stack([spec_centroid, spec_rolloff, peak_freq, peak_pow, mean_freq, med_freq, spec_flat], dim=1)
    return out  # (B, 7, C)

# -----------------------------
# Complexity (entropy) from PSD
# -----------------------------


def spectral_entropy(psd: torch.Tensor, base: float = 2.0, normalize: bool = True) -> torch.Tensor:
    """
    Shannon spectral entropy over discrete frequency bins.
    Args:
        psd: (B, F, C)
    Returns:
        (B, C) entropy (normalized to [0,1] if normalize=True)
    """
    eps = 1e-12
    P = psd.clamp_min(eps)
    P = P / P.sum(dim=1, keepdim=True)
    logP = torch.log(P) / torch.log(torch.tensor(base, device=psd.device, dtype=psd.dtype))
    H = -(P * logP).sum(dim=1)
    if normalize:
        H = H / torch.log(torch.tensor(P.shape[1], device=psd.device, dtype=psd.dtype)) * torch.log(torch.tensor(base, device=psd.device, dtype=psd.dtype))
    return H


def tsallis_entropy_from_psd(psd: torch.Tensor, q: float = 2.0) -> torch.Tensor:
    """Tsallis entropy computed on spectral distribution."""
    eps = 1e-12
    P = psd.clamp_min(eps)
    P = P / P.sum(dim=1, keepdim=True)
    return (1.0 - (P**q).sum(dim=1)) / (q - 1.0)

# -----------------------------
# Power-related features (bands + ratios)
# -----------------------------


def power_feature_block_from_psd(Pxx: torch.Tensor, freqs: torch.Tensor,
                                 bands: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """
    Assemble 11-D power features:
        4 absolute powers + total power + 2 ratios + 4 relative powers
    Returns:
        (B, 11, C)
    """
    eps = 1e-12
    total_power = torch.trapz(Pxx, freqs, dim=1).clamp_min(eps)  # (B, C)

    delta = compute_band_power(Pxx, freqs, bands["delta"])
    theta = compute_band_power(Pxx, freqs, bands["theta"])
    alpha = compute_band_power(Pxx, freqs, bands["alpha"])
    beta  = compute_band_power(Pxx, freqs, bands["beta"])

    rel_delta = delta / total_power
    rel_theta = theta / total_power
    rel_alpha = alpha / total_power
    rel_beta  = beta  / total_power

    theta_alpha = theta / alpha.clamp_min(eps)
    alpha_beta  = alpha / beta.clamp_min(eps)

    feat_list = [delta, theta, alpha, beta, total_power, theta_alpha, alpha_beta,
                 rel_delta, rel_theta, rel_alpha, rel_beta]
    return torch.stack(feat_list, dim=1)  # (B, 11, C)

# -----------------------------
# Wrappers
# -----------------------------


def spectral_feature_extractor(x_enc: torch.Tensor, fs: float,
                               use_welch: bool = True,
                               nperseg: Optional[int] = None, noverlap: Optional[int] = None) -> torch.Tensor:
    """Return (B, 7, C) spectral features derived from PSD."""
    if use_welch:
        Pxx, freqs = compute_psd_welch(x_enc, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    else:
        Pxx, freqs = compute_psd_periodogram(x_enc, fs, window="hann")
    return spectral_features_from_psd(Pxx, freqs, rolloff=0.85)


def complexity_feature_extractor(x_enc: torch.Tensor, fs: float,
                                 use_welch: bool = True,
                                 nperseg: Optional[int] = None, noverlap: Optional[int] = None) -> torch.Tensor:
    """Return (B, 3, C): [spectral_entropy_norm, tsallis_q2, spectral_entropy_bits]."""
    if use_welch:
        Pxx, freqs = compute_psd_welch(x_enc, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    else:
        Pxx, freqs = compute_psd_periodogram(x_enc, fs, window="hann")
    H_norm = spectral_entropy(Pxx, base=2.0, normalize=True)
    H_bits = spectral_entropy(Pxx, base=2.0, normalize=False)
    T_q2   = tsallis_entropy_from_psd(Pxx, q=2.0)
    return torch.stack([H_norm, T_q2, H_bits], dim=1)


def power_feature_extractor(x_enc: torch.Tensor, fs: float,
                            use_welch: bool = True,
                            nperseg: Optional[int] = None, noverlap: Optional[int] = None,
                            bands: Dict[str, Tuple[float, float]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (feat, Pxx, freqs) where feat is (B, 11, C)."""
    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta":  (12.0, 30.0),
        }
    if use_welch:
        Pxx, freqs = compute_psd_welch(x_enc, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    else:
        Pxx, freqs = compute_psd_periodogram(x_enc, fs, window="hann")
    feat = power_feature_block_from_psd(Pxx, freqs, bands)
    return feat, Pxx, freqs

# -----------------------------
# Master extractor (no ERP windows)
# -----------------------------


def feature_extractor(
    x_enc: torch.Tensor,
    fs: float = 200.0,
    use_welch: bool = True,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None
) -> torch.Tensor:
    """
    Extract EEG features from single-trial data (no ERP-window features).

    Input shape:
        x_enc: (B, T, C)
            B = number of trials
            T = time samples per trial
            C = EEG channels

    Output shape:
        (B, 31, C)
        Each channel has 31 handcrafted features:
            - 10 time-domain statistical features
            - 11 power / band-power features
            - 7 spectral-shape features
            - 3 entropy / complexity features

    Args
    ----
    x_enc : torch.Tensor
        EEG signal of shape (B, T, C)
    fs : float, default = 200.0
        Sampling frequency in Hz
    use_welch : bool, default = True
        Whether to compute PSD using Welch method (recommended)
    nperseg : int, optional
        Segment length for Welch (default: min(256, T))
        If None, choose automatically based on trial length
    noverlap : int, optional
        Overlap for Welch (default: nperseg // 2)
        If None, set automatically

    Returns
    -------
    torch.Tensor
        Feature tensor of shape (B, 31, C)
        31 features per channel
    """

    # ----- Time-domain statistical features -----
    stat = statistical_feature_extractor(x_enc)  # (B, 10, C)
    # ----- Power spectral and band-power features -----
    pow_feat, Pxx, freqs = power_feature_extractor(x_enc, fs, use_welch, nperseg, noverlap)  # pow_feat: (B, 11, C)
    # ----- Spectral-shape features -----
    spec = spectral_features_from_psd(Pxx, freqs, rolloff=0.85)  # (B, 7, C)
    # ----- Entropy / Complexity features -----
    comp = complexity_feature_extractor(x_enc, fs, use_welch, nperseg, noverlap)  # (B, 3, C)
    # ----- Concatenate all feature blocks along feature dimension -----
    out = torch.cat([stat, pow_feat, spec, comp], dim=1)  # (B, 31, C)

    return out

