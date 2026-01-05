import torch
import torch.fft as fft
from typing import List, Tuple, Optional, Dict

# -----------------------------
# Time helpers
# -----------------------------


def _segment_indices(T: int, nseg: int) -> List[slice]:
    """Split a length-T sequence into nseg nearly equal slices."""
    idx = []
    start = 0
    for k in range(nseg):
        end = round((k + 1) * T / nseg)
        idx.append(slice(start, end))
        start = end
    return idx

# -----------------------------
# Temporal Pyramid Pooling (TPP) stats
# -----------------------------


def _tpp_stats(x: torch.Tensor, levels: List[int]) -> torch.Tensor:
    """
    Compute stats over temporal pyramid segments per channel.
    x: (B, T, C)
    Returns: (B, S, C) where S = sum(levels) * n_stats, n_stats=5
        stats per segment: [mean, std, rms, line_length, peak_to_peak]
    """
    B, T, C = x.shape
    feats = []
    eps = 1e-9

    for nseg in levels:
        for sl in _segment_indices(T, nseg):
            seg = x[:, sl, :]                     # (B, L, C)
            # mean, std
            m = seg.mean(dim=1)
            sd = seg.std(dim=1, unbiased=False)
            # rms
            rms = torch.sqrt((seg**2).mean(dim=1) + eps)
            # line length (sum |diff|)
            ll = seg.diff(dim=1).abs().sum(dim=1)
            # peak-to-peak (max - min)
            ptp = seg.max(dim=1).values - seg.min(dim=1).values
            feats.extend([m, sd, rms, ll, ptp])

    return torch.stack(feats, dim=1)  # (B, 5*sum(levels), C)

# -----------------------------
# Peak features (task-agnostic)
# -----------------------------


def _peak_feats(x: torch.Tensor) -> torch.Tensor:
    """
    Positive/negative peak amplitude and normalized latency (0..1).
    x: (B, T, C)
    Returns: (B, 4, C) = [pos_amp, pos_lat_norm, neg_amp, neg_lat_norm]
    """
    B, T, C = x.shape
    # positive peak
    pos_amp, pos_idx = x.max(dim=1)         # (B, C)
    # negative peak
    neg_amp, neg_idx = x.min(dim=1)         # (B, C)
    # normalized latency in [0,1]
    tnorm = 1.0 / max(T - 1, 1)
    pos_lat = pos_idx * tnorm
    neg_lat = neg_idx * tnorm
    return torch.stack([pos_amp, pos_lat, neg_amp, neg_lat], dim=1)

# -----------------------------
# Hjorth parameters
# -----------------------------


def _hjorth(x: torch.Tensor) -> torch.Tensor:
    """
    Hjorth parameters over whole epoch per channel.
    x: (B, T, C)
    Returns: (B, 3, C) = [activity, mobility, complexity]
    """
    eps = 1e-9
    B, T, C = x.shape
    activity = x.var(dim=1, unbiased=False)
    dx = x.diff(dim=1)
    var_dx = dx.var(dim=1, unbiased=False) + eps
    mobility = torch.sqrt(var_dx / (activity + eps))
    ddx = dx.diff(dim=1)
    var_ddx = ddx.var(dim=1, unbiased=False) + eps
    mobility_dx = torch.sqrt(var_ddx / var_dx)
    complexity = (mobility_dx / (mobility + eps)).clamp(max=1e6)
    return torch.stack([activity, mobility, complexity], dim=1)

# -----------------------------
# PSD helpers (Welch or periodogram)
# -----------------------------


def _get_window(N: int, device, dtype, kind: str = "hann") -> torch.Tensor:
    if kind == "hann":
        return torch.hann_window(N, periodic=True, device=device, dtype=dtype)
    if kind == "hamming":
        return torch.hamming_window(N, periodic=True, device=device, dtype=dtype)
    raise ValueError(f"Unsupported window: {kind}")


def _periodogram_seg(x_seg: torch.Tensor, fs: float, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Periodogram density for one segment.
    x_seg: (B, L, C)
    Returns: (Pxx: B,F,C [power/Hz], freqs: F)
    """
    B, L, C = x_seg.shape
    win = window.view(1, -1, 1)
    xw = x_seg * win
    X = fft.rfft(xw, dim=1)             # (B, F, C)
    U = (window**2).sum()
    Pxx = (X.abs()**2) / (fs * U)
    if L % 2 == 0:
        Pxx[:, 1:-1, :] *= 2.0
    else:
        Pxx[:, 1:, :] *= 2.0
    freqs = fft.rfftfreq(L, d=1.0/fs).to(device=x_seg.device, dtype=x_seg.dtype)
    return Pxx, freqs


def _psd_welch(x: torch.Tensor, fs: float, nperseg: Optional[int] = None, noverlap: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Welch PSD with robust defaults; fallback to periodogram if needed.
    x: (B, T, C)
    Returns: (Pxx: B,F,C, freqs: F)
    """
    B, T, C = x.shape
    n = min(256, T) if nperseg is None else nperseg
    n = max(64, min(n, T))
    if n == T:
        w = _get_window(T, x.device, x.dtype, "hann")
        return _periodogram_seg(x, fs, w)

    ov = (n // 2) if noverlap is None else noverlap
    step = n - ov
    if n > T or step <= 0:
        w = _get_window(T, x.device, x.dtype, "hann")
        return _periodogram_seg(x, fs, w)

    w = _get_window(n, x.device, x.dtype, "hann")
    acc, cnt = None, 0
    for start in range(0, T - n + 1, step):
        seg = x[:, start:start + n, :]
        Pxx_seg, freqs = _periodogram_seg(seg, fs, w)
        acc = Pxx_seg if acc is None else (acc + Pxx_seg)
        cnt += 1
    Pxx = acc / cnt
    return Pxx, freqs

# -----------------------------
# Spectral block (task-agnostic)
# -----------------------------


def _band_power(psd: torch.Tensor, freqs: torch.Tensor, f_lo: float, f_hi: float) -> torch.Tensor:
    """Integrate PSD density over [f_lo, f_hi]. Returns (B, C)."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() < 2:
        return torch.zeros(psd.shape[0], psd.shape[2], device=psd.device, dtype=psd.dtype)
    P = psd[:, mask, :]
    f = freqs[mask]
    return torch.trapz(P, f, dim=1)


def _spectral_block(x: torch.Tensor, fs: float) -> torch.Tensor:
    """
    Build robust, task-agnostic spectral features per channel.
    Returns: (B, 8, C) = [rel_delta, rel_theta, rel_alpha, rel_beta,
                           total_power, centroid, flatness, median_freq]
    """
    eps = 1e-12
    Pxx, freqs = _psd_welch(x, fs)

    total = torch.trapz(Pxx, freqs, dim=1).clamp_min(eps)  # (B, C)

    delta = _band_power(Pxx, freqs, 0.5, 4.0)
    theta = _band_power(Pxx, freqs, 4.0, 8.0)
    alpha = _band_power(Pxx, freqs, 8.0, 12.0)
    beta  = _band_power(Pxx, freqs, 12.0, 30.0)

    rel_d = delta / total
    rel_t = theta / total
    rel_a = alpha / total
    rel_b = beta  / total

    # Spectral centroid
    f = freqs.view(1, -1, 1).to(device=x.device, dtype=x.dtype)
    num = torch.trapz(Pxx * f, freqs, dim=1)
    centroid = (num / total).clamp_min(0)

    # Spectral flatness
    geo = torch.exp(torch.mean(torch.log(Pxx.clamp_min(eps)), dim=1))
    ari = torch.mean(Pxx, dim=1).clamp_min(eps)
    flatness = (geo / ari).clamp(0, 1)

    # Median frequency
    # cumulative power via trapezoidal rule
    dF = (freqs[1:] - freqs[:-1]).view(1, -1, 1)
    cum = torch.cumsum(((Pxx[:, 1:, :] + Pxx[:, :-1, :]) / 2) * dF, dim=1)
    cum = torch.cat([torch.zeros_like(cum[:, :1, :]), cum], dim=1)  # align to (B,F,C)
    half = 0.5 * total
    idx = (cum >= half[:, None, :]).float().argmax(dim=1)
    median_f = freqs[idx]

    return torch.stack([rel_d, rel_t, rel_a, rel_b, total, centroid, flatness, median_f], dim=1)


def _spectral_entropy(x: torch.Tensor, fs: float) -> torch.Tensor:
    """
    Normalized Shannon spectral entropy per channel.
    Returns: (B, 1, C)
    """
    eps = 1e-12
    Pxx, freqs = _psd_welch(x, fs)
    P = Pxx.clamp_min(eps)
    P = P / P.sum(dim=1, keepdim=True)
    H = -(P * torch.log(P)).sum(dim=1)  # nat
    H_norm = H / torch.log(torch.tensor(P.shape[1], device=x.device, dtype=x.dtype))
    return H_norm.unsqueeze(1)

# -----------------------------
# Master extractor (task-agnostic ERP-averaged features)
# -----------------------------


def erp_agnostic_feature_extractor(
    x_avg: torch.Tensor,
    fs: float,
    levels: List[int] = [1, 2, 4, 8]
) -> torch.Tensor:
    """
    Task-agnostic ERP-averaged features per channel.
    Args:
        x_avg: (B, T, C) averaged ERP trials (already averaged within condition)
        fs: sampling rate in Hz
        levels: temporal pyramid levels (sum(levels) segments)
    Returns:
        (B, F, C) with F = 5*sum(levels) + 4 + 8 + 1 + 3 = 91 by default
    """
    # Optional: baseline correction before calling this (e.g., subtract pre-stimulus mean).
    tpp = _tpp_stats(x_avg, levels)                  # (B, 5*sum(levels), C)
    pk  = _peak_feats(x_avg)                         # (B, 4, C)
    sp  = _spectral_block(x_avg, fs)                 # (B, 8, C)
    se  = _spectral_entropy(x_avg, fs)               # (B, 1, C)
    hj  = _hjorth(x_avg)                             # (B, 3, C)
    return torch.cat([tpp, pk, sp, se, hj], dim=1)   # (B, 91, C)


def get_erp_agnostic_feat_dim(levels: List[int] = [1,2,4,8]) -> int:
    """Return per-channel feature dimension F for given temporal pyramid levels."""
    return 5*sum(levels) + 4 + 8 + 1 + 3  # = 91 (default)
