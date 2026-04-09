"""RDP + local AIC wavefront detector.

This module is extracted from the user's original script and keeps only the
algorithmic parts. All tunable values are exposed through function arguments so
that a PyQt UI can pass parameters directly.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pywt
from scipy import signal

FS = 4.2e6

DEFAULT_RDP_LOCAL_AIC_PARAMS: dict[str, Any] = {
    "fs": FS,
    "pre_n": 600,
    "rough_k": 5.0,
    "rough_consecutive": 8,
    "threshold_sg_window": 11,
    "threshold_sg_poly": 3,
    "rdp_left_us": 60.0,
    "rdp_right_us": 85.0,
    "rdp_epsilon": 0.008,
    "rdp_sg_window": 7,
    "rdp_sg_poly": 2,
    "rdp_seg_slope_k": 3.5,
    "rdp_seg_amp_k": 3.0,
    "rdp_weak_factor": 0.40,
    "search_left_us": 5.0,
    "search_right_us": 10.0,
    "noise_guard_us": 1.5,
    "noise_win_us": 100.0,
    "slope_win": 17,
    "slope_poly": 2,
    "amp_k": 3.5,
    "slope_k": 2.0,
    "min_consecutive": 5,
    "fit_n": 26,
    "polarity": "auto",
    "aic_left_us": 6.0,
    "aic_right_us": 8.0,
    "aic_min_split": 5,
    "aic_smooth_win": 5,
    "aic_smooth_poly": 2,
    "wavelet_enabled": True,
    "wavelet": "db4",
    "wavelet_level": 4,
    "wavelet_beta": 3.0,
    "wavelet_threshold_scale": 1.0,
    "wavelet_use_level_dependent_sigma": True,
    "wavelet_mode": "symmetric",
}


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return float(mad / 0.6745)


def smooth_savgol(x: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return x.copy()

    window_length = max(5, int(window_length))
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= n:
        window_length = n - 1 if n % 2 == 0 else n
    if window_length < 5:
        return x.copy()

    polyorder = min(int(polyorder), window_length - 2)
    return signal.savgol_filter(
        x,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp",
    )


def moving_sg_slope(y: np.ndarray, win: int = 17, poly: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return np.gradient(y)

    win = max(5, int(win))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win < 5:
        return np.gradient(y)

    poly = min(int(poly), win - 2)
    return signal.savgol_filter(
        y,
        window_length=win,
        polyorder=poly,
        deriv=1,
        delta=1.0,
        mode="interp",
    )


def improved_threshold(coeff: np.ndarray, lam: float, beta: float) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=float)
    out = np.zeros_like(coeff)
    if lam <= 0:
        return coeff.copy()

    mask = np.abs(coeff) >= lam
    abs_c = np.abs(coeff[mask])
    out[mask] = np.sign(coeff[mask]) * (
        abs_c - lam * np.exp(-beta * (abs_c / (lam + 1e-12) - 1.0))
    )
    return out


def _check_pywt() -> None:
    if pywt is None:
        raise ImportError("PyWavelets(pywt) 未安装或导入失败，无法执行小波降噪。") from _PYWT_IMPORT_ERROR


def wavelet_denoise_improved(
    x: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 4,
    beta: float = 3.0,
    threshold_scale: float = 1.0,
    use_level_dependent_sigma: bool = True,
    mode: str = "symmetric",
) -> tuple[np.ndarray, dict[str, Any]]:
    _check_pywt()
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 8:
        return x.copy(), {
            "enabled": True,
            "wavelet": wavelet,
            "level": 0,
            "beta": beta,
            "threshold_scale": threshold_scale,
            "noise_sigma": 0.0,
            "use_level_dependent_sigma": use_level_dependent_sigma,
            "mode": mode,
        }

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=n, filter_len=wavelet_obj.dec_len)
    if max_level <= 0:
        return x.copy(), {
            "enabled": True,
            "wavelet": wavelet,
            "level": 0,
            "beta": beta,
            "threshold_scale": threshold_scale,
            "noise_sigma": 0.0,
            "use_level_dependent_sigma": use_level_dependent_sigma,
            "mode": mode,
        }

    level = min(max(1, int(level)), max_level)
    coeffs = pywt.wavedec(x, wavelet=wavelet_obj, mode=mode, level=level)
    detail_coeffs = coeffs[1:]
    cD1 = coeffs[-1]
    noise_sigma_global = robust_sigma(cD1)

    new_coeffs = [coeffs[0]]
    level_sigmas: list[float] = []
    level_lambdas: list[float] = []

    for cD in detail_coeffs:
        sigma_j = robust_sigma(cD) if use_level_dependent_sigma else noise_sigma_global
        lam_j = threshold_scale * sigma_j * np.sqrt(2.0 * np.log(max(len(cD), 2)))
        level_sigmas.append(float(sigma_j))
        level_lambdas.append(float(lam_j))
        new_coeffs.append(improved_threshold(cD, lam=lam_j, beta=beta))

    x_denoised = pywt.waverec(new_coeffs, wavelet=wavelet_obj, mode=mode)
    x_denoised = np.asarray(x_denoised[:n], dtype=float)

    info = {
        "enabled": True,
        "wavelet": wavelet,
        "level": level,
        "beta": float(beta),
        "threshold_scale": float(threshold_scale),
        "noise_sigma": float(noise_sigma_global),
        "use_level_dependent_sigma": bool(use_level_dependent_sigma),
        "mode": mode,
        "level_sigmas": level_sigmas,
        "level_lambdas": level_lambdas,
    }
    return x_denoised, info


def rdp_indices(points: np.ndarray, epsilon: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return np.arange(n, dtype=int)

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue

        p1 = points[start]
        p2 = points[end]
        seg = p2 - p1
        seg_len = np.hypot(seg[0], seg[1])

        inner = points[start + 1 : end]
        if inner.size == 0:
            continue

        if seg_len < 1e-12:
            d = np.hypot(inner[:, 0] - p1[0], inner[:, 1] - p1[1])
        else:
            v = inner - p1
            d = np.abs(seg[0] * v[:, 1] - seg[1] * v[:, 0]) / seg_len

        i_rel = int(np.argmax(d))
        dmax = float(d[i_rel])
        if dmax > epsilon:
            idx = start + 1 + i_rel
            keep[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))

    return np.where(keep)[0]


def detect_polarity_from_window(y: np.ndarray, baseline: float, i0: int, i1: int) -> tuple[float, str]:
    seg = y[i0:i1] - baseline
    if seg.size == 0:
        return 1.0, "positive"
    sign = 1.0 if np.max(seg) >= abs(np.min(seg)) else -1.0
    return sign, ("positive" if sign > 0 else "negative")


def build_rdp_points_local(
    x: np.ndarray,
    center_idx: int,
    *,
    fs: float = FS,
    pre_n: int = 600,
    left_us: float = 60.0,
    right_us: float = 85.0,
    epsilon: float = 0.008,
    sg_window: int = 7,
    sg_poly: int = 2,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = smooth_savgol(x, sg_window, sg_poly)

    left_n = int(round(left_us * 1e-6 * fs))
    right_n = int(round(right_us * 1e-6 * fs))

    i0 = max(0, int(center_idx) - left_n)
    i1 = min(len(y), int(center_idx) + right_n)
    xx = np.arange(i0, i1, dtype=float)

    baseline = np.median(y[: min(pre_n, len(y))])
    y_centered = y[i0:i1] - baseline

    x_norm = (xx - xx.min()) / max(xx.max() - xx.min(), 1e-12)
    y_norm = y_centered / max(np.max(np.abs(y_centered)), 1e-12)

    pts = np.column_stack([x_norm, y_norm])
    keep_local = rdp_indices(pts, epsilon=epsilon)
    keep_idx = xx[keep_local].astype(int)

    return {
        "rdp_x": keep_idx.astype(float),
        "rdp_y": y[keep_idx],
        "turn_idx": keep_idx,
        "x_rdp_smooth": y,
        "rdp_epsilon": epsilon,
        "rdp_i0": i0,
        "rdp_i1": i1,
        "baseline": baseline,
    }


def threshold_rough_locator(
    x: np.ndarray,
    *,
    fs: float = FS,
    sg_window: int = 11,
    sg_poly: int = 3,
    pre_n: int = 600,
    k: float = 8.0,
    min_consecutive: int = 8,
) -> dict[str, Any]:
    y = smooth_savgol(x, sg_window, sg_poly)
    pre_n = min(int(pre_n), len(y))
    pre = y[:pre_n]
    baseline = float(np.median(pre))
    sigma = robust_sigma(pre)
    thr = float(k * sigma)

    flag = np.abs(y - baseline) > thr
    rough_idx = None
    count = 0
    for i, v in enumerate(flag):
        count = count + 1 if v else 0
        if count >= int(min_consecutive):
            rough_idx = int(i - min_consecutive + 1)
            break

    if rough_idx is None:
        rough_idx = int(np.argmax(np.abs(y - baseline)))

    return {
        "rough_idx": rough_idx,
        "rough_t": rough_idx / fs,
        "x_smooth": y,
        "baseline": baseline,
        "rough_thr": thr,
        "pre_n": pre_n,
    }


def rdp_rough_locator(
    x: np.ndarray,
    *,
    fs: float = FS,
    pre_n: int = 600,
    threshold_sg_window: int = 11,
    threshold_sg_poly: int = 3,
    rough_k: float = 8.0,
    rough_consecutive: int = 8,
    rdp_left_us: float = 60.0,
    rdp_right_us: float = 85.0,
    rdp_epsilon: float = 0.008,
    rdp_sg_window: int = 7,
    rdp_sg_poly: int = 2,
    seg_slope_k: float = 3.5,
    seg_amp_k: float = 3.0,
    weak_factor: float = 0.40,
    max_lead_us: float = 14.0,
) -> dict[str, Any]:
    base = threshold_rough_locator(
        x,
        fs=fs,
        sg_window=threshold_sg_window,
        sg_poly=threshold_sg_poly,
        pre_n=pre_n,
        k=rough_k,
        min_consecutive=rough_consecutive,
    )

    rough_idx = int(base["rough_idx"])
    y_base = base["x_smooth"]
    baseline = float(base["baseline"])

    pol_i0 = max(0, rough_idx - int(round(6e-6 * fs)))
    pol_i1 = min(len(y_base), rough_idx + int(round(18e-6 * fs)))
    sign, pol = detect_polarity_from_window(y_base, baseline, pol_i0, pol_i1)

    rdp_info = build_rdp_points_local(
        x,
        center_idx=rough_idx,
        fs=fs,
        pre_n=pre_n,
        left_us=rdp_left_us,
        right_us=rdp_right_us,
        epsilon=rdp_epsilon,
        sg_window=rdp_sg_window,
        sg_poly=rdp_sg_poly,
    )

    idx = np.asarray(rdp_info["turn_idx"], dtype=int)
    y_rdp = rdp_info["x_rdp_smooth"]

    cand = None
    seg_slope_signed = np.array([], dtype=float)
    rdp_seg_thr = np.nan
    rdp_amp_thr = np.nan

    if len(idx) >= 3:
        dx = np.diff(idx).astype(float)
        dy = sign * np.diff(y_rdp[idx])
        seg_slope_signed = dy / np.maximum(dx, 1.0)

        noise_mask = idx[1:] < rough_idx
        if np.sum(noise_mask) >= 2:
            sigma_seg = robust_sigma(seg_slope_signed[noise_mask])
        else:
            sigma_seg = robust_sigma(seg_slope_signed[: max(2, min(4, len(seg_slope_signed)))])

        amp_noise = robust_sigma(y_rdp[: min(pre_n, len(y_rdp))])

        rdp_seg_thr = float(seg_slope_k * max(sigma_seg, 1e-12))
        rdp_amp_thr = float(seg_amp_k * max(amp_noise, 1e-12))

        for j in range(len(seg_slope_signed) - 1):
            run = seg_slope_signed[j : j + 2]
            right_idx = idx[j + 2]
            if (
                np.all(run > weak_factor * rdp_seg_thr)
                and np.any(run > rdp_seg_thr)
                and sign * (y_rdp[right_idx] - baseline) > rdp_amp_thr
            ):
                cand = int(idx[j])
                break

        if cand is None:
            hits = np.where(
                (seg_slope_signed > rdp_seg_thr)
                & (sign * (y_rdp[idx[1:]] - baseline) > rdp_amp_thr)
            )[0]
            if hits.size > 0:
                cand = int(idx[hits[0]])

    coarse_idx = rough_idx
    max_lead_n = int(round(max_lead_us * 1e-6 * fs))
    if cand is not None and 0 <= (rough_idx - cand) <= max_lead_n:
        coarse_idx = min(rough_idx, cand)

    return {
        "threshold_rough_idx": rough_idx,
        "threshold_rough_t": float(base["rough_t"]),
        "threshold_baseline": baseline,
        "threshold_rough_thr": float(base["rough_thr"]),
        "rdp_candidate_idx": None if cand is None else int(cand),
        "coarse_idx": int(coarse_idx),
        "coarse_t": float(coarse_idx / fs),
        "rdp_info": rdp_info,
        "rdp_signed_seg_slope": seg_slope_signed,
        "rdp_slope_thr": rdp_seg_thr,
        "rdp_amp_thr": rdp_amp_thr,
        "polarity": pol,
        "sign": sign,
        "base_smooth": y_base,
    }


def _safe_ratio(a: float, b: float) -> float:
    return float(a / (b + 1e-12))


def _run_length_from(mask: np.ndarray, start: int) -> int:
    n = 0
    for i in range(int(start), len(mask)):
        if mask[i]:
            n += 1
        else:
            break
    return int(n)


def quantify_wavefront_metrics(result: dict[str, Any], *, fs: float = FS, step_win_us: float = 3.0) -> dict[str, Any]:
    y = np.asarray(result["x_smooth"], dtype=float)
    amp_dev = np.asarray(result["amp_dev"], dtype=float)
    slope_dev = np.asarray(result["slope_dev"], dtype=float)

    idx_trigger = int(result["idx_trigger"])
    idx_head = int(round(result["idx_head_float"]))
    search_i0 = int(result["search_i0"])
    search_i1 = int(result["search_i1"])
    fit_i0 = int(result["fit_i0"])
    fit_i1 = int(result["fit_i1"])

    amp_thr = float(result["amp_thr"])
    slope_thr = float(result["slope_thr"])
    baseline = float(result["baseline"])

    params = result.get("params", {})
    amp_k = float(params.get("amp_k", 1.0))
    slope_k = float(params.get("slope_k", 1.0))
    min_consecutive = int(params.get("min_consecutive", 1))
    sign = 1.0 if result.get("polarity", "positive") == "positive" else -1.0

    amp_noise_sigma = amp_thr / max(amp_k, 1e-12)
    slope_noise_sigma = slope_thr / max(slope_k, 1e-12)

    amp_at_trigger = float(amp_dev[idx_trigger])
    slope_at_trigger = float(slope_dev[idx_trigger])
    amp_ratio = _safe_ratio(amp_at_trigger, amp_thr)
    slope_ratio = _safe_ratio(slope_at_trigger, slope_thr)

    trigger_mask = (amp_dev > amp_thr) & (slope_dev > slope_thr)
    consecutive_len = _run_length_from(trigger_mask, idx_trigger)
    consecutive_us = consecutive_len / fs * 1e6

    step_n = max(3, int(round(step_win_us * 1e-6 * fs)))
    pre0, pre1 = max(0, idx_head - step_n), idx_head
    post0, post1 = idx_head, min(len(y), idx_head + step_n)
    pre_seg = sign * (y[pre0:pre1] - baseline)
    post_seg = sign * (y[post0:post1] - baseline)

    pre_median = float(np.median(pre_seg)) if pre_seg.size > 0 else 0.0
    post_median = float(np.median(post_seg)) if post_seg.size > 0 else 0.0
    step_height = post_median - pre_median
    step_sigma_ratio = _safe_ratio(step_height, amp_noise_sigma)

    pre_energy = float(np.mean(pre_seg**2) + 1e-12) if pre_seg.size > 0 else 1e-12
    post_energy = float(np.mean(post_seg**2) + 1e-12) if post_seg.size > 0 else 1e-12
    energy_ratio = float(post_energy / pre_energy)

    if search_i1 > search_i0:
        local_peak_amp = float(np.max(amp_dev[search_i0:search_i1]))
        local_peak_slope = float(np.max(slope_dev[search_i0:search_i1]))
    else:
        local_peak_amp = amp_at_trigger
        local_peak_slope = slope_at_trigger

    peak_amp_ratio = _safe_ratio(local_peak_amp, amp_thr)
    peak_slope_ratio = _safe_ratio(local_peak_slope, slope_thr)

    fit_a = float(result.get("fit_a", np.nan))
    fit_b = float(result.get("fit_b", np.nan))
    if fit_i1 - fit_i0 >= 5 and np.isfinite(fit_a):
        tt = np.arange(fit_i0, fit_i1, dtype=float)
        yy = y[fit_i0:fit_i1]
        y_hat = fit_a * tt + fit_b
        ss_res = float(np.sum((yy - y_hat) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2) + 1e-12)
        fit_r2 = float(1.0 - ss_res / ss_tot)
        fit_slope_ratio = _safe_ratio(sign * fit_a, slope_thr)
    else:
        fit_r2 = np.nan
        fit_slope_ratio = np.nan

    backtrack_us = float((idx_trigger - result["idx_head_float"]) / fs * 1e6)

    rdp_candidate_idx = result.get("rdp_candidate_idx", None)
    has_rdp_support = rdp_candidate_idx is not None
    if has_rdp_support:
        rdp_to_head_us = float((result["idx_head_float"] - int(rdp_candidate_idx)) / fs * 1e6)
        rdp_to_trigger_us = float((idx_trigger - int(rdp_candidate_idx)) / fs * 1e6)
    else:
        rdp_to_head_us = np.nan
        rdp_to_trigger_us = np.nan

    s_amp = np.clip((amp_ratio - 1.0) / 2.0, 0.0, 1.0)
    s_slope = np.clip((slope_ratio - 1.0) / 2.0, 0.0, 1.0)
    s_step = np.clip(step_sigma_ratio / 6.0, 0.0, 1.0)
    s_energy = np.clip(np.log10(max(energy_ratio, 1.0)) / 1.0, 0.0, 1.0)
    s_fit = np.clip(0.0 if not np.isfinite(fit_r2) else fit_r2, 0.0, 1.0)
    s_persist = np.clip(consecutive_len / max(2 * min_consecutive, 1), 0.0, 1.0)
    s_rdp = 1.0 if has_rdp_support else 0.5
    confidence = 100.0 * (
        0.22 * s_amp
        + 0.22 * s_slope
        + 0.18 * s_step
        + 0.12 * s_energy
        + 0.14 * s_fit
        + 0.08 * s_persist
        + 0.04 * s_rdp
    )

    head_value_signed = float(sign * (y[idx_head] - baseline))
    if search_i1 > search_i0:
        local_seg_signed = sign * (y[search_i0:search_i1] - baseline)
        peak_rel_idx = int(np.argmax(local_seg_signed))
        peak_idx = int(search_i0 + peak_rel_idx)
        peak_value_signed = float(local_seg_signed[peak_rel_idx])
        peak_value_raw = float(y[peak_idx])
    else:
        peak_idx = idx_head
        peak_value_signed = head_value_signed
        peak_value_raw = float(y[idx_head])

    head_to_peak_percent = float(100.0 * head_value_signed / (peak_value_signed + 1e-12))

    return {
        "amp_at_trigger": amp_at_trigger,
        "slope_at_trigger": slope_at_trigger,
        "amp_ratio": amp_ratio,
        "slope_ratio": slope_ratio,
        "consecutive_len": consecutive_len,
        "consecutive_us": consecutive_us,
        "pre_median": pre_median,
        "post_median": post_median,
        "step_height": step_height,
        "step_sigma_ratio": step_sigma_ratio,
        "pre_energy": pre_energy,
        "post_energy": post_energy,
        "energy_ratio": energy_ratio,
        "local_peak_amp": local_peak_amp,
        "local_peak_slope": local_peak_slope,
        "peak_amp_ratio": peak_amp_ratio,
        "peak_slope_ratio": peak_slope_ratio,
        "fit_r2": fit_r2,
        "fit_slope_ratio": fit_slope_ratio,
        "backtrack_us": backtrack_us,
        "has_rdp_support": has_rdp_support,
        "rdp_to_head_us": rdp_to_head_us,
        "rdp_to_trigger_us": rdp_to_trigger_us,
        "amp_noise_sigma": amp_noise_sigma,
        "slope_noise_sigma": slope_noise_sigma,
        "confidence": confidence,
        "head_value_signed": head_value_signed,
        "peak_idx": peak_idx,
        "peak_t_us": float(peak_idx / fs * 1e6),
        "peak_value_signed": peak_value_signed,
        "peak_value_raw": peak_value_raw,
        "head_to_peak_percent": head_to_peak_percent,
    }


def explain_wavefront(metrics: dict[str, Any], *, min_consecutive: int = 5) -> str:
    reasons: list[str] = []
    reasons.append(
        f"触发点幅值{'达到' if metrics['amp_ratio'] >= 1.0 else '仅为'}阈值的 {metrics['amp_ratio']:.2f} 倍"
    )
    reasons.append(
        f"触发点斜率{'达到' if metrics['slope_ratio'] >= 1.0 else '仅为'}阈值的 {metrics['slope_ratio']:.2f} 倍"
    )
    reasons.append(
        f"联合越阈{'连续维持' if metrics['consecutive_len'] >= min_consecutive else '仅维持'} {metrics['consecutive_len']} 点"
    )
    reasons.append(f"波头前后台阶高度为噪声的 {metrics['step_sigma_ratio']:.2f} 倍")
    reasons.append(f"波头后局部能量是波头前的 {metrics['energy_ratio']:.2f} 倍")
    if metrics["has_rdp_support"]:
        reasons.append(f"RDP 候选点领先触发点 {metrics['rdp_to_trigger_us']:.3f} us")
    reasons.append(f"综合置信度 = {metrics['confidence']:.1f}/100")
    return "；".join(reasons)


def aic_curve(x: np.ndarray, *, min_split: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    aic = np.full(n, np.inf, dtype=float)
    if n < 2 * int(min_split) + 1:
        return aic

    eps = 1e-12
    min_split = int(min_split)
    for k in range(min_split, n - min_split):
        v1 = np.var(x[:k]) + eps
        v2 = np.var(x[k:]) + eps
        aic[k] = k * np.log(v1) + (n - k - 1) * np.log(v2)
    return aic


def pick_head_by_local_aic(
    y: np.ndarray,
    baseline: float,
    idx_trigger: int,
    search_i0: int,
    search_i1: int,
    fs: float,
    *,
    sign: float = 1.0,
    aic_left_us: float = 6.0,
    aic_right_us: float = 10.0,
    aic_min_split: int = 5,
    aic_smooth_win: int = 5,
    aic_smooth_poly: int = 2,
) -> tuple[float, dict[str, Any]]:
    n = len(y)
    left_n = int(round(aic_left_us * 1e-6 * fs))
    right_n = int(round(aic_right_us * 1e-6 * fs))
    aic_i0 = max(search_i0, int(idx_trigger) - left_n)
    aic_i1 = min(search_i1, int(idx_trigger) + right_n)

    if aic_i1 - aic_i0 < max(2 * int(aic_min_split) + 1, 9):
        return float(idx_trigger), {
            "aic_enabled": True,
            "aic_i0": int(aic_i0),
            "aic_i1": int(aic_i1),
            "aic_k_local": None,
            "aic_min_value": np.nan,
            "aic_curve": np.array([], dtype=float),
            "aic_signal": np.array([], dtype=float),
            "aic_fallback": True,
        }

    x_local = sign * (y[aic_i0:aic_i1] - baseline)
    x_local = smooth_savgol(x_local, window_length=aic_smooth_win, polyorder=aic_smooth_poly)
    ref_n = max(3, min(int(aic_min_split), len(x_local) // 4 if len(x_local) >= 4 else len(x_local)))
    x_local = x_local - np.median(x_local[:ref_n])

    aic_vals = aic_curve(x_local, min_split=aic_min_split)
    if not np.any(np.isfinite(aic_vals)):
        return float(idx_trigger), {
            "aic_enabled": True,
            "aic_i0": int(aic_i0),
            "aic_i1": int(aic_i1),
            "aic_k_local": None,
            "aic_min_value": np.nan,
            "aic_curve": aic_vals,
            "aic_signal": x_local,
            "aic_fallback": True,
        }

    k_local = int(np.nanargmin(aic_vals))
    idx_head = float(np.clip(aic_i0 + k_local, 0.0, n - 1.0))
    return idx_head, {
        "aic_enabled": True,
        "aic_i0": int(aic_i0),
        "aic_i1": int(aic_i1),
        "aic_k_local": int(k_local),
        "aic_min_value": float(aic_vals[k_local]),
        "aic_curve": aic_vals,
        "aic_signal": x_local,
        "aic_fallback": False,
    }


def detect_wavefront_rdp(
    x: np.ndarray,
    *,
    fs: float = FS,
    pre_n: int = 600,
    rough_k: float = 5.0,
    rough_consecutive: int = 8,
    threshold_sg_window: int = 11,
    threshold_sg_poly: int = 3,
    rdp_left_us: float = 60.0,
    rdp_right_us: float = 85.0,
    rdp_epsilon: float = 0.008,
    rdp_sg_window: int = 7,
    rdp_sg_poly: int = 2,
    rdp_seg_slope_k: float = 3.5,
    rdp_seg_amp_k: float = 3.0,
    rdp_weak_factor: float = 0.40,
    search_left_us: float = 5.0,
    search_right_us: float = 10.0,
    noise_guard_us: float = 1.5,
    noise_win_us: float = 100.0,
    slope_win: int = 17,
    slope_poly: int = 2,
    amp_k: float = 3.5,
    slope_k: float = 2.0,
    min_consecutive: int = 5,
    fit_n: int = 26,
    polarity: str = "auto",
    aic_left_us: float = 6.0,
    aic_right_us: float = 8.0,
    aic_min_split: int = 5,
    aic_smooth_win: int = 5,
    aic_smooth_poly: int = 2,
    wavelet_enabled: bool = True,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    wavelet_beta: float = 3.0,
    wavelet_threshold_scale: float = 1.0,
    wavelet_use_level_dependent_sigma: bool = True,
    wavelet_mode: str = "symmetric",
) -> dict[str, Any]:
    x_raw = np.asarray(x, dtype=float)
    n = len(x_raw)

    if wavelet_enabled:
        x_wavelet, wavelet_info = wavelet_denoise_improved(
            x_raw,
            wavelet=wavelet,
            level=wavelet_level,
            beta=wavelet_beta,
            threshold_scale=wavelet_threshold_scale,
            use_level_dependent_sigma=wavelet_use_level_dependent_sigma,
            mode=wavelet_mode,
        )
    else:
        x_wavelet = x_raw.copy()
        wavelet_info = {
            "enabled": False,
            "wavelet": wavelet,
            "level": 0,
            "beta": wavelet_beta,
            "threshold_scale": wavelet_threshold_scale,
            "noise_sigma": 0.0,
            "use_level_dependent_sigma": wavelet_use_level_dependent_sigma,
            "mode": wavelet_mode,
        }

    rough_info = rdp_rough_locator(
        x=x_wavelet,
        fs=fs,
        pre_n=pre_n,
        threshold_sg_window=threshold_sg_window,
        threshold_sg_poly=threshold_sg_poly,
        rough_k=rough_k,
        rough_consecutive=rough_consecutive,
        rdp_left_us=rdp_left_us,
        rdp_right_us=rdp_right_us,
        rdp_epsilon=rdp_epsilon,
        rdp_sg_window=rdp_sg_window,
        rdp_sg_poly=rdp_sg_poly,
        seg_slope_k=rdp_seg_slope_k,
        seg_amp_k=rdp_seg_amp_k,
        weak_factor=rdp_weak_factor,
    )

    y = smooth_savgol(x_wavelet, threshold_sg_window, threshold_sg_poly)
    baseline = float(np.median(y[: min(pre_n, len(y))]))
    coarse_idx = int(rough_info["coarse_idx"])

    search_left_n = int(round(search_left_us * 1e-6 * fs))
    search_right_n = int(round(search_right_us * 1e-6 * fs))
    search_i0 = max(0, coarse_idx - search_left_n)
    search_i1 = min(n, coarse_idx + search_right_n)

    noise_guard_n = int(round(noise_guard_us * 1e-6 * fs))
    noise_win_n = int(round(noise_win_us * 1e-6 * fs))
    noise_i1 = max(5, search_i0 - noise_guard_n)
    noise_i0 = max(0, noise_i1 - noise_win_n)

    slope = moving_sg_slope(y, win=slope_win, poly=slope_poly)
    noise_y = y[noise_i0:noise_i1]
    noise_slope = slope[noise_i0:noise_i1]

    amp_noise = robust_sigma(noise_y) if noise_y.size > 0 else robust_sigma(y[: min(pre_n, len(y))])
    slope_noise = robust_sigma(noise_slope) if noise_slope.size > 0 else robust_sigma(
        slope[: min(pre_n, len(slope))]
    )
    slope_base = float(np.median(noise_slope)) if noise_slope.size > 0 else 0.0

    if polarity == "positive":
        sign, pol = 1.0, "positive"
    elif polarity == "negative":
        sign, pol = -1.0, "negative"
    else:
        sign, pol = detect_polarity_from_window(y, baseline, search_i0, search_i1)

    amp_dev = sign * (y - baseline)
    slope_dev = sign * (slope - slope_base)

    amp_thr = float(amp_k * max(amp_noise, 1e-12))
    slope_thr = float(slope_k * max(slope_noise, 1e-12))

    amp_mask = amp_dev > amp_thr
    slope_mask = slope_dev > slope_thr
    trigger_mask = amp_mask & slope_mask

    first_cross_idx = None
    hit_amp = np.where(amp_mask[search_i0:search_i1])[0]
    if hit_amp.size > 0:
        first_cross_idx = int(search_i0 + hit_amp[0])

    idx_trigger = None
    for i in range(search_i0, max(search_i0, search_i1 - min_consecutive + 1)):
        if np.all(trigger_mask[i : i + min_consecutive]):
            idx_trigger = int(i)
            break

    if idx_trigger is None:
        idx_trigger = first_cross_idx if first_cross_idx is not None else coarse_idx

    idx_head, aic_info = pick_head_by_local_aic(
        y=y,
        baseline=baseline,
        idx_trigger=idx_trigger,
        search_i0=search_i0,
        search_i1=search_i1,
        fs=fs,
        sign=sign,
        aic_left_us=aic_left_us,
        aic_right_us=aic_right_us,
        aic_min_split=aic_min_split,
        aic_smooth_win=aic_smooth_win,
        aic_smooth_poly=aic_smooth_poly,
    )

    fit_peak_idx = idx_trigger
    fit_i0 = int(aic_info["aic_i0"])
    fit_i1 = int(aic_info["aic_i1"])
    idx_head = float(np.clip(idx_head, 0.0, n - 1.0))

    result: dict[str, Any] = {
        "algorithm": "rdp_local_aic",
        "algorithm_label": "RDP-assisted wavelet + local AIC",
        "x_raw": x_raw,
        "x_wavelet": x_wavelet,
        "x_smooth": y,
        "baseline": baseline,
        "threshold_rough_idx": int(rough_info["threshold_rough_idx"]),
        "threshold_rough_t": float(rough_info["threshold_rough_t"]),
        "rdp_candidate_idx": rough_info["rdp_candidate_idx"],
        "coarse_idx": int(coarse_idx),
        "coarse_t": float(coarse_idx / fs),
        "idx_trigger": int(idx_trigger),
        "t_trigger": float(idx_trigger / fs),
        "idx_head": int(round(idx_head)),
        "idx_head_float": float(idx_head),
        "t_head": float(idx_head / fs),
        "first_cross_idx": first_cross_idx,
        "polarity": pol,
        "slope": slope,
        "amp_dev": amp_dev,
        "slope_dev": slope_dev,
        "amp_thr": amp_thr,
        "slope_thr": slope_thr,
        "search_i0": int(search_i0),
        "search_i1": int(search_i1),
        "noise_i0": int(noise_i0),
        "noise_i1": int(noise_i1),
        "fit_i0": int(fit_i0),
        "fit_i1": int(fit_i1),
        "fit_peak_idx": int(fit_peak_idx),
        "fit_a": float(np.nan),
        "fit_b": float(np.nan),
        "aic_i0": int(aic_info["aic_i0"]),
        "aic_i1": int(aic_info["aic_i1"]),
        "aic_k_local": aic_info["aic_k_local"],
        "aic_min_value": float(aic_info["aic_min_value"]) if np.isfinite(aic_info["aic_min_value"]) else np.nan,
        "aic_fallback": bool(aic_info["aic_fallback"]),
        "aic_curve": aic_info["aic_curve"],
        "aic_signal": aic_info["aic_signal"],
        "rdp_info": rough_info["rdp_info"],
        "rdp_slope_thr": float(rough_info["rdp_slope_thr"]) if np.isfinite(rough_info["rdp_slope_thr"]) else np.nan,
        "rdp_amp_thr": float(rough_info["rdp_amp_thr"]) if np.isfinite(rough_info["rdp_amp_thr"]) else np.nan,
        "wavelet_info": wavelet_info,
        "params": {
            "fs": fs,
            "pre_n": pre_n,
            "rough_k": rough_k,
            "rough_consecutive": rough_consecutive,
            "threshold_sg_window": threshold_sg_window,
            "threshold_sg_poly": threshold_sg_poly,
            "rdp_left_us": rdp_left_us,
            "rdp_right_us": rdp_right_us,
            "rdp_epsilon": rdp_epsilon,
            "rdp_sg_window": rdp_sg_window,
            "rdp_sg_poly": rdp_sg_poly,
            "rdp_seg_slope_k": rdp_seg_slope_k,
            "rdp_seg_amp_k": rdp_seg_amp_k,
            "rdp_weak_factor": rdp_weak_factor,
            "search_left_us": search_left_us,
            "search_right_us": search_right_us,
            "noise_guard_us": noise_guard_us,
            "noise_win_us": noise_win_us,
            "slope_win": slope_win,
            "slope_poly": slope_poly,
            "amp_k": amp_k,
            "slope_k": slope_k,
            "min_consecutive": min_consecutive,
            "fit_n": fit_n,
            "polarity": polarity,
            "aic_left_us": aic_left_us,
            "aic_right_us": aic_right_us,
            "aic_min_split": aic_min_split,
            "aic_smooth_win": aic_smooth_win,
            "aic_smooth_poly": aic_smooth_poly,
            "wavelet_enabled": wavelet_enabled,
            "wavelet": wavelet,
            "wavelet_level": wavelet_level,
            "wavelet_beta": wavelet_beta,
            "wavelet_threshold_scale": wavelet_threshold_scale,
            "wavelet_use_level_dependent_sigma": wavelet_use_level_dependent_sigma,
            "wavelet_mode": wavelet_mode,
        },
    }

    result["metrics"] = quantify_wavefront_metrics(result, fs=fs, step_win_us=3.0)
    result["summary_text"] = explain_wavefront(
        result["metrics"], min_consecutive=result["params"]["min_consecutive"]
    )
    return result


__all__ = [
    "DEFAULT_RDP_LOCAL_AIC_PARAMS",
    "FS",
    "detect_wavefront_rdp",
    "explain_wavefront",
    "quantify_wavefront_metrics",
]
