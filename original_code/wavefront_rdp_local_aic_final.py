import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FS = 4.2e6


# =========================================================
# 0. 数据读取与基础工具
# =========================================================
def load_csv_no_header(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：第一列索引，第二列信号值")

    idx = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)
    return idx, x, df


def time_vector(n: int, fs: float) -> np.ndarray:
    return np.arange(n) / fs


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

    window_length = int(window_length)
    if window_length < 5:
        window_length = 5
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
    """
    用 SG 导数代替逐点滑动线性拟合，速度更快。
    返回“每个采样点上的斜率”（单位：幅值/采样点）。
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 5:
        return np.gradient(y)

    win = int(win)
    if win < 5:
        win = 5
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


# =========================================================
# 0.5 前置改进小波阈值降噪
# =========================================================
def improved_threshold(coeff: np.ndarray, lam: float, beta: float) -> np.ndarray:
    """
    改进小波阈值函数：
        w_tilde = sgn(w) * (|w| - lam * exp(-beta*(|w|/lam - 1))), |w|>=lam
                = 0, |w|<lam
    """
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


def wavelet_denoise_improved(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    beta: float = 3.0,
    threshold_scale: float = 1.0,
    use_level_dependent_sigma: bool = True,
    mode: str = "symmetric",
):
    """
    前置改进小波降噪。
    建议默认：
        wavelet='db4', level=4, beta=3.0, threshold_scale=1.0

    说明：
    1. 用 cD1 的 MAD 估计全局噪声；
    2. 可选每层自适应估计 sigma_j；
    3. 仅对细节系数做改进阈值收缩，保留近似系数。
    """
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

    level = int(level)
    if level <= 0:
        level = min(4, max_level)
    level = min(level, max_level)

    coeffs = pywt.wavedec(x, wavelet=wavelet_obj, mode=mode, level=level)

    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    detail_coeffs = coeffs[1:]
    cD1 = coeffs[-1]
    noise_sigma_global = robust_sigma(cD1)

    new_coeffs = [coeffs[0]]
    level_sigmas = []
    level_lambdas = []

    for cD in detail_coeffs:
        sigma_j = robust_sigma(cD) if use_level_dependent_sigma else noise_sigma_global
        lam_j = threshold_scale * sigma_j * np.sqrt(2.0 * np.log(max(len(cD), 2)))

        level_sigmas.append(float(sigma_j))
        level_lambdas.append(float(lam_j))

        cD_new = improved_threshold(cD, lam=lam_j, beta=beta)
        new_coeffs.append(cD_new)

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


# =========================================================
# 1. RDP
# =========================================================
def rdp_indices(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    纯 NumPy 的 RDP 索引版本
    """
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

        inner = points[start + 1:end]
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


def detect_polarity_from_window(
    y: np.ndarray,
    baseline: float,
    i0: int,
    i1: int,
):
    seg = y[i0:i1] - baseline
    if seg.size == 0:
        return 1.0, "positive"

    sign = 1.0 if np.max(seg) >= abs(np.min(seg)) else -1.0
    pol = "positive" if sign > 0 else "negative"
    return sign, pol


def build_rdp_points_local(
    x: np.ndarray,
    center_idx: int,
    fs: float = FS,
    pre_n: int = 600,
    left_us: float = 60.0,
    right_us: float = 85.0,
    epsilon: float = 0.008,
    sg_window: int = 7,
    sg_poly: int = 2,
):
    """
    在 threshold 粗定位附近的“局部窗”内做 RDP，而不是整段信号。
    这能让 epsilon 更稳定。
    """
    x = np.asarray(x, dtype=float)
    y = smooth_savgol(x, sg_window, sg_poly)

    left_n = int(round(left_us * 1e-6 * fs))
    right_n = int(round(right_us * 1e-6 * fs))

    i0 = max(0, int(center_idx) - left_n)
    i1 = min(len(y), int(center_idx) + right_n)

    xx = np.arange(i0, i1, dtype=float)

    baseline = np.median(y[:min(pre_n, len(y))])
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


# =========================================================
# 2. 第一层粗定位：threshold
# =========================================================
def threshold_rough_locator(
    x: np.ndarray,
    fs: float = FS,
    sg_window: int = 11,
    sg_poly: int = 3,
    pre_n: int = 600,
    k: float = 8.0,
    min_consecutive: int = 8,
):
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


# =========================================================
# 3. 第二层粗定位：RDP 真正参与
# =========================================================
def rdp_rough_locator(
    x: np.ndarray,
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
):
    """
    先用 threshold 给一个初始粗定位 rough_idx，
    再在 rough_idx 附近做局部 RDP，
    最后用“RDP 线段斜率 + 幅值越噪声阈值”的规则，
    找到一个真正参与后续搜索窗中心的 rdp_candidate_idx。
    """
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
            sigma_seg = robust_sigma(seg_slope_signed[:max(2, min(4, len(seg_slope_signed)))])

        amp_noise = robust_sigma(y_rdp[:min(pre_n, len(y_rdp))])

        rdp_seg_thr = float(seg_slope_k * max(sigma_seg, 1e-12))
        rdp_amp_thr = float(seg_amp_k * max(amp_noise, 1e-12))

        for j in range(len(seg_slope_signed) - 1):
            run = seg_slope_signed[j:j + 2]
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


# 量化分析
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


def quantify_wavefront_metrics(result: dict, fs: float = FS, step_win_us: float = 3.0):
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

    # 由阈值反推噪声尺度
    amp_noise_sigma = amp_thr / max(amp_k, 1e-12)
    slope_noise_sigma = slope_thr / max(slope_k, 1e-12)

    # 1) 触发强度
    amp_at_trigger = float(amp_dev[idx_trigger])
    slope_at_trigger = float(slope_dev[idx_trigger])

    amp_ratio = _safe_ratio(amp_at_trigger, amp_thr)
    slope_ratio = _safe_ratio(slope_at_trigger, slope_thr)

    # 2) 连续越阈长度
    trigger_mask = (amp_dev > amp_thr) & (slope_dev > slope_thr)
    consecutive_len = _run_length_from(trigger_mask, idx_trigger)
    consecutive_us = consecutive_len / fs * 1e6

    # 3) 波头前后短窗对比
    step_n = max(3, int(round(step_win_us * 1e-6 * fs)))

    pre0 = max(0, idx_head - step_n)
    pre1 = idx_head
    post0 = idx_head
    post1 = min(len(y), idx_head + step_n)

    pre_seg = sign * (y[pre0:pre1] - baseline)
    post_seg = sign * (y[post0:post1] - baseline)

    pre_median = float(np.median(pre_seg)) if pre_seg.size > 0 else 0.0
    post_median = float(np.median(post_seg)) if post_seg.size > 0 else 0.0

    step_height = post_median - pre_median
    step_sigma_ratio = _safe_ratio(step_height, amp_noise_sigma)

    pre_energy = float(np.mean(pre_seg ** 2) + 1e-12) if pre_seg.size > 0 else 1e-12
    post_energy = float(np.mean(post_seg ** 2) + 1e-12) if post_seg.size > 0 else 1e-12
    energy_ratio = float(post_energy / pre_energy)

    # 4) 搜索窗内峰值强度
    if search_i1 > search_i0:
        local_peak_amp = float(np.max(amp_dev[search_i0:search_i1]))
        local_peak_slope = float(np.max(slope_dev[search_i0:search_i1]))
    else:
        local_peak_amp = amp_at_trigger
        local_peak_slope = slope_at_trigger

    peak_amp_ratio = _safe_ratio(local_peak_amp, amp_thr)
    peak_slope_ratio = _safe_ratio(local_peak_slope, slope_thr)

    # 5) 切线拟合质量
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

    # 6) RDP 支撑
    rdp_candidate_idx = result.get("rdp_candidate_idx", None)
    has_rdp_support = rdp_candidate_idx is not None

    if has_rdp_support:
        rdp_to_head_us = float((result["idx_head_float"] - int(rdp_candidate_idx)) / fs * 1e6)
        rdp_to_trigger_us = float((idx_trigger - int(rdp_candidate_idx)) / fs * 1e6)
    else:
        rdp_to_head_us = np.nan
        rdp_to_trigger_us = np.nan

    # 7) 一个启发式综合置信度（不是物理真值，只是便于排序/筛选）
    s_amp = np.clip((amp_ratio - 1.0) / 2.0, 0.0, 1.0)
    s_slope = np.clip((slope_ratio - 1.0) / 2.0, 0.0, 1.0)
    s_step = np.clip(step_sigma_ratio / 6.0, 0.0, 1.0)
    s_energy = np.clip(np.log10(max(energy_ratio, 1.0)) / 1.0, 0.0, 1.0)
    s_fit = np.clip(0.0 if not np.isfinite(fit_r2) else fit_r2, 0.0, 1.0)
    s_persist = np.clip(consecutive_len / max(2 * min_consecutive, 1), 0.0, 1.0)
    s_rdp = 1.0 if has_rdp_support else 0.5

    confidence = 100.0 * (
        0.22 * s_amp +
        0.22 * s_slope +
        0.18 * s_step +
        0.12 * s_energy +
        0.14 * s_fit +
        0.08 * s_persist +
        0.04 * s_rdp
    )

    # 4) 波头点值占波峰值百分比（同极性、相对基线）
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

    head_to_peak_percent = float(
        100.0 * head_value_signed / (peak_value_signed + 1e-12)
    )

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


def explain_wavefront(metrics: dict, min_consecutive: int = 5) -> str:
    reasons = []

    if metrics["amp_ratio"] >= 1.0:
        reasons.append(f"触发点幅值达到阈值的 {metrics['amp_ratio']:.2f} 倍")
    else:
        reasons.append(f"触发点幅值仅为阈值的 {metrics['amp_ratio']:.2f} 倍")

    if metrics["slope_ratio"] >= 1.0:
        reasons.append(f"触发点斜率达到阈值的 {metrics['slope_ratio']:.2f} 倍")
    else:
        reasons.append(f"触发点斜率仅为阈值的 {metrics['slope_ratio']:.2f} 倍")

    if metrics["consecutive_len"] >= min_consecutive:
        reasons.append(f"联合越阈连续维持 {metrics['consecutive_len']} 点")
    else:
        reasons.append(f"联合越阈仅维持 {metrics['consecutive_len']} 点")

    reasons.append(f"波头前后台阶高度为噪声的 {metrics['step_sigma_ratio']:.2f} 倍")
    reasons.append(f"波头后局部能量是波头前的 {metrics['energy_ratio']:.2f} 倍")

    if metrics["has_rdp_support"]:
        reasons.append(f"RDP 候选点领先触发点 {metrics['rdp_to_trigger_us']:.3f} us")

    reasons.append(f"综合置信度 = {metrics['confidence']:.1f}/100")
    return "；".join(reasons)


def aic_curve(x: np.ndarray, min_split: int = 5) -> np.ndarray:
    """
    经典 AIC 变点曲线：
        AIC(k) = k * ln(var(x[:k])) + (N-k-1) * ln(var(x[k:]))
    取 AIC 最小点作为变点位置。
    """
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
    sign: float = 1.0,
    aic_left_us: float = 6.0,
    aic_right_us: float = 10.0,
    aic_min_split: int = 5,
    aic_smooth_win: int = 5,
    aic_smooth_poly: int = 2,
):
    """
    在 idx_trigger 附近做局部 AIC 变点检测。
    返回：
        idx_head_float, aic_info
    """
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
    idx_head = float(aic_i0 + k_local)
    idx_head = float(np.clip(idx_head, 0.0, n - 1.0))

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

# =========================================================
# 4. 精定位：coarse_idx -> 搜索窗 -> 斜率/幅值触发 -> 局部 AIC 变点
# =========================================================
def detect_wavefront_rdp(
    x: np.ndarray,
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
    slope_k: float = 2,
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
):
    x_raw = np.asarray(x, dtype=float)
    n = len(x_raw)

    # ---------- 前置改进小波降噪 ----------
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
    baseline = float(np.median(y[:min(pre_n, len(y))]))
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

    amp_noise = robust_sigma(noise_y) if noise_y.size > 0 else robust_sigma(y[:min(pre_n, len(y))])
    slope_noise = robust_sigma(noise_slope) if noise_slope.size > 0 else robust_sigma(slope[:min(pre_n, len(slope))])
    slope_base = float(np.median(noise_slope)) if noise_slope.size > 0 else 0.0

    if polarity == "positive":
        sign = 1.0
        pol = "positive"
    elif polarity == "negative":
        sign = -1.0
        pol = "negative"
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
        if np.all(trigger_mask[i:i + min_consecutive]):
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
    a, b = np.nan, np.nan

    idx_head = float(np.clip(idx_head, 0.0, n - 1.0))

    result = {
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
        "fit_a": float(a),
        "fit_b": float(b),
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
    return result


# =========================================================
# 5. 输出
# =========================================================
def print_result(result: dict, fs: float = FS, end_name: str = ""):
    prefix = f"[{end_name}] " if end_name else ""

    print("=" * 78)
    print(f"{prefix}检测结果（前置改进小波降噪 -> RDP 参与粗定位 -> threshold/slope 触发 -> 局部 AIC 变点）")
    print("=" * 78)

    wavelet_info = result.get("wavelet_info", {})
    if wavelet_info.get("enabled", False):
        print(f"{prefix}前置小波降噪 = 开启")
        print(f"{prefix}wavelet = {wavelet_info.get('wavelet')}")
        print(f"{prefix}level = {wavelet_info.get('level')}")
        print(f"{prefix}beta = {wavelet_info.get('beta')}")
        print(f"{prefix}threshold_scale = {wavelet_info.get('threshold_scale')}")
        print(f"{prefix}noise_sigma = {wavelet_info.get('noise_sigma'):.6e}")
        print(f"{prefix}level_dependent_sigma = {wavelet_info.get('use_level_dependent_sigma')}")
        print(f"{prefix}boundary_mode = {wavelet_info.get('mode')}")
    else:
        print(f"{prefix}前置小波降噪 = 关闭")

    print("-" * 78)
    print(f"{prefix}极性 = {result['polarity']}")
    print("-" * 78)
    print(f"{prefix}threshold 粗定位 idx = {result['threshold_rough_idx']}")
    print(f"{prefix}threshold 粗定位 t = {result['threshold_rough_t'] * 1e6:.6f} us")
    print(f"{prefix}RDP 候选点 idx = {result['rdp_candidate_idx']}")
    print(f"{prefix}融合后 coarse_idx = {result['coarse_idx']}")
    print(f"{prefix}融合后 coarse_t = {result['coarse_t'] * 1e6:.6f} us")
    print("-" * 78)
    print(f"{prefix}触发点 idx_trigger = {result['idx_trigger']}")
    print(f"{prefix}触发点 t_trigger = {result['t_trigger'] * 1e6:.6f} us")
    print(f"{prefix}最终波头样点 idx_head = {result['idx_head']}")
    print(f"{prefix}最终波头时刻 t_head = {result['t_head'] * 1e6:.6f} us")
    print(f"{prefix}首次幅值越阈 first_cross_idx = {result['first_cross_idx']}")
    print("-" * 78)
    print(f"{prefix}RDP 线段斜率阈值 = {result['rdp_slope_thr']:.6e}")
    print(f"{prefix}RDP 幅值阈值 = {result['rdp_amp_thr']:.6e}")
    print(f"{prefix}幅值阈值 amp_thr = {result['amp_thr']:.6e}")
    print(f"{prefix}斜率阈值 slope_thr = {result['slope_thr']:.6e}")
    print(f"{prefix}搜索窗 = [{result['search_i0'] / fs * 1e6:.3f}, {result['search_i1'] / fs * 1e6:.3f}] us")
    print(f"{prefix}噪声窗 = [{result['noise_i0'] / fs * 1e6:.3f}, {result['noise_i1'] / fs * 1e6:.3f}] us")
    print(f"{prefix}AIC 搜索窗 = [{result['aic_i0'] / fs * 1e6:.3f}, {result['aic_i1'] / fs * 1e6:.3f}] us")
    print(f"{prefix}AIC 是否回退到 idx_trigger = {result['aic_fallback']}")
    print("=" * 78)

    metrics = result.get("metrics", {})
    if metrics:
        min_consecutive = result.get("params", {}).get("min_consecutive", 5)

        print("-" * 78)
        print(f"{prefix}[量化指标]")
        print(f"{prefix}触发点幅值 = {metrics['amp_at_trigger']:.6e}")
        print(f"{prefix}触发点斜率 = {metrics['slope_at_trigger']:.6e}")
        print(f"{prefix}幅值越阈倍数 M_amp = {metrics['amp_ratio']:.3f}")
        print(f"{prefix}斜率越阈倍数 M_slope = {metrics['slope_ratio']:.3f}")
        print(f"{prefix}联合越阈连续点数 = {metrics['consecutive_len']} "
              f"(≈ {metrics['consecutive_us']:.3f} us)")
        print(f"{prefix}波头前后台阶高度 = {metrics['step_height']:.6e}")
        print(f"{prefix}台阶高度噪声比 M_step = {metrics['step_sigma_ratio']:.3f}")
        print(f"{prefix}前后能量比 R_energy = {metrics['energy_ratio']:.3f}")
        print(f"{prefix}搜索窗内峰值幅值越阈倍数 = {metrics['peak_amp_ratio']:.3f}")
        print(f"{prefix}搜索窗内峰值斜率越阈倍数 = {metrics['peak_slope_ratio']:.3f}")
        print(f"{prefix}AIC 最小值 = {result['aic_min_value']:.6e}")
        print(f"{prefix}AIC 局部最优点 = {result['aic_k_local']}")
        print(f"{prefix}触发点到波头回推距离 = {metrics['backtrack_us']:.3f} us")

        if metrics["has_rdp_support"]:
            print(f"{prefix}RDP->波头距离 = {metrics['rdp_to_head_us']:.3f} us")
            print(f"{prefix}RDP->触发点距离 = {metrics['rdp_to_trigger_us']:.3f} us")
        else:
            print(f"{prefix}RDP 支撑 = 无")

        print(f"{prefix}综合置信度 = {metrics['confidence']:.1f}/100")
        print(f"{prefix}判定说明 = {explain_wavefront(metrics, min_consecutive=min_consecutive)}")

# =========================================================
# 6. 绘图
# =========================================================
def _safe_set_ylim(ax, x, y, x_left, x_right, pad_ratio=0.08):
    mask = (x >= x_left) & (x <= x_right)
    yv = np.asarray(y)[mask]
    if yv.size == 0:
        return

    y_min = np.nanmin(yv)
    y_max = np.nanmax(yv)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return

    if abs(y_max - y_min) < 1e-15:
        pad = max(1e-6, abs(y_max) * 0.1 + 1e-6)
    else:
        pad = (y_max - y_min) * pad_ratio

    ax.set_ylim(y_min - pad, y_max + pad)


def _plot_one_result_row_global(ax_row, result: dict, end_label: str, fs: float = FS):
    x_raw = result["x_raw"]
    x_wavelet = result["x_wavelet"]
    x_smooth = result["x_smooth"]
    baseline = result["baseline"]
    threshold_rough_idx = result["threshold_rough_idx"]
    rdp_candidate_idx = result["rdp_candidate_idx"]
    coarse_idx = result["coarse_idx"]
    idx_trigger = result["idx_trigger"]
    idx_head = result["idx_head_float"]
    slope_dev = result["slope_dev"]
    rdp_info = result["rdp_info"]

    t = time_vector(len(x_raw), fs) * 1e6
    threshold_rough_t_us = threshold_rough_idx / fs * 1e6
    coarse_t_us = coarse_idx / fs * 1e6
    trigger_t_us = idx_trigger / fs * 1e6
    head_t_us = idx_head / fs * 1e6
    search_start_us = result["search_i0"] / fs * 1e6
    search_end_us = result["search_i1"] / fs * 1e6
    noise_start_us = result["noise_i0"] / fs * 1e6
    noise_end_us = result["noise_i1"] / fs * 1e6

    ax1, ax2, ax3, ax4 = ax_row

    ax1.plot(t, x_raw, linewidth=0.9, alpha=0.45, label="Raw signal")
    ax1.plot(t, x_wavelet, linewidth=1.0, alpha=0.95, label="Wavelet denoised")
    turn_idx = rdp_info["turn_idx"]
    ax1.scatter(turn_idx / fs * 1e6, x_smooth[turn_idx], s=10, label="RDP points")
    ax1.axvline(threshold_rough_t_us, linestyle="--", label="Threshold rough")
    if rdp_candidate_idx is not None:
        ax1.axvline(rdp_candidate_idx / fs * 1e6, linestyle="--", color="tab:green", label="RDP candidate")
    ax1.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax1.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax1.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax1.axvspan(noise_start_us, noise_end_us, alpha=0.08, label="Noise window")
    ax1.set_title(f"{end_label} - Raw / Wavelet / RDP")
    ax1.set_xlabel("Time (us)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(t, x_wavelet, linewidth=0.9, alpha=0.6, label="Wavelet denoised")
    ax2.plot(t, x_smooth, linewidth=1.0, label="SG smoothed")
    ax2.axhline(baseline, linestyle="--", label="Baseline")
    if result["polarity"] == "positive":
        ax2.axhline(baseline + result["amp_thr"], linestyle=":", label="Amp threshold")
    else:
        ax2.axhline(baseline - result["amp_thr"], linestyle=":", label="Amp threshold")
    ax2.axvline(threshold_rough_t_us, linestyle="--", label="Threshold rough")
    if rdp_candidate_idx is not None:
        ax2.axvline(rdp_candidate_idx / fs * 1e6, linestyle="--", color="tab:green", label="RDP candidate")
    ax2.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax2.axvline(trigger_t_us, linestyle="--", color="tab:purple", label="Trigger")
    ax2.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax2.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax2.set_title(f"{end_label} - Denoised / Smoothed signal")
    ax2.set_xlabel("Time (us)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="best", fontsize=8)

    ax3.plot(t, slope_dev, linewidth=1.0, label="Signed slope")
    ax3.axhline(result["slope_thr"], linestyle="--", label="Slope threshold")
    ax3.axvline(threshold_rough_t_us, linestyle="--", label="Threshold rough")
    if rdp_candidate_idx is not None:
        ax3.axvline(rdp_candidate_idx / fs * 1e6, linestyle="--", color="tab:green", label="RDP candidate")
    ax3.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax3.axvline(trigger_t_us, linestyle="--", color="tab:purple", label="Trigger")
    ax3.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax3.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax3.axvspan(noise_start_us, noise_end_us, alpha=0.08, label="Noise window")
    ax3.set_title(f"{end_label} - Slope feature")
    ax3.set_xlabel("Time (us)")
    ax3.set_ylabel("Feature")
    ax3.legend(loc="best", fontsize=8)

    aic_i0 = result["aic_i0"]
    aic_i1 = result["aic_i1"]
    pad_l = 60
    pad_r = 90
    z0 = max(0, aic_i0 - pad_l)
    z1 = min(len(x_raw), aic_i1 + pad_r)

    ax4.plot(t[z0:z1], x_raw[z0:z1], linewidth=0.9, alpha=0.35, label="Raw")
    ax4.plot(t[z0:z1], x_wavelet[z0:z1], linewidth=1.0, alpha=0.8, label="Wavelet denoised")
    ax4.plot(t[z0:z1], x_smooth[z0:z1], linewidth=1.2, label="SG smoothed")
    ax4.axhline(baseline, linestyle="--", label="Baseline")
    ax4.axvspan(aic_i0 / fs * 1e6, aic_i1 / fs * 1e6, alpha=0.12, color="tab:cyan", label="AIC window")
    ax4.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax4.axvline(trigger_t_us, linestyle="--", color="tab:purple", label="Trigger")
    ax4.axvline(head_t_us, linestyle="--", color="orange", label="AIC head")
    ax4.scatter([head_t_us], [x_smooth[int(round(idx_head))]], s=28, zorder=5, label="AIC change point")
    ax4.set_title(f"{end_label} - Zoom & local AIC")
    ax4.set_xlabel("Time (us)")
    ax4.set_ylabel("Amplitude")
    ax4.legend(loc="best", fontsize=8)


def _plot_one_result_row_local(
    ax_row,
    result: dict,
    end_label: str,
    fs: float = FS,
    x_left: float = None,
    x_right: float = None,
):
    x_raw = result["x_raw"]
    x_wavelet = result["x_wavelet"]
    x_smooth = result["x_smooth"]
    baseline = result["baseline"]
    threshold_rough_idx = result["threshold_rough_idx"]
    rdp_candidate_idx = result["rdp_candidate_idx"]
    coarse_idx = result["coarse_idx"]
    idx_trigger = result["idx_trigger"]
    idx_head = result["idx_head_float"]
    slope_dev = result["slope_dev"]
    rdp_info = result["rdp_info"]

    t = time_vector(len(x_raw), fs) * 1e6
    threshold_rough_t_us = threshold_rough_idx / fs * 1e6
    coarse_t_us = coarse_idx / fs * 1e6
    trigger_t_us = idx_trigger / fs * 1e6
    head_t_us = idx_head / fs * 1e6
    search_start_us = result["search_i0"] / fs * 1e6
    search_end_us = result["search_i1"] / fs * 1e6
    noise_start_us = result["noise_i0"] / fs * 1e6
    noise_end_us = result["noise_i1"] / fs * 1e6

    if x_left is None:
        x_left = t[0]
    if x_right is None:
        x_right = t[-1]

    ax1, ax2, ax3, ax4 = ax_row

    ax1.plot(t, x_raw, linewidth=0.9, alpha=0.40, label="Raw signal")
    ax1.plot(t, x_wavelet, linewidth=1.0, alpha=0.95, label="Wavelet denoised")
    ax1.plot(t, x_smooth, linewidth=1.0, alpha=0.95, label="SG smoothed")
    turn_idx = rdp_info["turn_idx"]
    turn_idx = turn_idx[(turn_idx >= 0) & (turn_idx < len(x_smooth))]
    if len(turn_idx) > 0:
        turn_mask = (t[turn_idx] >= x_left) & (t[turn_idx] <= x_right)
        ax1.scatter(
            t[turn_idx][turn_mask],
            x_smooth[turn_idx][turn_mask],
            s=10,
            label="RDP points",
        )
    ax1.axvline(threshold_rough_t_us, linestyle="--", label="Threshold rough")
    if rdp_candidate_idx is not None:
        ax1.axvline(rdp_candidate_idx / fs * 1e6, linestyle="--", color="tab:green", label="RDP candidate")
    ax1.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax1.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax1.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax1.axvspan(noise_start_us, noise_end_us, alpha=0.08, label="Noise window")
    ax1.set_title(f"{end_label} - Raw / Wavelet / SG / RDP")
    ax1.set_xlabel("Time (us)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(x_left, x_right)
    _safe_set_ylim(ax1, t, x_smooth, x_left, x_right)
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(t, x_wavelet, linewidth=0.9, alpha=0.6, label="Wavelet denoised")
    ax2.plot(t, x_smooth, linewidth=1.0, label="SG smoothed")
    ax2.axhline(baseline, linestyle="--", label="Baseline")
    if result["polarity"] == "positive":
        ax2.axhline(baseline + result["amp_thr"], linestyle=":", label="Amp threshold")
    else:
        ax2.axhline(baseline - result["amp_thr"], linestyle=":", label="Amp threshold")
    ax2.axvline(threshold_rough_t_us, linestyle="--", label="Threshold rough")
    if rdp_candidate_idx is not None:
        ax2.axvline(rdp_candidate_idx / fs * 1e6, linestyle="--", color="tab:green", label="RDP candidate")
    ax2.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax2.axvline(trigger_t_us, linestyle="--", color="tab:purple", label="Trigger")
    ax2.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax2.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax2.set_title(f"{end_label} - Denoised / Smoothed signal")
    ax2.set_xlabel("Time (us)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(x_left, x_right)
    _safe_set_ylim(ax2, t, x_smooth, x_left, x_right)
    ax2.legend(loc="best", fontsize=8)

    ax3.plot(t, slope_dev, linewidth=1.0, label="Signed slope")
    ax3.axhline(result["slope_thr"], linestyle="--", label="Slope threshold")
    ax3.axvline(threshold_rough_t_us, linestyle="--", label="Threshold rough")
    if rdp_candidate_idx is not None:
        ax3.axvline(rdp_candidate_idx / fs * 1e6, linestyle="--", color="tab:green", label="RDP candidate")
    ax3.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax3.axvline(trigger_t_us, linestyle="--", color="tab:purple", label="Trigger")
    ax3.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax3.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax3.axvspan(noise_start_us, noise_end_us, alpha=0.08, label="Noise window")
    ax3.set_title(f"{end_label} - Slope feature")
    ax3.set_xlabel("Time (us)")
    ax3.set_ylabel("Feature")
    ax3.set_xlim(x_left, x_right)
    mask3 = (t >= x_left) & (t <= x_right)
    y3 = slope_dev[mask3]
    if y3.size > 0:
        y3_min = min(np.nanmin(y3), result["slope_thr"])
        y3_max = max(np.nanmax(y3), result["slope_thr"])
        pad3 = max((y3_max - y3_min) * 0.08, 1e-9)
        ax3.set_ylim(y3_min - pad3, y3_max + pad3)
    ax3.legend(loc="best", fontsize=8)

    ax4.plot(t, x_raw, linewidth=0.9, alpha=0.35, label="Raw")
    ax4.plot(t, x_wavelet, linewidth=1.0, alpha=0.8, label="Wavelet denoised")
    ax4.plot(t, x_smooth, linewidth=1.2, label="SG smoothed")
    ax4.axhline(baseline, linestyle="--", label="Baseline")
    ax4.axvspan(result["aic_i0"] / fs * 1e6, result["aic_i1"] / fs * 1e6, alpha=0.12, color="tab:cyan", label="AIC window")
    ax4.axvline(coarse_t_us, linestyle="--", color="tab:red", label="Coarse idx")
    ax4.axvline(trigger_t_us, linestyle="--", color="tab:purple", label="Trigger")
    ax4.axvline(head_t_us, linestyle="--", color="orange", label="AIC head")
    ax4.scatter([head_t_us], [x_smooth[int(round(idx_head))]], s=28, zorder=5, label="AIC change point")
    ax4.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax4.set_title(f"{end_label} - Zoom & local AIC")
    ax4.set_xlabel("Time (us)")
    ax4.set_ylabel("Amplitude")
    ax4.set_xlim(x_left, x_right)

    mask4 = (t >= x_left) & (t <= x_right)
    y4_list = [x_smooth[mask4], np.array([baseline])]
    y4 = np.concatenate(y4_list) if len(y4_list) > 0 else x_smooth[mask4]
    if y4.size > 0:
        y4_min = np.nanmin(y4)
        y4_max = np.nanmax(y4)
        pad4 = max((y4_max - y4_min) * 0.08, 1e-9)
        ax4.set_ylim(y4_min - pad4, y4_max + pad4)
    ax4.legend(loc="best", fontsize=8)


def plot_result_ab_global(
    result_a: dict,
    result_b: dict,
    file_a: str,
    file_b: str,
    fs: float = FS,
    title_prefix: str = "RDP-assisted wavefront detection",
    save_dir: str = "fig",
):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), dpi=200)
    plt.subplots_adjust(wspace=0.28, hspace=0.40)

    _plot_one_result_row_global(axes[0, :], result_a, end_label="A端", fs=fs)
    _plot_one_result_row_global(axes[1, :], result_b, end_label="B端", fs=fs)

    t_head_a_us = result_a["t_head"] * 1e6
    coarse_a_us = result_a["coarse_t"] * 1e6
    t_head_b_us = result_b["t_head"] * 1e6
    coarse_b_us = result_b["coarse_t"] * 1e6

    fig.suptitle(
        f"{title_prefix} - Global view\n"
        f"A端: final={t_head_a_us:.3f} us, coarse={coarse_a_us:.3f} us    |    "
        f"B端: final={t_head_b_us:.3f} us, coarse={coarse_b_us:.3f} us",
        fontweight="bold",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(save_dir, exist_ok=True)
    base_a = os.path.splitext(os.path.basename(file_a))[0]
    base_b = os.path.splitext(os.path.basename(file_b))[0]
    save_name = f"{base_a}__{base_b}_AB_rdp_assisted_wavelet_global.jpg"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"全局图已保存到: {save_path}")
    return save_path


def plot_result_ab_local(
    result_a: dict,
    result_b: dict,
    file_a: str,
    file_b: str,
    fs: float = FS,
    title_prefix: str = "RDP-assisted wavefront detection",
    plot_pad_ratio: float = 0.20,
    save_dir: str = "fig",
):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=True, dpi=200)
    plt.subplots_adjust(wspace=0.28, hspace=0.40)

    t_a = time_vector(len(result_a["x_raw"]), fs) * 1e6
    t_b = time_vector(len(result_b["x_raw"]), fs) * 1e6

    a_start = result_a["search_i0"] / fs * 1e6
    a_end = result_a["search_i1"] / fs * 1e6
    b_start = result_b["search_i0"] / fs * 1e6
    b_end = result_b["search_i1"] / fs * 1e6

    common_start = min(a_start, b_start)
    common_end = max(a_end, b_end)
    common_width = max(common_end - common_start, 1e-9)
    pad_us = common_width * plot_pad_ratio

    global_left = max(0.0, common_start - pad_us)
    global_right = min(max(t_a[-1], t_b[-1]), common_end + pad_us)

    _plot_one_result_row_local(
        axes[0, :],
        result_a,
        end_label="A端",
        fs=fs,
        x_left=global_left,
        x_right=global_right,
    )
    _plot_one_result_row_local(
        axes[1, :],
        result_b,
        end_label="B端",
        fs=fs,
        x_left=global_left,
        x_right=global_right,
    )

    t_head_a_us = result_a["t_head"] * 1e6
    coarse_a_us = result_a["coarse_t"] * 1e6
    t_head_b_us = result_b["t_head"] * 1e6
    coarse_b_us = result_b["coarse_t"] * 1e6

    fig.suptitle(
        f"{title_prefix} - Local zoom\n"
        f"A端: final={t_head_a_us:.3f} us, coarse={coarse_a_us:.3f} us    |    "
        f"B端: final={t_head_b_us:.3f} us, coarse={coarse_b_us:.3f} us",
        fontweight="bold",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(save_dir, exist_ok=True)
    base_a = os.path.splitext(os.path.basename(file_a))[0]
    base_b = os.path.splitext(os.path.basename(file_b))[0]
    save_name = f"{base_a}__{base_b}_AB_rdp_assisted_wavelet_local.jpg"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"局部图已保存到: {save_path}")
    return save_path


def plot_result_ab(
    result_a: dict,
    result_b: dict,
    file_a: str,
    file_b: str,
    fs: float = FS,
    title_prefix: str = "RDP-assisted wavefront detection",
    plot_pad_ratio: float = 0.20,
    save_dir: str = "fig",
):
    global_path = plot_result_ab_global(
        result_a=result_a,
        result_b=result_b,
        file_a=file_a,
        file_b=file_b,
        fs=fs,
        title_prefix=title_prefix,
        save_dir=save_dir,
    )

    local_path = plot_result_ab_local(
        result_a=result_a,
        result_b=result_b,
        file_a=file_a,
        file_b=file_b,
        fs=fs,
        title_prefix=title_prefix,
        plot_pad_ratio=plot_pad_ratio,
        save_dir=save_dir,
    )

    return {"global": global_path, "local": local_path}


# =========================================================
# 7. 主流程
# =========================================================
def summary_single(file_name: str, fs: float = FS):
    idx, x, df = load_csv_no_header(file_name)

    result = detect_wavefront_rdp(
        x=x,
        fs=fs,
        pre_n=600,

        # 前置改进小波降噪
        wavelet_enabled=True,
        wavelet="db4",
        wavelet_level=4,
        wavelet_beta=3.0,
        wavelet_threshold_scale=1.0,
        wavelet_use_level_dependent_sigma=True,
        wavelet_mode="symmetric",

        # 后续原有流程
        rough_k=8.0,
        rough_consecutive=8,
        threshold_sg_window=8,
        threshold_sg_poly=3,
        rdp_left_us=60.0,
        rdp_right_us=85.0,
        rdp_epsilon=0.008,
        rdp_sg_window=7,
        rdp_sg_poly=2,
        rdp_seg_slope_k=3.5,
        rdp_seg_amp_k=3.0,
        rdp_weak_factor=0.40,
        search_left_us=4.0,
        search_right_us=20.0,
        noise_guard_us=1.5,
        noise_win_us=10.0,
        slope_win=10,
        slope_poly=2,
        amp_k=2.8,
        slope_k=3,
        min_consecutive=3,
        fit_n=26,
        polarity="auto",
        aic_left_us=6.0,
        aic_right_us=10.0,
        aic_min_split=3,
        aic_smooth_win=4,
        aic_smooth_poly=2,
    )

    return result


def summary_ab(file_a: str, file_b: str, fs: float = FS, save_dir: str = "fig"):
    print("\n" + "#" * 80)
    print("A端结果")
    print("#" * 80)
    result_a = summary_single(file_a, fs=fs)
    print_result(result_a, fs=fs, end_name="A端")

    print("\n" + "#" * 80)
    print("B端结果")
    print("#" * 80)
    result_b = summary_single(file_b, fs=fs)
    print_result(result_b, fs=fs, end_name="B端")

    save_paths = plot_result_ab(
        result_a=result_a,
        result_b=result_b,
        file_a=file_a,
        file_b=file_b,
        fs=fs,
        title_prefix="RDP-assisted rounded-wavefront detection with improved wavelet denoising + local AIC",
        plot_pad_ratio=0.20,
        save_dir=save_dir,
    )

    return result_a, result_b, save_paths


# =========================================================
# 8. 文件名匹配规则
# =========================================================
FILENAME_PATTERN = re.compile(
    r"""
    ^
    (?P<date>\d{4}-\d{2}-\d{2})_
    (?P<time>\d{2}-\d{2}-\d{2})_
    (?P<head>\d+)
    -
    (?P<tail1>\d+)
    -
    (?P<tail2>\d+)
    \.csv
    $
    """,
    re.VERBOSE | re.IGNORECASE
)


def extract_match_key(filename: str):
    """
    从文件名中提取匹配字段，例如：
    2026-03-13_11-56-36_321-686-374.csv
    -> 11-56-36_321
    """
    name = os.path.basename(filename)
    m = FILENAME_PATTERN.match(name)
    if not m:
        return None
    return f"{m.group('time')}_{m.group('head')}"


def scan_and_group_csvs(folder: str):
    groups = defaultdict(list)

    for name in os.listdir(folder):
        if not name.lower().endswith(".csv"):
            continue

        key = extract_match_key(name)
        if key is None:
            print(f"[跳过] 文件名不符合规则: {name}")
            continue

        groups[key].append(name)

    for key in groups:
        groups[key] = sorted(groups[key])

    return dict(sorted(groups.items(), key=lambda x: x[0]))


def build_pairs(folder: str, target_key: str = None):
    groups = scan_and_group_csvs(folder)
    pairs = []

    for key, names in groups.items():
        if target_key is not None and key != target_key:
            continue

        if len(names) < 2:
            print(f"[跳过] 匹配键 {key} 只有 1 个文件，无法配对")
            continue

        if len(names) == 2:
            paira = os.path.join(folder, names[0])
            pairb = os.path.join(folder, names[1])
            pairs.append((key, paira, pairb))
            continue

        if len(names) % 2 != 0:
            print(f"[警告] 匹配键 {key} 下有 {len(names)} 个文件，最后一个将被忽略")

        for i in range(0, len(names) - 1, 2):
            paira = os.path.join(folder, names[i])
            pairb = os.path.join(folder, names[i + 1])
            sub_key = f"{key}__pair{i // 2 + 1}"
            pairs.append((sub_key, paira, pairb))

    return pairs

def build_save_dir_with_dt(save_root: str, match_key: str, result_a: dict, result_b: dict):
    """
    根据 A/B 两端最终波头时刻，生成带时间差后缀的保存文件夹名。
    例如：
        11-52-46_933__dt_5.23us
    """
    dt_us = abs(result_a["t_head"] - result_b["t_head"]) * 1e6
    folder_name = f"{match_key}_{dt_us:.2f}us-3"
    save_dir = os.path.join(save_root, folder_name)
    return save_dir, dt_us

def run_summary_for_pairs(pairs, fs: float, save_root: str = "fig"):
    if not pairs:
        print("没有找到可处理的数据对。")
        return

    os.makedirs(save_root, exist_ok=True)

    total = len(pairs)
    print(f"共找到 {total} 对数据。")

    for i, (match_key, paira, pairb) in enumerate(pairs, start=1):
        print("=" * 80)
        print(f"[{i}/{total}] 正在处理匹配键: {match_key}")
        print(f"paira: {os.path.basename(paira)}")
        print(f"pairb: {os.path.basename(pairb)}")

        try:
            print("\n" + "#" * 80)
            print("A端结果")
            print("#" * 80)
            result_a = summary_single(paira, fs=fs)
            print_result(result_a, fs=fs, end_name="A端")

            print("\n" + "#" * 80)
            print("B端结果")
            print("#" * 80)
            result_b = summary_single(pairb, fs=fs)
            print_result(result_b, fs=fs, end_name="B端")

            # 计算两个最终波头时间差（单位 us），并拼到文件夹名后缀里
            save_dir, dt_us = build_save_dir_with_dt(
                save_root=save_root,
                match_key=match_key,
                result_a=result_a,
                result_b=result_b,
            )
            os.makedirs(save_dir, exist_ok=True)

            print(f"波头时间差 Δt = {dt_us:.2f} us")
            print(f"save_dir: {save_dir}")

            save_paths = plot_result_ab(
                result_a=result_a,
                result_b=result_b,
                file_a=paira,
                file_b=pairb,
                fs=fs,
                title_prefix="RDP-assisted rounded-wavefront detection with improved wavelet denoising + local AIC",
                plot_pad_ratio=0.20,
                save_dir=save_dir,
            )

            print(f"[完成] {match_key}")
            print(f"保存结果: {save_paths}")

        except Exception as e:
            print(f"[失败] {match_key} -> {e}")

# =========================================================
# 9. main：两种运行模式
# =========================================================
if __name__ == "__main__":
    DATA_DIR = r"D:\PSCAD\Intern\dataAnalyse\csv"
    SAVE_ROOT = "fig"

    # "all" -> 绘制文件夹中所有能配对的数据
    # "key" -> 只绘制指定匹配字段
    MODE = "all"

    # 当 MODE = "key" 时生效
    TARGET_KEY = "11-52-46_933"

    if MODE == "all":
        pairs = build_pairs(DATA_DIR, target_key=None)
        run_summary_for_pairs(pairs, fs=FS, save_root=SAVE_ROOT)

    elif MODE == "key":
        pairs = build_pairs(DATA_DIR, target_key=TARGET_KEY)
        run_summary_for_pairs(pairs, fs=FS, save_root=SAVE_ROOT)

    else:
        raise ValueError("MODE 只能是 'all' 或 'key'")