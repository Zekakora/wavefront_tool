import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks
from PyEMD import EMD
import fastrdp

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# 0. 基础配置
# =========================================================
FS = 4.2e6  # 采样频率 4.2 MHz
DEFAULT_CSV_PATH = "2025-12-06_15-21-21_136-675-381.csv"


# =========================================================
# 1. 数据读取
# =========================================================
def load_csv_no_header(csv_path: str):
    """
    读取无列名 CSV:
    第1列: index
    第2列: value
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：第一列索引，第二列信号值")

    idx = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)
    return idx, x, df


# =========================================================
# 2. 基础预处理
# =========================================================
def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mad = np.median(np.abs(x - np.median(x))) + 1e-12
    return float(mad / 0.6745)



def preprocess_signal(x: np.ndarray, detrend_type: str = "constant", normalize: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    x = signal.detrend(x, type=detrend_type)
    if normalize:
        x = x / (np.max(np.abs(x)) + 1e-12)
    return x



def time_vector(n: int, fs: float) -> np.ndarray:
    return np.arange(n) / fs



def smooth_savgol(x: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    """
    轻微平滑，避免孤立毛刺对 RDP 或 TEO 过敏。
    """
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
    return signal.savgol_filter(x, window_length=window_length, polyorder=polyorder, mode="interp")


# =========================================================
# 3. RDP 粗定位
# =========================================================
def _robust_threshold_from_pre(x_pre, method="mad", k=6.0, use_abs=True):
    x_pre = np.asarray(x_pre, dtype=float)
    baseline = np.median(x_pre)
    d = x_pre - baseline
    if use_abs:
        d = np.abs(d)

    if method == "max":
        thr = np.max(d)
    elif method == "std":
        thr = np.mean(d) + k * np.std(d, ddof=1)
    elif method == "mad":
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-12
        sigma = 1.4826 * mad
        thr = med + k * sigma
    else:
        raise ValueError("method must be one of {'mad','std','max'}")

    return float(thr), float(baseline)



def detect_wavefront_by_turningpoints(
    x,
    turn_idx,
    pre_n=500,
    post_check_n=200,
    thr_method="mad",
    k=6.0,
    use_abs=True,
    min_consecutive=1,
):
    """
    从 RDP 转折点中找第一个可信的波头候选。
    """
    x = np.asarray(x, dtype=float)
    turn_idx = np.asarray(turn_idx, dtype=int)
    turn_idx = np.unique(turn_idx)
    turn_idx = turn_idx[(turn_idx >= 0) & (turn_idx < len(x))]
    turn_idx.sort()

    if len(x) < pre_n:
        raise ValueError("x 长度小于 pre_n，无法用前窗估计阈值")

    thr, baseline = _robust_threshold_from_pre(
        x[:pre_n], method=thr_method, k=k, use_abs=use_abs
    )

    def _exceed_mask(seg):
        d = seg - baseline
        if use_abs:
            d = np.abs(d)
        return d > thr

    turn_idx_head = None
    first_cross_idx = None

    for i in turn_idx:
        j = min(len(x), i + int(post_check_n))
        if j <= i + 1:
            continue
        m = _exceed_mask(x[i:j])
        if not np.any(m):
            continue

        if min_consecutive <= 1:
            rel = int(np.argmax(m))
            turn_idx_head = int(i)
            first_cross_idx = int(i + rel)
            break
        else:
            w = np.ones(int(min_consecutive), dtype=int)
            conv = np.convolve(m.astype(int), w, mode="valid")
            hit = np.where(conv >= int(min_consecutive))[0]
            if hit.size > 0:
                rel = int(hit[0])
                turn_idx_head = int(i)
                first_cross_idx = int(i + rel)
                break

    return {
        "turn_idx_head": turn_idx_head,
        "first_cross_idx": first_cross_idx,
        "thr": thr,
        "baseline": baseline,
        "thr_method": thr_method,
        "k": k,
        "pre_n": pre_n,
        "post_check_n": post_check_n,
    }



def rough_onset_by_threshold(x: np.ndarray, fs: float, k: float = 6.0, consecutive: int = 5):
    """
    当 RDP 候选失败时，退化成一个简单但稳健的首达粗定位。
    """
    x = np.asarray(x, dtype=float)
    n0 = min(max(200, len(x) // 5), len(x) - 1)
    base = x[:n0]
    mu = np.median(base)
    sigma = robust_sigma(base)
    thr = k * sigma

    flag = np.abs(x - mu) > thr
    count = 0
    for i, v in enumerate(flag):
        count = count + 1 if v else 0
        if count >= consecutive:
            return int(i - consecutive + 1)

    return int(np.argmax(np.abs(x - mu)))



def rdp_rough_locator(
    x: np.ndarray,
    fs: float = FS,
    preprocess: bool = True,
    sg_window: int = 5,
    sg_polyorder: int = 2,
    rdp_epsilon: float = 0.03,
    pre_n: int = 500,
    post_check_n: int = 200,
    thr_method: str = "mad",
    k: float = 6.0,
    use_abs: bool = True,
    min_consecutive: int = 5,
    fallback_k: float = 6.0,
    fallback_consecutive: int = 5,
):
    """
    用 RDP 找到粗定位中心 rough_idx。
    注意：这里 rough_idx 只用于后续缩小搜索窗口，不用于裁剪 ICEEMDAN 输入。
    """
    x = np.asarray(x, dtype=float)
    if preprocess:
        x_rdp = preprocess_signal(x, detrend_type="constant", normalize=True)
    else:
        x_rdp = x.copy()

    x_rdp_smooth = smooth_savgol(x_rdp, window_length=sg_window, polyorder=sg_polyorder)

    xx = np.arange(len(x_rdp_smooth), dtype=float)
    x_keep, y_keep = fastrdp.rdp(xx, x_rdp_smooth, rdp_epsilon)
    turn_idx = np.asarray(np.round(x_keep), dtype=int)
    turn_idx = np.unique(turn_idx)

    res = detect_wavefront_by_turningpoints(
        x_rdp_smooth,
        turn_idx,
        pre_n=pre_n,
        post_check_n=post_check_n,
        thr_method=thr_method,
        k=k,
        use_abs=use_abs,
        min_consecutive=min_consecutive,
    )

    if res["turn_idx_head"] is None:
        rough_idx = rough_onset_by_threshold(
            x_rdp_smooth,
            fs=fs,
            k=fallback_k,
            consecutive=fallback_consecutive,
        )
        method = "fallback_threshold"
    else:
        rough_idx = int(res["turn_idx_head"])
        if res["first_cross_idx"] is not None:
            # 保守一些：取更靠前的一个，避免搜索窗整体偏后
            rough_idx = int(min(res["turn_idx_head"], res["first_cross_idx"]))
        method = "rdp_turningpoint"

    return {
        "rough_idx": int(rough_idx),
        "rough_t": float(rough_idx / fs),
        "x_rdp": x_rdp,
        "x_rdp_smooth": x_rdp_smooth,
        "rdp_x": np.asarray(x_keep, dtype=float),
        "rdp_y": np.asarray(y_keep, dtype=float),
        "turn_idx": turn_idx,
        "rdp_result": res,
        "rough_method": method,
        "params": {
            "rdp_epsilon": rdp_epsilon,
            "pre_n": pre_n,
            "post_check_n": post_check_n,
            "thr_method": thr_method,
            "k": k,
            "use_abs": use_abs,
            "min_consecutive": min_consecutive,
            "sg_window": sg_window,
            "sg_polyorder": sg_polyorder,
        },
    }


# =========================================================
# 4. ICEEMDAN（工程复现版）
# =========================================================
def _count_extrema(x: np.ndarray) -> int:
    dx = np.diff(x)
    s = np.sign(dx)
    s[s == 0] = 1
    return int(np.sum(s[:-1] * s[1:] < 0))



def _stop_decomposition(r: np.ndarray, std_thr: float = 1e-8) -> bool:
    if np.std(r) < std_thr:
        return True
    if _count_extrema(r) < 2:
        return True
    return False



def _emd_first_imf(x: np.ndarray, emd: EMD) -> np.ndarray:
    imfs = emd.emd(x, max_imf=1)
    if imfs is None or len(imfs) == 0:
        return np.zeros_like(x)

    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1:
        return imfs
    return imfs[0]



def _precompute_noise_imfs(
    n_samples: int,
    ensemble_size: int,
    emd: EMD,
    rng: np.random.Generator,
):
    noise_bank = []

    for _ in range(int(ensemble_size)):
        w = rng.normal(0.0, 1.0, size=n_samples)
        nimfs = emd.emd(w)

        if nimfs is None or len(nimfs) == 0:
            noise_bank.append([])
            continue

        nimfs = np.asarray(nimfs, dtype=float)
        if nimfs.ndim == 1:
            nimfs = nimfs[None, :]

        normed = []
        for row in nimfs:
            s = np.std(row)
            if s < 1e-12:
                normed.append(np.zeros_like(row))
            else:
                normed.append(row / s)
        noise_bank.append(normed)

    return noise_bank



def iceemdan(
    x: np.ndarray,
    ensemble_size: int = 50,
    noise_strength: float = 0.12,
    max_imfs: int | None = 6,
    random_state: int = 42,
):
    """
    工程化 ICEEMDAN 复现。
    返回:
        imfs:    shape (n_imfs, n_samples)
        residue: shape (n_samples,)
    """
    x = np.asarray(x, dtype=float).copy()
    x = x - np.mean(x)

    n = len(x)
    rng = np.random.default_rng(random_state)
    emd = EMD()

    noise_bank = _precompute_noise_imfs(n, ensemble_size, emd, rng)

    residue = x.copy()
    imfs = []
    k = 0

    while True:
        if max_imfs is not None and k >= int(max_imfs):
            break
        if _stop_decomposition(residue):
            break

        sigma_r = np.std(residue) + 1e-12
        local_means = []

        for i in range(int(ensemble_size)):
            if k < len(noise_bank[i]):
                nk = noise_bank[i][k]
            else:
                nk = np.zeros(n)

            x_perturbed = residue + noise_strength * sigma_r * nk
            first_imf = _emd_first_imf(x_perturbed, emd)
            local_mean = x_perturbed - first_imf
            local_means.append(local_mean)

        new_residue = np.mean(local_means, axis=0)
        current_imf = residue - new_residue

        imfs.append(current_imf)
        residue = new_residue
        k += 1

    if len(imfs) == 0:
        return np.empty((0, len(x))), residue

    return np.vstack(imfs), residue


# =========================================================
# 5. 镜像延拓，减弱端点效应
# =========================================================
def mirror_pad_signal(x: np.ndarray, pad_n: int):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy(), 0

    pad_n = int(max(0, pad_n))
    pad_n = min(pad_n, n - 2)
    if pad_n <= 0:
        return x.copy(), 0

    left = x[1:pad_n + 1][::-1]
    right = x[-pad_n - 1:-1][::-1]
    x_pad = np.concatenate([left, x, right])
    return x_pad, pad_n



def unpad_imfs(imfs: np.ndarray, residue: np.ndarray, pad_n: int):
    pad_n = int(max(0, pad_n))
    if pad_n <= 0:
        return imfs, residue

    if imfs.ndim == 2 and imfs.shape[1] > 2 * pad_n:
        imfs = imfs[:, pad_n:-pad_n]
    if residue.ndim == 1 and residue.shape[0] > 2 * pad_n:
        residue = residue[pad_n:-pad_n]
    return imfs, residue


# =========================================================
# 6. TEO 与特征检测
# =========================================================
def teager_energy_operator(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return y



def choose_teo_input(imfs: np.ndarray, mode: str = "iimf1", alpha2: float = 0.3) -> np.ndarray:
    """
    选择送入 TEO 的信号。
    mode:
        - 'iimf1'  : 只用第一 IMF
        - 'iimf12' : IMF1 + alpha2 * IMF2
    """
    if imfs.ndim != 2 or imfs.shape[0] == 0:
        raise ValueError("imfs 为空，无法选择 TEO 输入")

    if mode == "iimf1":
        return imfs[1]

    if mode == "iimf12":
        if imfs.shape[0] >= 2:
            return imfs[0] + alpha2 * imfs[1]
        return imfs[0]

    raise ValueError("mode 只能取 'iimf1' 或 'iimf12'")



def first_consecutive_crossing(feature: np.ndarray, thr: float, consecutive: int = 3):
    count = 0
    for i, v in enumerate(feature):
        if v > thr:
            count += 1
            if count >= consecutive:
                return int(i - consecutive + 1)
        else:
            count = 0
    return None



def detect_head_from_teo(
    x_imf: np.ndarray,
    fs: float,
    search_start_s: float,
    search_end_s: float,
    pre_noise_end_s: float,
    smooth_win: int = 7,
    pick_mode: str = "first_cross",   # "max" / "first_cross" / "first_sig_slope"
    sigma_k: float = 5.5,
    min_peak_distance_samples: int = 10,
    cross_consecutive: int = 3,
    slope_polarity: str = "abs",      # "abs" / "positive" / "negative"
    slope_smooth_win: int = 5,
    edge_ignore_samples: int = 0,
):
    """
    三种检测模式：
    1) max              : 搜索窗内 TEO 能量最大点
    2) first_cross      : 搜索窗内 TEO 能量首次连续越阈
    3) first_sig_slope  : 搜索窗内 IMF 首个显著斜率点

    注意：
    - edge_ignore_samples > 0 时，会忽略 IMF 首末端这部分样本
    - 不改变原始索引，只是不参与阈值估计和波头搜索
    """
    x_imf = np.asarray(x_imf, dtype=float)
    n = len(x_imf)

    edge_ignore_samples = int(max(0, edge_ignore_samples))
    if 2 * edge_ignore_samples >= n:
        raise ValueError("edge_ignore_samples 过大，已经超过信号长度的一半")

    valid_start = edge_ignore_samples
    valid_end = n - edge_ignore_samples

    if smooth_win < 5:
        smooth_win = 5
    if smooth_win % 2 == 0:
        smooth_win += 1
    if smooth_win >= n:
        smooth_win = n - 1 if n % 2 == 0 else n

    if smooth_win >= 5:
        x_use = signal.savgol_filter(x_imf, smooth_win, 2, mode="interp")
    else:
        x_use = x_imf.copy()

    i0 = max(0, int(round(search_start_s * fs)))
    i1 = min(n, int(round(search_end_s * fs)))
    j1 = min(n, int(round(pre_noise_end_s * fs)))

    i0 = max(i0, valid_start)
    i1 = min(i1, valid_end)

    if i1 <= i0:
        raise ValueError("搜索窗口无效：被 edge_ignore_samples 截没了")

    noise_i0 = valid_start
    noise_i1 = min(j1, valid_end)
    if noise_i1 - noise_i0 < 10:
        raise ValueError("噪声窗口太短：请增大 pre_noise_end_s 或减小 edge_ignore_samples")
    if noise_i1 > i0:
        raise ValueError("噪声窗口与搜索窗口重叠：请确保 pre_noise_end_s < search_start_s")

    if pick_mode in ["max", "first_cross"]:
        e = teager_energy_operator(x_use)
        e = np.maximum(e, 0.0)

        if len(e) >= 7:
            feature = signal.savgol_filter(e, 7, 2, mode="interp")
        else:
            feature = e

        noise_seg = feature[noise_i0:noise_i1]
        sigma = robust_sigma(noise_seg)
        thr = sigma_k * sigma

        seg = feature[i0:i1]

        if pick_mode == "max":
            idx_local = int(np.argmax(seg))
            peak_found = True
        else:
            idx_local = first_consecutive_crossing(seg, thr, consecutive=cross_consecutive)
            if idx_local is None:
                idx_local = int(np.argmax(seg))
                peak_found = False
            else:
                peak_found = True

        idx_head = i0 + idx_local
        t_head = idx_head / fs

        feature_plot = feature.copy()
        if edge_ignore_samples > 0:
            feature_plot[:valid_start] = np.nan
            feature_plot[valid_end:] = np.nan

        return {
            "idx_head": int(idx_head),
            "t_head": float(t_head),
            "feature": feature_plot,
            "threshold": float(thr),
            "feature_name": "TEO energy",
            "peak_found": bool(peak_found),
            "x_imf_smooth": x_use,
            "search_i0": int(i0),
            "search_i1": int(i1),
            "noise_i0": int(noise_i0),
            "noise_i1": int(noise_i1),
            "valid_start": int(valid_start),
            "valid_end": int(valid_end),
            "edge_ignore_samples": int(edge_ignore_samples),
        }

    elif pick_mode == "first_sig_slope":
        dx = np.diff(x_use, prepend=x_use[0]) * fs

        if slope_polarity == "positive":
            slope_feature = np.maximum(dx, 0.0)
        elif slope_polarity == "negative":
            slope_feature = np.maximum(-dx, 0.0)
        else:
            slope_feature = np.abs(dx)

        if slope_smooth_win < 5:
            slope_smooth_win = 5
        if slope_smooth_win % 2 == 0:
            slope_smooth_win += 1
        if slope_smooth_win >= len(slope_feature):
            slope_smooth_win = len(slope_feature) - 1 if len(slope_feature) % 2 == 0 else len(slope_feature)

        if slope_smooth_win >= 5:
            feature = signal.savgol_filter(slope_feature, slope_smooth_win, 2, mode="interp")
        else:
            feature = slope_feature

        noise_seg = feature[noise_i0:noise_i1]
        sigma = robust_sigma(noise_seg)
        thr = sigma_k * sigma

        seg = feature[i0:i1]
        peaks, props = find_peaks(seg, height=thr, distance=min_peak_distance_samples)

        if len(peaks) == 0:
            idx_local = int(np.argmax(seg))
            peak_found = False
        else:
            idx_local = int(peaks[0])
            peak_found = True

        idx_head = i0 + idx_local
        t_head = idx_head / fs

        feature_plot = feature.copy()
        if edge_ignore_samples > 0:
            feature_plot[:valid_start] = np.nan
            feature_plot[valid_end:] = np.nan

        return {
            "idx_head": int(idx_head),
            "t_head": float(t_head),
            "feature": feature_plot,
            "threshold": float(thr),
            "feature_name": "Slope feature",
            "peak_found": bool(peak_found),
            "x_imf_smooth": x_use,
            "search_i0": int(i0),
            "search_i1": int(i1),
            "noise_i0": int(noise_i0),
            "noise_i1": int(noise_i1),
            "valid_start": int(valid_start),
            "valid_end": int(valid_end),
            "edge_ignore_samples": int(edge_ignore_samples),
        }

    else:
        raise ValueError("pick_mode 只能取 'max'、'first_cross' 或 'first_sig_slope'")


# =========================================================
# 7. 全局 ICEEMDAN + 局部搜索窗
# =========================================================
def detect_wavefront_global_iceemdan_teo(
    x: np.ndarray,
    fs: float = FS,
    preprocess: bool = True,
    use_imf_mode: str = "iimf12",
    alpha2: float = 0.35,
    ensemble_size: int = 50,
    noise_strength: float = 0.10,
    max_imfs: int = 6,
    random_state: int = 42,
    use_mirror_pad: bool = True,
    mirror_pad_us: float = 30.0,
    search_start_s: float = 108e-6,
    search_end_s: float = 125e-6,
    pre_noise_end_s: float = 100e-6,
    pre_sg_window: int = 5,
    sigma_k: float = 5.0,
    min_peak_distance_samples: int = 10,
    cross_consecutive: int = 3,
    pick_mode: str = "first_cross",
    slope_polarity: str = "abs",
    slope_smooth_win: int = 5,
    edge_ignore_samples: int = 30,
):
    x = np.asarray(x, dtype=float)

    if preprocess:
        x_proc = preprocess_signal(x)
    else:
        x_proc = x.copy()

    if use_mirror_pad:
        pad_n = int(round(mirror_pad_us * 1e-6 * fs))
        x_for_ice, pad_n = mirror_pad_signal(x_proc, pad_n)
    else:
        x_for_ice = x_proc.copy()
        pad_n = 0

    imfs_pad, residue_pad = iceemdan(
        x_for_ice,
        ensemble_size=ensemble_size,
        noise_strength=noise_strength,
        max_imfs=max_imfs,
        random_state=random_state,
    )

    if imfs_pad.shape[0] == 0:
        raise RuntimeError("ICEEMDAN 未分解出有效 IMF")

    imfs, residue = unpad_imfs(imfs_pad, residue_pad, pad_n)

    if imfs.shape[1] != len(x_proc):
        raise RuntimeError("去延拓后的 IMF 长度与原始信号不一致，请检查镜像延拓流程")

    teo_input = choose_teo_input(imfs, mode=use_imf_mode, alpha2=alpha2)

    teo_res = detect_head_from_teo(
        teo_input,
        fs=fs,
        search_start_s=search_start_s,
        search_end_s=search_end_s,
        pre_noise_end_s=pre_noise_end_s,
        smooth_win=pre_sg_window,
        pick_mode=pick_mode,
        sigma_k=sigma_k,
        min_peak_distance_samples=min_peak_distance_samples,
        cross_consecutive=cross_consecutive,
        slope_polarity=slope_polarity,
        slope_smooth_win=slope_smooth_win,
        edge_ignore_samples=edge_ignore_samples,
    )

    return {
        "x_raw": x,
        "x_proc": x_proc,
        "imfs": imfs,
        "residue": residue,
        "teo_input": teo_input,
        "x_smooth": teo_res["x_imf_smooth"],
        "idx_head": teo_res["idx_head"],
        "t_head": teo_res["t_head"],
        "feature": teo_res["feature"],
        "feature_name": teo_res["feature_name"],
        "threshold": teo_res["threshold"],
        "peak_found": teo_res["peak_found"],
        "search_i0": teo_res["search_i0"],
        "search_i1": teo_res["search_i1"],
        "noise_i0": teo_res["noise_i0"],
        "noise_i1": teo_res["noise_i1"],
        "valid_start": teo_res["valid_start"],
        "valid_end": teo_res["valid_end"],
        "params": {
            "fs": fs,
            "use_imf_mode": use_imf_mode,
            "alpha2": alpha2,
            "ensemble_size": ensemble_size,
            "noise_strength": noise_strength,
            "max_imfs": max_imfs,
            "random_state": random_state,
            "use_mirror_pad": use_mirror_pad,
            "mirror_pad_us": mirror_pad_us,
            "search_start_s": search_start_s,
            "search_end_s": search_end_s,
            "pre_noise_end_s": pre_noise_end_s,
            "pre_sg_window": pre_sg_window,
            "sigma_k": sigma_k,
            "min_peak_distance_samples": min_peak_distance_samples,
            "cross_consecutive": cross_consecutive,
            "pick_mode": pick_mode,
            "slope_polarity": slope_polarity,
            "slope_smooth_win": slope_smooth_win,
            "edge_ignore_samples": edge_ignore_samples,
        },
    }


# =========================================================
# 8. 总流程：RDP 粗定位 -> 全局 ICEEMDAN -> 局部搜索窗精定位
# =========================================================
def detect_wavefront_rdp_global_iceemdan_teo(
    x: np.ndarray,
    fs: float = FS,
    # RDP 参数
    rdp_preprocess: bool = True,
    rdp_sg_window: int = 5,
    rdp_sg_polyorder: int = 2,
    rdp_epsilon: float = 0.03,
    rdp_pre_n: int = 500,
    rdp_post_check_n: int = 200,
    rdp_thr_method: str = "mad",
    rdp_k: float = 6.0,
    rdp_use_abs: bool = True,
    rdp_min_consecutive: int = 5,
    # 基于 RDP 的搜索窗参数（单位 us）
    search_left_us: float = 10.0,
    search_right_us: float = 15.0,
    noise_guard_us: float = 2.0,
    min_pre_noise_us: float = 20.0,
    # 全局 ICEEMDAN-TEO 参数
    global_preprocess: bool = True,
    use_imf_mode: str = "iimf12",
    alpha2: float = 0.35,
    ensemble_size: int = 50,
    noise_strength: float = 0.10,
    max_imfs: int = 6,
    random_state: int = 42,
    use_mirror_pad: bool = True,
    mirror_pad_us: float = 30.0,
    pre_sg_window: int = 5,
    sigma_k: float = 5.0,
    min_peak_distance_samples: int = 10,
    cross_consecutive: int = 3,
    pick_mode: str = "first_cross",
    slope_polarity: str = "abs",
    slope_smooth_win: int = 5,
    edge_ignore_samples: int = 30,
):
    x = np.asarray(x, dtype=float)
    n = len(x)
    total_t = n / fs

    rdp_info = rdp_rough_locator(
        x,
        fs=fs,
        preprocess=rdp_preprocess,
        sg_window=rdp_sg_window,
        sg_polyorder=rdp_sg_polyorder,
        rdp_epsilon=rdp_epsilon,
        pre_n=rdp_pre_n,
        post_check_n=rdp_post_check_n,
        thr_method=rdp_thr_method,
        k=rdp_k,
        use_abs=rdp_use_abs,
        min_consecutive=rdp_min_consecutive,
        fallback_k=rdp_k,
        fallback_consecutive=rdp_min_consecutive,
    )

    rough_idx = int(rdp_info["rough_idx"])
    rough_t = rough_idx / fs

    search_start_s = max(0.0, rough_t - search_left_us * 1e-6)
    search_end_s = min(total_t, rough_t + search_right_us * 1e-6)

    pre_noise_end_s = search_start_s - noise_guard_us * 1e-6
    min_pre_noise_end_s = min_pre_noise_us * 1e-6
    pre_noise_end_s = max(pre_noise_end_s, min_pre_noise_end_s)
    pre_noise_end_s = min(pre_noise_end_s, search_start_s - 1.0 / fs)

    if pre_noise_end_s <= 0:
        pre_noise_end_s = min(search_start_s * 0.8, search_start_s - 1.0 / fs)

    if pre_noise_end_s <= 0:
        raise ValueError("无法构造有效的噪声窗口：粗定位过于靠前，请增大 min_pre_noise_us 或检查数据")

    global_info = detect_wavefront_global_iceemdan_teo(
        x=x,
        fs=fs,
        preprocess=global_preprocess,
        use_imf_mode=use_imf_mode,
        alpha2=alpha2,
        ensemble_size=ensemble_size,
        noise_strength=noise_strength,
        max_imfs=max_imfs,
        random_state=random_state,
        use_mirror_pad=use_mirror_pad,
        mirror_pad_us=mirror_pad_us,
        search_start_s=search_start_s,
        search_end_s=search_end_s,
        pre_noise_end_s=pre_noise_end_s,
        pre_sg_window=pre_sg_window,
        sigma_k=sigma_k,
        min_peak_distance_samples=min_peak_distance_samples,
        cross_consecutive=cross_consecutive,
        pick_mode=pick_mode,
        slope_polarity=slope_polarity,
        slope_smooth_win=slope_smooth_win,
        edge_ignore_samples=edge_ignore_samples,
    )

    global_info["rough_idx"] = rough_idx
    global_info["rough_t"] = rough_t
    global_info["rdp_info"] = rdp_info
    global_info["search_start_s"] = search_start_s
    global_info["search_end_s"] = search_end_s
    global_info["pre_noise_end_s"] = pre_noise_end_s
    return global_info


# =========================================================
# 9. 结果输出与绘图（A/B 双端，2行4列）
# =========================================================
def print_result(result: dict, fs: float = FS, end_name: str = ""):
    prefix = f"[{end_name}] " if end_name else ""
    print("=" * 72)
    print(f"{prefix}检测结果")
    print("=" * 72)
    print(f"{prefix}RDP 粗定位样点 rough_idx = {result['rough_idx']}")
    print(f"{prefix}RDP 粗定位时刻 rough_t = {result['rough_t'] * 1e6:.6f} us")
    print("-" * 72)
    print(f"{prefix}最终波头样点 idx_head = {result['idx_head']}")
    print(f"{prefix}最终波头时刻 t_head = {result['t_head'] * 1e6:.6f} us")
    print(f"{prefix}是否找到超过阈值的显著点 = {result['peak_found']}")
    print(f"{prefix}特征类型 = {result['feature_name']}")
    print(f"{prefix}阈值 = {result['threshold']:.6e}")
    print("-" * 72)
    print(f"{prefix}搜索窗 = [{result['search_start_s'] * 1e6:.3f}, {result['search_end_s'] * 1e6:.3f}] us")
    print(f"{prefix}噪声窗终点 = {result['pre_noise_end_s'] * 1e6:.3f} us")
    print("=" * 72)


def _plot_one_result_row(ax_row, result: dict, end_label: str, fs: float = FS, title_prefix: str = "RDP + Global ICEEMDAN-TEO"):
    """
    在一整行 4 个子图上画单端结果
    ax_row: 长度为4的 axes，例如 axes[0, :]
    """
    x_raw = result["x_raw"]
    x_proc = result["x_proc"]
    imfs = result["imfs"]
    teo_input = result["teo_input"]
    feature = result["feature"]
    threshold = result["threshold"]
    idx_head = result["idx_head"]
    rough_idx = result["rough_idx"]
    rdp_info = result["rdp_info"]

    t = time_vector(len(x_raw), fs) * 1e6
    t_head_us = idx_head / fs * 1e6
    rough_t_us = rough_idx / fs * 1e6
    search_start_us = result["search_start_s"] * 1e6
    search_end_us = result["search_end_s"] * 1e6
    pre_noise_end_us = result["pre_noise_end_s"] * 1e6

    ax1, ax2, ax3, ax4 = ax_row

    # ---- 1. 原始信号 + RDP ----
    ax1.plot(t, x_raw, linewidth=1.0, label="Raw signal")
    ax1.scatter(rdp_info["rdp_x"] / fs * 1e6, rdp_info["rdp_y"], s=10, label="RDP points")
    ax1.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax1.axvline(t_head_us, linestyle="--", color="orange", label="Final head")
    ax1.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax1.axvspan(0.0, pre_noise_end_us, alpha=0.08, label="Noise window")
    ax1.set_title(f"{end_label} - Raw & RDP")
    ax1.set_xlabel("Time (us)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="best", fontsize=8)

    # ---- 2. 预处理信号 / IMF ----
    ax2.plot(t, x_proc, linewidth=1.0, label="Preprocessed signal")
    if imfs.shape[0] > 0:
        ax2.plot(t, imfs[0], linewidth=1.0, label="IMF1")
    if imfs.shape[0] > 1 and result["params"]["use_imf_mode"] == "iimf12":
        ax2.plot(t, imfs[1], linewidth=1.0, alpha=0.7, label="IMF2")
    ax2.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax2.axvline(t_head_us, linestyle="--", color="orange", label="Final head")
    ax2.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax2.set_title(f"{end_label} - Preprocessed / IMF")
    ax2.set_xlabel("Time (us)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="best", fontsize=8)

    # ---- 3. TEO 输入 ----
    ax3.plot(t, teo_input, linewidth=1.0, label=result["params"]["use_imf_mode"])
    ax3.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax3.axvline(t_head_us, linestyle="--", color="orange", label="Final head")
    ax3.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax3.set_title(f"{end_label} - TEO input")
    ax3.set_xlabel("Time (us)")
    ax3.set_ylabel("Amplitude")
    ax3.legend(loc="best", fontsize=8)

    # ---- 4. 特征 + 阈值检测 ----
    ax4.plot(t, feature, linewidth=1.0, label=result.get("feature_name", "Feature"))
    ax4.axhline(threshold, linestyle="--", label="Threshold")
    ax4.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax4.axvline(t_head_us, linestyle="--", color="orange", label="Final head")
    ax4.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window", color="green")
    ax4.axvspan(0.0, pre_noise_end_us, alpha=0.08, label="Noise window")
    ax4.set_title(f"{end_label} - Feature")
    ax4.set_xlabel("Time (us)")
    ax4.set_ylabel("Feature")
    ax4.legend(loc="best", fontsize=8)


def plot_result_ab(
    result_a: dict,
    result_b: dict,
    file_a: str,
    file_b: str,
    fs: float = FS,
    title_prefix: str = "RDP + Global ICEEMDAN-TEO"
):
    """
    A/B 双端绘图：
    第1行 A端 4图
    第2行 B端 4图
    共 2x4 子图
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), dpi=200)
    plt.subplots_adjust(wspace=0.28, hspace=0.40)

    _plot_one_result_row(axes[0, :], result_a, end_label="A端", fs=fs, title_prefix=title_prefix)
    _plot_one_result_row(axes[1, :], result_b, end_label="B端", fs=fs, title_prefix=title_prefix)

    t_head_a_us = result_a["t_head"] * 1e6
    rough_a_us = result_a["rough_t"] * 1e6
    t_head_b_us = result_b["t_head"] * 1e6
    rough_b_us = result_b["rough_t"] * 1e6

    base_a = os.path.splitext(os.path.basename(file_a))[0]
    base_b = os.path.splitext(os.path.basename(file_b))[0]

    fig.suptitle(
        f"{title_prefix}\n"
        f"A端: final={t_head_a_us:.3f} us, rough={rough_a_us:.3f} us    |    "
        f"B端: final={t_head_b_us:.3f} us, rough={rough_b_us:.3f} us",
        fontweight="bold",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs("fig", exist_ok=True)
    save_name = f"{base_a}__{base_b}_AB_rdp_global_iceemdan_teo.jpg"
    save_path = os.path.join("fig", save_name)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"图像已保存到: {save_path}")

# =========================================================
# 10. 示例主流程（A/B 双端）
# =========================================================
def summary_single(file_name: str):
    idx, x, df = load_csv_no_header(file_name)

    result = detect_wavefront_rdp_global_iceemdan_teo(
        x=x,
        fs=FS,
        # ---------- RDP 粗定位 ----------
        rdp_preprocess=True,
        rdp_sg_window=5,
        rdp_sg_polyorder=2,
        rdp_epsilon=0.03,
        rdp_pre_n=500,
        rdp_post_check_n=200,
        rdp_thr_method="mad",
        rdp_k=6.0,
        rdp_use_abs=True,
        rdp_min_consecutive=5,
        # ---------- 基于 RDP 的局部搜索窗 ----------
        search_left_us=10.0,
        search_right_us=15.0,
        noise_guard_us=2.0,
        min_pre_noise_us=20.0,
        # ---------- 全局 ICEEMDAN-TEO ----------
        global_preprocess=True,
        use_imf_mode="iimf1",
        alpha2=0.35,
        ensemble_size=50,
        noise_strength=0.10,
        max_imfs=6,
        random_state=42,
        use_mirror_pad=True,
        mirror_pad_us=30.0,
        pre_sg_window=5,
        sigma_k=5.0,
        min_peak_distance_samples=10,
        cross_consecutive=3,
        pick_mode="first_cross",
        slope_polarity="abs",
        slope_smooth_win=5,
        edge_ignore_samples=30,
    )

    print_result(result, fs=FS)
    return result


def summary_ab(file_a: str, file_b: str):
    print("\n" + "#" * 80)
    print("A端结果")
    print("#" * 80)
    result_a = summary_single(file_a)
    print_result(result_a, fs=FS, end_name="A端")

    print("\n" + "#" * 80)
    print("B端结果")
    print("#" * 80)
    result_b = summary_single(file_b)
    print_result(result_b, fs=FS, end_name="B端")

    plot_result_ab(
        result_a=result_a,
        result_b=result_b,
        file_a=file_a,
        file_b=file_b,
        fs=FS,
        title_prefix="RDP + Global ICEEMDAN-TEO"
    )

    return result_a, result_b


if __name__ == "__main__":
    file_a = "2026-03-13_12-07-22_440-396-971.csv"
    file_b = "2026-03-13_12-07-22_440-406-255.csv"

    summary_ab(file_a, file_b)