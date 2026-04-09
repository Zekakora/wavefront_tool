"""Plotting and saving helpers for modular wavefront detection.

Designed for PyQt integration:
- create_result_figure_* returns matplotlib Figure/Axes for embedding
- save_figure writes images to disk
- save_result_summary_json exports a lightweight summary without dumping huge arrays
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei', 'SimHei', 'DejaVu Sans']
FS = 4.2e6


def time_vector(n: int, fs: float = FS) -> np.ndarray:
    return np.arange(int(n), dtype=float) / float(fs)


def _safe_set_ylim(ax, x, y, x_left, x_right, pad_ratio: float = 0.08) -> None:
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


def _result_fs(result: dict[str, Any], fs: float | None) -> float:
    if fs is not None:
        return float(fs)
    return float(result.get("params", {}).get("fs", FS))


def _plot_rdp_row_global(ax_row, result: dict[str, Any], *, end_label: str, fs: float) -> None:
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
    ax1.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
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
    ax2.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
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
    ax3.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax3.axvspan(noise_start_us, noise_end_us, alpha=0.08, label="Noise window")
    ax3.set_title(f"{end_label} - Slope feature")
    ax3.set_xlabel("Time (us)")
    ax3.set_ylabel("Feature")
    ax3.legend(loc="best", fontsize=8)

    aic_i0 = result["aic_i0"]
    aic_i1 = result["aic_i1"]
    z0 = max(0, aic_i0 - 60)
    z1 = min(len(x_raw), aic_i1 + 90)
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


def _plot_rdp_row_local(ax_row, result: dict[str, Any], *, end_label: str, fs: float, x_left: float,
                        x_right: float) -> None:
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

    ax1.plot(t, x_raw, linewidth=0.9, alpha=0.40, label="Raw signal")
    ax1.plot(t, x_wavelet, linewidth=1.0, alpha=0.95, label="Wavelet denoised")
    ax1.plot(t, x_smooth, linewidth=1.0, alpha=0.95, label="SG smoothed")
    turn_idx = np.asarray(rdp_info["turn_idx"])
    turn_idx = turn_idx[(turn_idx >= 0) & (turn_idx < len(x_smooth))]
    if len(turn_idx) > 0:
        turn_mask = (t[turn_idx] >= x_left) & (t[turn_idx] <= x_right)
        ax1.scatter(t[turn_idx][turn_mask], x_smooth[turn_idx][turn_mask], s=10, label="RDP points")
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
    ax4.axvspan(result["aic_i0"] / fs * 1e6, result["aic_i1"] / fs * 1e6, alpha=0.12, color="tab:cyan",
                label="AIC window")
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
    y4 = np.concatenate([x_smooth[mask4], np.array([baseline])])
    if y4.size > 0:
        y4_min = np.nanmin(y4)
        y4_max = np.nanmax(y4)
        pad4 = max((y4_max - y4_min) * 0.08, 1e-9)
        ax4.set_ylim(y4_min - pad4, y4_max + pad4)
    ax4.legend(loc="best", fontsize=8)


def _plot_ice_row_global(ax_row, result: dict[str, Any], *, end_label: str, fs: float) -> None:
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
    head_t_us = idx_head / fs * 1e6
    rough_t_us = rough_idx / fs * 1e6
    search_start_us = result["search_start_s"] * 1e6
    search_end_us = result["search_end_s"] * 1e6
    pre_noise_end_us = result["pre_noise_end_s"] * 1e6

    ax1, ax2, ax3, ax4 = ax_row
    ax1.plot(t, x_raw, linewidth=1.0, label="Raw signal")
    ax1.scatter(rdp_info["rdp_x"] / fs * 1e6, rdp_info["rdp_y"], s=10, label="RDP points")
    ax1.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax1.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax1.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax1.axvspan(0.0, pre_noise_end_us, alpha=0.08, label="Noise window")
    ax1.set_title(f"{end_label} - Raw & RDP")
    ax1.set_xlabel("Time (us)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(t, x_proc, linewidth=1.0, label="Preprocessed signal")
    if imfs.shape[0] > 0:
        ax2.plot(t, imfs[0], linewidth=1.0, label="IMF1")
    if imfs.shape[0] > 1 and result["params"]["use_imf_mode"] == "iimf12":
        ax2.plot(t, imfs[1], linewidth=1.0, alpha=0.7, label="IMF2")
    ax2.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax2.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax2.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax2.set_title(f"{end_label} - Preprocessed / IMF")
    ax2.set_xlabel("Time (us)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="best", fontsize=8)

    ax3.plot(t, teo_input, linewidth=1.0, label=result["params"]["use_imf_mode"])
    ax3.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax3.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax3.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax3.set_title(f"{end_label} - TEO input")
    ax3.set_xlabel("Time (us)")
    ax3.set_ylabel("Amplitude")
    ax3.legend(loc="best", fontsize=8)

    ax4.plot(t, feature, linewidth=1.0, label=result.get("feature_name", "Feature"))
    ax4.axhline(threshold, linestyle="--", label="Threshold")
    ax4.axvline(rough_t_us, linestyle="--", label="Rough head")
    ax4.axvline(head_t_us, linestyle="--", color="orange", label="Final head")
    ax4.axvspan(search_start_us, search_end_us, alpha=0.15, label="Search window")
    ax4.axvspan(0.0, pre_noise_end_us, alpha=0.08, label="Noise window")
    ax4.set_title(f"{end_label} - Feature")
    ax4.set_xlabel("Time (us)")
    ax4.set_ylabel("Feature")
    ax4.legend(loc="best", fontsize=8)


def _plot_ice_row_local(ax_row, result: dict[str, Any], *, end_label: str, fs: float, x_left: float,
                        x_right: float) -> None:
    _plot_ice_row_global(ax_row, result, end_label=end_label, fs=fs)
    for ax in ax_row:
        ax.set_xlim(x_left, x_right)


def create_result_figure_single(
        result: dict[str, Any],
        *,
        end_label: str = "Signal",
        fs: float | None = None,
        local_zoom: bool = False,
        x_left: float | None = None,
        x_right: float | None = None,
        title_prefix: str | None = None,
        figsize: tuple[float, float] = (24, 5),
        dpi: int = 200,
):
    fs = _result_fs(result, fs)
    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=0.28, hspace=0.35)

    algo = result.get("algorithm")
    if algo == "rdp_local_aic":
        t = time_vector(len(result["x_raw"]), fs) * 1e6
        if local_zoom:
            if x_left is None or x_right is None:
                start = result["search_i0"] / fs * 1e6
                end = result["search_i1"] / fs * 1e6
                width = max(end - start, 1e-9)
                pad = width * 0.20
                x_left = max(0.0, start - pad)
                x_right = min(t[-1], end + pad)
            _plot_rdp_row_local(axes, result, end_label=end_label, fs=fs, x_left=x_left, x_right=x_right)
        else:
            _plot_rdp_row_global(axes, result, end_label=end_label, fs=fs)
    elif algo == "rdp_global_iceemdan_teo":
        t = time_vector(len(result["x_raw"]), fs) * 1e6
        if local_zoom:
            if x_left is None or x_right is None:
                start = result["search_start_s"] * 1e6
                end = result["search_end_s"] * 1e6
                width = max(end - start, 1e-9)
                pad = width * 0.20
                x_left = max(0.0, start - pad)
                x_right = min(t[-1], end + pad)
            _plot_ice_row_local(axes, result, end_label=end_label, fs=fs, x_left=x_left, x_right=x_right)
        else:
            _plot_ice_row_global(axes, result, end_label=end_label, fs=fs)
    else:
        raise ValueError(f"不支持的算法结果类型: {algo}")

    title_prefix = title_prefix or result.get("algorithm_label", "Wavefront detection")
    fig.suptitle(title_prefix, fontweight="bold", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, axes


def create_result_figure_ab(
        result_a: dict[str, Any],
        result_b: dict[str, Any],
        *,
        file_a: str | None = None,
        file_b: str | None = None,
        fs: float | None = None,
        local_zoom: bool = False,
        plot_pad_ratio: float = 0.20,
        title_prefix: str | None = None,
        figsize: tuple[float, float] = (24, 10),
        dpi: int = 200,
):
    if result_a.get("algorithm") != result_b.get("algorithm"):
        raise ValueError("A/B 结果的算法类型不一致，无法共用同一套子图模板")

    fs = _result_fs(result_a, fs)
    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi, sharex=local_zoom)
    plt.subplots_adjust(wspace=0.28, hspace=0.40)

    algo = result_a.get("algorithm")
    if algo == "rdp_local_aic":
        if local_zoom:
            t_a = time_vector(len(result_a["x_raw"]), fs) * 1e6
            t_b = time_vector(len(result_b["x_raw"]), fs) * 1e6
            a_start, a_end = result_a["search_i0"] / fs * 1e6, result_a["search_i1"] / fs * 1e6
            b_start, b_end = result_b["search_i0"] / fs * 1e6, result_b["search_i1"] / fs * 1e6
            common_start = min(a_start, b_start)
            common_end = max(a_end, b_end)
            pad_us = max(common_end - common_start, 1e-9) * plot_pad_ratio
            global_left = max(0.0, common_start - pad_us)
            global_right = min(max(t_a[-1], t_b[-1]), common_end + pad_us)
            _plot_rdp_row_local(axes[0, :], result_a, end_label="A端", fs=fs, x_left=global_left, x_right=global_right)
            _plot_rdp_row_local(axes[1, :], result_b, end_label="B端", fs=fs, x_left=global_left, x_right=global_right)
        else:
            _plot_rdp_row_global(axes[0, :], result_a, end_label="A端", fs=fs)
            _plot_rdp_row_global(axes[1, :], result_b, end_label="B端", fs=fs)

        t_head_a_us = result_a["t_head"] * 1e6
        coarse_a_us = result_a["coarse_t"] * 1e6
        t_head_b_us = result_b["t_head"] * 1e6
        coarse_b_us = result_b["coarse_t"] * 1e6
        mode_label = "Local zoom" if local_zoom else "Global view"
        title_prefix = title_prefix or result_a.get("algorithm_label", "RDP-assisted wavefront detection")
        fig.suptitle(
            f"{title_prefix} - {mode_label}\n"
            f"A端: final={t_head_a_us:.3f} us, coarse={coarse_a_us:.3f} us    |    "
            f"B端: final={t_head_b_us:.3f} us, coarse={coarse_b_us:.3f} us",
            fontweight="bold",
            fontsize=14,
        )
    elif algo == "rdp_global_iceemdan_teo":
        if local_zoom:
            t_a = time_vector(len(result_a["x_raw"]), fs) * 1e6
            t_b = time_vector(len(result_b["x_raw"]), fs) * 1e6
            a_start, a_end = result_a["search_start_s"] * 1e6, result_a["search_end_s"] * 1e6
            b_start, b_end = result_b["search_start_s"] * 1e6, result_b["search_end_s"] * 1e6
            common_start = min(a_start, b_start)
            common_end = max(a_end, b_end)
            pad_us = max(common_end - common_start, 1e-9) * plot_pad_ratio
            global_left = max(0.0, common_start - pad_us)
            global_right = min(max(t_a[-1], t_b[-1]), common_end + pad_us)
            _plot_ice_row_local(axes[0, :], result_a, end_label="A端", fs=fs, x_left=global_left, x_right=global_right)
            _plot_ice_row_local(axes[1, :], result_b, end_label="B端", fs=fs, x_left=global_left, x_right=global_right)
        else:
            _plot_ice_row_global(axes[0, :], result_a, end_label="A端", fs=fs)
            _plot_ice_row_global(axes[1, :], result_b, end_label="B端", fs=fs)

        t_head_a_us = result_a["t_head"] * 1e6
        rough_a_us = result_a["rough_t"] * 1e6
        t_head_b_us = result_b["t_head"] * 1e6
        rough_b_us = result_b["rough_t"] * 1e6
        mode_label = "Local zoom" if local_zoom else "Global view"
        title_prefix = title_prefix or result_a.get("algorithm_label", "RDP + global ICEEMDAN-TEO")
        fig.suptitle(
            f"{title_prefix} - {mode_label}\n"
            f"A端: final={t_head_a_us:.3f} us, rough={rough_a_us:.3f} us    |    "
            f"B端: final={t_head_b_us:.3f} us, rough={rough_b_us:.3f} us",
            fontweight="bold",
            fontsize=14,
        )
    else:
        raise ValueError(f"不支持的算法结果类型: {algo}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, axes


def save_figure(fig, save_path: str, *, dpi: int = 200, close_fig: bool = False) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return save_path


def _summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        info: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            info.update(
                {
                    "min": float(np.nanmin(value)),
                    "max": float(np.nanmax(value)),
                    "mean": float(np.nanmean(value)),
                }
            )
        return info
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in value]
    return value


def build_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _summarize_value(v) for k, v in result.items()}


def save_result_summary_json(result: dict[str, Any], save_path: str, *, ensure_ascii: bool = False,
                             indent: int = 2) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    summary = build_result_summary(result)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=ensure_ascii, indent=indent)
    return save_path


def print_result_summary(result: dict[str, Any], *, end_name: str = "") -> None:
    prefix = f"[{end_name}] " if end_name else ""
    algo = result.get("algorithm")
    print("=" * 78)
    print(f"{prefix}{result.get('algorithm_label', algo)}")
    print("=" * 78)

    if algo == "rdp_local_aic":
        print(f"{prefix}极性 = {result['polarity']}")
        print(f"{prefix}threshold 粗定位 idx = {result['threshold_rough_idx']}")
        print(f"{prefix}RDP 候选点 idx = {result['rdp_candidate_idx']}")
        print(f"{prefix}融合后 coarse_idx = {result['coarse_idx']}")
        print(f"{prefix}触发点 idx_trigger = {result['idx_trigger']}")
        print(f"{prefix}最终波头 idx_head = {result['idx_head']}")
        print(f"{prefix}最终波头 t_head = {result['t_head'] * 1e6:.6f} us")
        print(
            f"{prefix}搜索窗 = [{result['search_i0'] / result['params']['fs'] * 1e6:.3f}, {result['search_i1'] / result['params']['fs'] * 1e6:.3f}] us")
        metrics = result.get("metrics", {})
        if metrics:
            print(f"{prefix}综合置信度 = {metrics['confidence']:.1f}/100")
            print(f"{prefix}判定说明 = {result.get('summary_text', '')}")
    elif algo == "rdp_global_iceemdan_teo":
        print(f"{prefix}RDP 粗定位 idx = {result['rough_idx']}")
        print(f"{prefix}RDP 粗定位 rough_t = {result['rough_t'] * 1e6:.6f} us")
        print(f"{prefix}最终波头 idx_head = {result['idx_head']}")
        print(f"{prefix}最终波头 t_head = {result['t_head'] * 1e6:.6f} us")
        print(f"{prefix}特征类型 = {result['feature_name']}")
        print(f"{prefix}阈值 = {result['threshold']:.6e}")
        print(f"{prefix}搜索窗 = [{result['search_start_s'] * 1e6:.3f}, {result['search_end_s'] * 1e6:.3f}] us")
        print(f"{prefix}噪声窗终点 = {result['pre_noise_end_s'] * 1e6:.3f} us")
        print(f"{prefix}说明 = {result.get('summary_text', '')}")
    else:
        print(f"{prefix}未知结果类型，无法格式化打印")

    print("=" * 78)


__all__ = [
    "FS",
    "build_result_summary",
    "create_result_figure_ab",
    "create_result_figure_single",
    "print_result_summary",
    "save_figure",
    "save_result_summary_json",
]

# ============================
# PyQtGraph interactive viewer
# ============================
from pathlib import Path

try:
    import pyqtgraph as pg  # type: ignore
    import pyqtgraph.exporters as pg_exporters  # type: ignore

    _PG_IMPORT_ERROR: Exception | None = None
except Exception as _exc_pg:  # pragma: no cover - import-time fallback
    pg = None  # type: ignore
    pg_exporters = None  # type: ignore
    _PG_IMPORT_ERROR = _exc_pg

try:  # pragma: no cover - import-time fallback
    from PyQt6 import QtCore, QtWidgets

    QT_PEN_SOLID = QtCore.Qt.PenStyle.SolidLine
    QT_RIGHT_BUTTON = QtCore.Qt.MouseButton.RightButton
    QT_HORIZONTAL = QtCore.Qt.Orientation.Horizontal
except Exception:
    try:
        from PyQt5 import QtCore, QtWidgets  # type: ignore

        QT_PEN_SOLID = QtCore.Qt.SolidLine
        QT_RIGHT_BUTTON = QtCore.Qt.RightButton
        QT_HORIZONTAL = QtCore.Qt.Horizontal
    except Exception as _exc_qt:  # pragma: no cover - import-time fallback
        QtCore = None  # type: ignore
        QtWidgets = None  # type: ignore
        if _PG_IMPORT_ERROR is None:
            _PG_IMPORT_ERROR = _exc_qt

if pg is not None:
    pg.setConfigOptions(antialias=True)

A_COLORS = [
    (25, 118, 210),
    (30, 136, 229),
    (66, 165, 245),
    (100, 181, 246),
    (21, 101, 192),
    (13, 71, 161),
]
B_COLORS = [
    (230, 81, 0),
    (244, 81, 30),
    (255, 112, 67),
    (255, 138, 101),
    (216, 67, 21),
    (191, 54, 12),
]


def _require_pyqtgraph() -> None:
    if pg is None or QtWidgets is None or QtCore is None:
        raise ImportError(
            "pyqtgraph viewer requires pyqtgraph and PyQt (PyQt6 or PyQt5). "
            f"Original import error: {_PG_IMPORT_ERROR}"
        )


def _clear_qt_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_qt_layout(child_layout)


def export_plot_widget_image(plot_widget, file_path: str, width: int = 1800) -> str:
    _require_pyqtgraph()
    file_path = str(file_path)
    suffix = Path(file_path).suffix.lower()
    plot_item = getattr(plot_widget, "plotItem", plot_widget)

    if suffix == ".svg":
        exporter = pg_exporters.SVGExporter(plot_item)
    else:
        exporter = pg_exporters.ImageExporter(plot_item)
        try:
            params = exporter.parameters()
            if "width" in params:
                params["width"] = int(width)
        except Exception:
            pass
    exporter.export(file_path)
    return file_path


if pg is not None and QtWidgets is not None and QtCore is not None:

    class WavefrontInteractiveViewer(QtWidgets.QWidget):
        """Interactive A/B waveform viewer based on pyqtgraph.

        Features:
        - A/B traces drawn in the same plot with different colors
        - Each trace uses solid lines
        - Per-channel waveform selection via checkboxes
        - Default mode: left-drag pans the view
        - Right click toggles crosshair mode; while active, left-drag performs rectangle zoom
        - The visible x-range is shown as selected time window
        - Wavehead positions are marked for both ends
        - Raw signal plotted on Left Y-axis; Features/Filters plotted on Right Y-axis
        """

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.result_a: dict[str, Any] | None = None
            self.result_b: dict[str, Any] | None = None
            self.fs: float = FS
            self.title_prefix: str = ""
            self.file_a: str | None = None
            self.file_b: str | None = None
            self.series_a: dict[str, np.ndarray] = {}
            self.series_b: dict[str, np.ndarray] = {}
            self.checkboxes_a: dict[str, QtWidgets.QCheckBox] = {}
            self.checkboxes_b: dict[str, QtWidgets.QCheckBox] = {}
            self._full_x_range: tuple[float, float] = (0.0, 1.0)
            self._crosshair_enabled = False
            self._updating_checks = False
            self._build_ui()

        def _build_ui(self) -> None:
            root = QtWidgets.QHBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(6)

            splitter = QtWidgets.QSplitter(QT_HORIZONTAL, self)
            root.addWidget(splitter, 1)

            left_panel = QtWidgets.QWidget(splitter)
            left_panel.setMinimumWidth(250)
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            left_layout.setContentsMargins(6, 6, 6, 6)
            left_layout.setSpacing(6)

            self.label_hint = QtWidgets.QLabel(
                "控制说明：\n"
                "• 左键拖拽：框选缩放 / 选择可见时间窗口\n"
                "• 鼠标滚轮：放大/缩小\n"
                "• 图形内右键：激活十字光标读数\n"
                "• 重置视图：恢复全时间段显示"
            )
            self.label_hint.setWordWrap(True)
            left_layout.addWidget(self.label_hint)

            self.label_range = QtWidgets.QLabel("已选 / 可见时间窗口: --")
            self.label_range.setWordWrap(True)
            left_layout.addWidget(self.label_range)

            self.label_cursor = QtWidgets.QLabel("十字光标: 未激活")
            self.label_cursor.setWordWrap(True)
            left_layout.addWidget(self.label_cursor)

            btn_row = QtWidgets.QHBoxLayout()
            self.btn_reset_view = QtWidgets.QPushButton("重置视图")
            self.btn_reset_view.clicked.connect(self.reset_view)
            btn_row.addWidget(self.btn_reset_view)
            self.btn_clear_cursor = QtWidgets.QPushButton("清除光标")
            self.btn_clear_cursor.clicked.connect(self.clear_cursor)
            btn_row.addWidget(self.btn_clear_cursor)
            left_layout.addLayout(btn_row)

            self.group_a = QtWidgets.QGroupBox("A波形路径")
            self.layout_a = QtWidgets.QVBoxLayout(self.group_a)
            self.layout_a.setContentsMargins(6, 10, 6, 6)
            self.layout_a.setSpacing(4)
            left_layout.addWidget(self.group_a)

            self.group_b = QtWidgets.QGroupBox("B波形路径")
            self.layout_b = QtWidgets.QVBoxLayout(self.group_b)
            self.layout_b.setContentsMargins(6, 10, 6, 6)
            self.layout_b.setSpacing(4)
            left_layout.addWidget(self.group_b)
            left_layout.addStretch(1)

            right_panel = QtWidgets.QWidget(splitter)
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(4)

            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground("w")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.22)
            self.plot_widget.setLabel("bottom", "Time", units="us")
            self.plot_widget.setLabel("left", "Amplitude (原始信号)")
            self.plot_widget.setLabel("right", "Feature / Filtered (特征/滤波)")
            self._set_mouse_pan_mode()
            self.plot_widget.setMenuEnabled(False)
            right_layout.addWidget(self.plot_widget, 1)

            self.plot_item = self.plot_widget.getPlotItem()
            self.legend = self.plot_item.addLegend(offset=(12, 8))

            # ========== 新增：创建右侧坐标轴 ViewBox ==========
            self.vb_sec = pg.ViewBox()
            self.plot_item.scene().addItem(self.vb_sec)
            self.plot_item.showAxis('right')
            self.plot_item.getAxis('right').linkToView(self.vb_sec)
            self.vb_sec.setXLink(self.plot_item.vb)

            def updateViews():
                self.vb_sec.setGeometry(self.plot_item.vb.sceneBoundingRect())
                self.vb_sec.linkedViewChanged(self.plot_item.vb, self.vb_sec.XAxis)

            updateViews()
            self.plot_item.vb.sigResized.connect(updateViews)
            # ==================================================

            self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((50, 50, 50), width=1))
            self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((50, 50, 50), width=1))
            self.vline.hide()
            self.hline.hide()
            self.plot_item.addItem(self.vline, ignoreBounds=True)
            self.plot_item.addItem(self.hline, ignoreBounds=True)

            self.marker_items: list[Any] = []
            self.proxy_move = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60,
                                             slot=self._on_mouse_moved)
            self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
            self.plot_widget.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)

            splitter.setSizes([280, 1000])

        def _set_mouse_pan_mode(self) -> None:
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        def _set_mouse_rect_mode(self) -> None:
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        def _extract_series(self, result: dict[str, Any]) -> dict[str, np.ndarray]:
            algo = result.get("algorithm")
            out: dict[str, np.ndarray] = {}

            def add_series(name: str, value: Any) -> None:
                if value is None:
                    return
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1 or arr.size == 0:
                    return
                out[name] = arr

            if algo == "rdp_local_aic":
                add_series("原始信号", result.get("x_raw"))
                add_series("小波降噪", result.get("x_wavelet"))
                add_series("SG平滑", result.get("x_smooth"))
                add_series("斜率特性", result.get("slope_dev"))
                add_series("幅值 dev", result.get("amp_dev"))
                add_series("AIC信号", result.get("aic_signal"))
            elif algo == "rdp_global_iceemdan_teo":
                add_series("原始信号", result.get("x_raw"))
                add_series("预处理信号", result.get("x_proc"))
                add_series("TEO能量算子 输入", result.get("teo_input"))
                add_series("IMF 平滑", result.get("x_smooth"))
                add_series(result.get("feature_name", "Feature"), result.get("feature"))
                imfs = result.get("imfs")
                if imfs is not None:
                    imfs = np.asarray(imfs, dtype=float)
                    if imfs.ndim == 2 and imfs.shape[0] >= 1:
                        add_series("IMF1", imfs[0])
                    if imfs.ndim == 2 and imfs.shape[0] >= 2:
                        add_series("IMF2", imfs[1])
                add_series("Residue", result.get("residue"))
            else:
                add_series("Raw", result.get("x_raw"))
            return out

        def _default_visible_names(self, result: dict[str, Any]) -> set[str]:
            algo = result.get("algorithm")
            if algo == "rdp_local_aic":
                return {"Raw", "Wavelet", "Smoothed"}
            if algo == "rdp_global_iceemdan_teo":
                return {"Raw", "Preprocessed", "TEO Input"}
            return {"Raw"}

        def _rebuild_trace_controls(self) -> None:
            self._updating_checks = True
            _clear_qt_layout(self.layout_a)
            _clear_qt_layout(self.layout_b)
            self.checkboxes_a.clear()
            self.checkboxes_b.clear()

            def populate(layout, names, store_dict, defaults):
                for name in names:
                    cb = QtWidgets.QCheckBox(name)
                    cb.setChecked(name in defaults)
                    cb.toggled.connect(self._render_plot)
                    layout.addWidget(cb)
                    store_dict[name] = cb
                layout.addStretch(1)

            defaults_a = self._default_visible_names(self.result_a or {})
            defaults_b = self._default_visible_names(self.result_b or {})
            populate(self.layout_a, self.series_a.keys(), self.checkboxes_a, defaults_a)
            populate(self.layout_b, self.series_b.keys(), self.checkboxes_b, defaults_b)
            self._updating_checks = False

        def _channel_pen(self, channel: str, idx: int):
            colors = A_COLORS if channel == "A" else B_COLORS
            return pg.mkPen(color=colors[idx % len(colors)], width=1.8, style=QT_PEN_SOLID)

        def _head_pen(self, channel: str):
            color = A_COLORS[0] if channel == "A" else B_COLORS[0]
            return pg.mkPen(color=color, width=2.2, style=QT_PEN_SOLID)

        def _plot_channel_series(self, channel: str, result: dict[str, Any], series_map: dict[str, np.ndarray],
                                 checks: dict[str, QtWidgets.QCheckBox]) -> None:
            for idx, (name, y) in enumerate(series_map.items()):
                cb = checks.get(name)
                if cb is None or not cb.isChecked():
                    continue
                x_us = time_vector(len(y), self.fs) * 1e6

                # ===== 修改：区分原始信号与其余信号的坐标轴 =====
                if name == "原始信号":
                    # 原始信号画在左坐标轴
                    self.plot_item.plot(
                        x_us,
                        np.asarray(y, dtype=float),
                        pen=self._channel_pen(channel, idx),
                        name=f"{channel}-{name}",
                    )
                else:
                    # 其他特征信号画在右坐标轴
                    curve = pg.PlotCurveItem(
                        x_us,
                        np.asarray(y, dtype=float),
                        pen=self._channel_pen(channel, idx),
                        name=f"{channel}-{name} (右轴)"
                    )
                    self.vb_sec.addItem(curve)
                    # 手动向图例添加标签
                    if self.legend is not None:
                        self.legend.addItem(curve, f"{channel}-{name} (右轴)")
                # ==================================================

            t_head_us = float(result.get("t_head", 0.0)) * 1e6
            head_line = pg.InfiniteLine(pos=t_head_us, angle=90, movable=False, pen=self._head_pen(channel))
            self.plot_item.addItem(head_line, ignoreBounds=True)
            self.marker_items.append(head_line)

            y_ref = None
            is_right_axis = False  # <--- 新增：用于记录标记点应该放在哪个坐标轴

            x_smooth = result.get("x_smooth")
            if x_smooth is not None:
                arr = np.asarray(x_smooth, dtype=float)
                idx_head = int(
                    np.clip(int(round(result.get("idx_head", 0))), 0, max(len(arr) - 1, 0))) if arr.size else 0
                if arr.size:
                    y_ref = float(arr[idx_head])
                    is_right_axis = True  # <--- 新增：x_smooth 画在右轴，标记点也要去右轴

            if y_ref is None:
                x_raw = np.asarray(result.get("x_raw", []), dtype=float)
                idx_head = int(
                    np.clip(int(round(result.get("idx_head", 0))), 0, max(len(x_raw) - 1, 0))) if x_raw.size else 0
                if x_raw.size:
                    y_ref = float(x_raw[idx_head])
                    is_right_axis = False  # <--- 新增：如果降级使用 x_raw，则画在左轴

            if y_ref is not None:
                scatter = pg.ScatterPlotItem(
                    [t_head_us],
                    [y_ref],
                    pen=self._head_pen(channel),
                    brush=pg.mkBrush(A_COLORS[0] if channel == "A" else B_COLORS[0]),
                    size=9,
                    symbol="o",
                )

                # ========== 修改处：根据数据的来源，将散点加入对应的坐标轴 ==========
                if is_right_axis:
                    self.vb_sec.addItem(scatter)
                else:
                    self.plot_item.addItem(scatter)
                # ====================================================================

                self.marker_items.append(scatter)

        def _render_plot(self) -> None:
            if self._updating_checks:
                return
            x_range = None
            try:
                x_range = tuple(self.plot_widget.getViewBox().viewRange()[0])
            except Exception:
                x_range = None

            self.plot_item.clear()
            self.vb_sec.clear()  # <--- 新增：清理右轴

            if self.legend is None:
                self.legend = self.plot_item.addLegend(offset=(12, 8))
            else:
                try:
                    self.legend.clear()
                except Exception:
                    self.legend = self.plot_item.addLegend(offset=(12, 8))
            self.marker_items.clear()
            self.plot_item.addItem(self.vline, ignoreBounds=True)
            self.plot_item.addItem(self.hline, ignoreBounds=True)
            if not self._crosshair_enabled:
                self.vline.hide()
                self.hline.hide()

            if self.result_a is not None:
                self._plot_channel_series("A", self.result_a, self.series_a, self.checkboxes_a)
            if self.result_b is not None:
                self._plot_channel_series("B", self.result_b, self.series_b, self.checkboxes_b)

            # self.plot_widget.setTitle(self.title_prefix or "Wavefront Interactive Viewer")
            self.plot_widget.setLabel("bottom", "Time", units="us")
            self.plot_widget.setLabel("left", "Amplitude (原始信号)")
            self.plot_widget.setLabel("right", "Feature / Filtered (特征/滤波)")  # <--- 新增：设置右侧标签
            self.plot_widget.showGrid(x=True, y=True, alpha=0.22)

            if x_range is None:
                self.reset_view()
            else:
                self.plot_widget.setXRange(float(x_range[0]), float(x_range[1]), padding=0.0)
                self.plot_widget.enableAutoRange(axis="y", enable=True)
                self.vb_sec.enableAutoRange(axis="y", enable=True)  # <--- 新增：右轴开启自动缩放
                self._update_range_label()

        def _on_x_range_changed(self, *args) -> None:
            self._update_range_label()

        def _update_range_label(self) -> None:
            try:
                x0, x1 = self.plot_widget.getViewBox().viewRange()[0]
                dt = float(x1 - x0)
                self.label_range.setText(
                    f"已选择/窗口时间: [{x0:.3f}, {x1:.3f}] us    Δt = {dt:.3f} us"
                )
            except Exception:
                self.label_range.setText("已选择/窗口时间: --")

        def _on_mouse_clicked(self, event) -> None:
            if event.button() != QT_RIGHT_BUTTON:
                return
            if not self.plot_widget.sceneBoundingRect().contains(event.scenePos()):
                return
            if self._crosshair_enabled:
                self.clear_cursor()
            else:
                point = self.plot_widget.getViewBox().mapSceneToView(event.scenePos())
                self._crosshair_enabled = True
                self.vline.show()
                self.hline.show()
                self._set_mouse_rect_mode()
                self._set_crosshair(point.x(), point.y())
            event.accept()

        def _on_mouse_moved(self, evt) -> None:
            if not self._crosshair_enabled:
                return
            pos = evt[0] if isinstance(evt, tuple) else evt
            if not self.plot_widget.sceneBoundingRect().contains(pos):
                return
            point = self.plot_widget.getViewBox().mapSceneToView(pos)
            self._set_crosshair(point.x(), point.y())

        def _set_crosshair(self, x_us: float, y_val: float) -> None:
            self.vline.setPos(float(x_us))
            self.hline.setPos(float(y_val))
            self.label_cursor.setText(f"指针准星： t = {float(x_us):.3f} us, y = {float(y_val):.6g}")

        def clear_cursor(self) -> None:
            self._crosshair_enabled = False
            self.vline.hide()
            self.hline.hide()
            self._set_mouse_pan_mode()
            self.label_cursor.setText("指针准星：未启用")

        def reset_view(self) -> None:
            x0, x1 = self._full_x_range
            self.plot_widget.setXRange(float(x0), float(x1), padding=0.02)
            self.plot_widget.enableAutoRange(axis="y", enable=True)
            self.vb_sec.enableAutoRange(axis="y", enable=True)  # <--- 新增：右轴开启自动缩放
            self._update_range_label()

        def set_results(
                self,
                result_a: dict[str, Any],
                result_b: dict[str, Any],
                *,
                file_a: str | None = None,
                file_b: str | None = None,
                fs: float | None = None,
                title_prefix: str | None = None,
        ) -> None:
            self.result_a = result_a
            self.result_b = result_b
            self.file_a = file_a
            self.file_b = file_b
            self.fs = float(fs) if fs is not None else _result_fs(result_a, None)
            self.title_prefix = title_prefix or result_a.get("algorithm_label", "Wavefront Interactive Viewer")
            self.series_a = self._extract_series(result_a)
            self.series_b = self._extract_series(result_b)

            max_len = max(
                max((len(v) for v in self.series_a.values()), default=0),
                max((len(v) for v in self.series_b.values()), default=0),
            )
            max_x = max_len / self.fs * 1e6 if self.fs > 0 else 1.0
            self._full_x_range = (0.0, max_x)

            self._rebuild_trace_controls()
            self.clear_cursor()
            self._render_plot()
            self.reset_view()

        def clear_results(self) -> None:
            self.result_a = None
            self.result_b = None
            self.series_a.clear()
            self.series_b.clear()
            self._updating_checks = True
            _clear_qt_layout(self.layout_a)
            _clear_qt_layout(self.layout_b)
            self.checkboxes_a.clear()
            self.checkboxes_b.clear()
            self._updating_checks = False
            self.plot_item.clear()
            self.vb_sec.clear()  # <--- 新增：清理右轴
            self.plot_item.addItem(self.vline, ignoreBounds=True)
            self.plot_item.addItem(self.hline, ignoreBounds=True)
            self.clear_cursor()
            self.label_range.setText("Selected / visible time window: --")
            self.plot_widget.setTitle("Wavefront Interactive Viewer")

else:

    class WavefrontInteractiveViewer:  # pragma: no cover - fallback class for missing GUI deps
        def __init__(self, *args, **kwargs) -> None:
            _require_pyqtgraph()

        def set_results(self, *args, **kwargs) -> None:
            _require_pyqtgraph()

        def clear_results(self) -> None:
            _require_pyqtgraph()

__all__ = [
    "FS",
    "WavefrontInteractiveViewer",
    "build_result_summary",
    "create_result_figure_ab",
    "create_result_figure_single",
    "export_plot_widget_image",
    "print_result_summary",
    "save_figure",
    "save_result_summary_json",
]