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


def _plot_rdp_row_local(ax_row, result: dict[str, Any], *, end_label: str, fs: float, x_left: float, x_right: float) -> None:
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


def _plot_ice_row_local(ax_row, result: dict[str, Any], *, end_label: str, fs: float, x_left: float, x_right: float) -> None:
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


def save_result_summary_json(result: dict[str, Any], save_path: str, *, ensure_ascii: bool = False, indent: int = 2) -> str:
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
        print(f"{prefix}搜索窗 = [{result['search_i0'] / result['params']['fs'] * 1e6:.3f}, {result['search_i1'] / result['params']['fs'] * 1e6:.3f}] us")
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
