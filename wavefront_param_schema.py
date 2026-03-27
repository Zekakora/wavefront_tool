from __future__ import annotations

from typing import Any

try:
    from .wavefront_algo_rdp_aic import DEFAULT_RDP_LOCAL_AIC_PARAMS
    from .wavefront_algo_iceemdan_teo import DEFAULT_ICEEMDAN_TEO_PARAMS
except ImportError:
    from wavefront_algo_rdp_aic import DEFAULT_RDP_LOCAL_AIC_PARAMS
    from wavefront_algo_iceemdan_teo import DEFAULT_ICEEMDAN_TEO_PARAMS


def _pretty_label(key: str) -> str:
    parts = key.split("_")
    return " ".join(p.upper() if len(p) <= 3 else p.capitalize() for p in parts)


_RDP_META: dict[str, dict[str, Any]] = {
    "fs": {"label": "Sampling Frequency", "min": 1.0, "max": 1e9, "decimals": 2, "step": 1e5},
    "pre_n": {"label": "Pre-noise Samples", "min": 1, "max": 1_000_000},
    "rough_k": {"label": "Rough Threshold K", "min": 0.1, "max": 100.0, "decimals": 3, "step": 0.1},
    "rough_consecutive": {"label": "Rough Consecutive Points", "min": 1, "max": 10_000},
    "threshold_sg_window": {"label": "Threshold SG Window", "min": 3, "max": 9999},
    "threshold_sg_poly": {"label": "Threshold SG Polyorder", "min": 1, "max": 20},
    "rdp_left_us": {"label": "RDP Left Window (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 1.0},
    "rdp_right_us": {"label": "RDP Right Window (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 1.0},
    "rdp_epsilon": {"label": "RDP Epsilon", "min": 0.0, "max": 10.0, "decimals": 6, "step": 0.001},
    "rdp_sg_window": {"label": "RDP SG Window", "min": 3, "max": 9999},
    "rdp_sg_poly": {"label": "RDP SG Polyorder", "min": 1, "max": 20},
    "rdp_seg_slope_k": {"label": "RDP Segment Slope K", "min": 0.0, "max": 100.0, "decimals": 3, "step": 0.1},
    "rdp_seg_amp_k": {"label": "RDP Segment Amplitude K", "min": 0.0, "max": 100.0, "decimals": 3, "step": 0.1},
    "rdp_weak_factor": {"label": "RDP Weak Factor", "min": 0.0, "max": 10.0, "decimals": 4, "step": 0.05},
    "search_left_us": {"label": "Search Left (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "search_right_us": {"label": "Search Right (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "noise_guard_us": {"label": "Noise Guard (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.1},
    "noise_win_us": {"label": "Noise Window (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 1.0},
    "slope_win": {"label": "Slope Window", "min": 3, "max": 9999},
    "slope_poly": {"label": "Slope Polyorder", "min": 1, "max": 20},
    "amp_k": {"label": "Amplitude Threshold K", "min": 0.0, "max": 100.0, "decimals": 3, "step": 0.1},
    "slope_k": {"label": "Slope Threshold K", "min": 0.0, "max": 100.0, "decimals": 3, "step": 0.1},
    "min_consecutive": {"label": "Min Consecutive", "min": 1, "max": 10_000},
    "fit_n": {"label": "Fit Points", "min": 2, "max": 10_000},
    "polarity": {"label": "Polarity", "choices": ["auto", "positive", "negative"]},
    "aic_left_us": {"label": "AIC Left (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "aic_right_us": {"label": "AIC Right (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "aic_min_split": {"label": "AIC Min Split", "min": 1, "max": 10_000},
    "aic_smooth_win": {"label": "AIC Smooth Window", "min": 3, "max": 9999},
    "aic_smooth_poly": {"label": "AIC Smooth Polyorder", "min": 1, "max": 20},
    "wavelet_enabled": {"label": "Enable Wavelet Denoise"},
    "wavelet": {"label": "Wavelet Name"},
    "wavelet_level": {"label": "Wavelet Level", "min": 1, "max": 20},
    "wavelet_beta": {"label": "Wavelet Beta", "min": 0.0, "max": 20.0, "decimals": 4, "step": 0.1},
    "wavelet_threshold_scale": {"label": "Wavelet Threshold Scale", "min": 0.0, "max": 20.0, "decimals": 4, "step": 0.1},
    "wavelet_use_level_dependent_sigma": {"label": "Level-dependent Sigma"},
    "wavelet_mode": {"label": "Wavelet Padding Mode", "choices": ["symmetric", "periodization", "reflect", "zero", "constant"]},
}

_ICE_META: dict[str, dict[str, Any]] = {
    "fs": {"label": "Sampling Frequency", "min": 1.0, "max": 1e9, "decimals": 2, "step": 1e5},
    "rdp_preprocess": {"label": "RDP Preprocess"},
    "rdp_sg_window": {"label": "RDP SG Window", "min": 3, "max": 9999},
    "rdp_sg_polyorder": {"label": "RDP SG Polyorder", "min": 1, "max": 20},
    "rdp_epsilon": {"label": "RDP Epsilon", "min": 0.0, "max": 10.0, "decimals": 6, "step": 0.001},
    "rdp_pre_n": {"label": "RDP Pre-noise Samples", "min": 1, "max": 1_000_000},
    "rdp_post_check_n": {"label": "RDP Post-check Samples", "min": 1, "max": 1_000_000},
    "rdp_thr_method": {"label": "RDP Threshold Method", "choices": ["mad", "std"]},
    "rdp_k": {"label": "RDP Threshold K", "min": 0.0, "max": 100.0, "decimals": 3, "step": 0.1},
    "rdp_use_abs": {"label": "RDP Use Absolute Value"},
    "rdp_min_consecutive": {"label": "RDP Min Consecutive", "min": 1, "max": 10_000},
    "search_left_us": {"label": "Search Left (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "search_right_us": {"label": "Search Right (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "noise_guard_us": {"label": "Noise Guard (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.1},
    "min_pre_noise_us": {"label": "Min Pre-noise (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 0.5},
    "global_preprocess": {"label": "Global Preprocess"},
    "use_imf_mode": {"label": "IMF Mode", "choices": ["iimf1", "iimf12"]},
    "alpha2": {"label": "IMF2 Weight Alpha", "min": 0.0, "max": 10.0, "decimals": 4, "step": 0.05},
    "ensemble_size": {"label": "ICEEMDAN Ensemble Size", "min": 1, "max": 10_000},
    "noise_strength": {"label": "ICEEMDAN Noise Strength", "min": 0.0, "max": 10.0, "decimals": 4, "step": 0.01},
    "max_imfs": {"label": "Max IMFs", "min": 1, "max": 100},
    "random_state": {"label": "Random State", "min": -2_147_483_648, "max": 2_147_483_647},
    "use_mirror_pad": {"label": "Use Mirror Padding"},
    "mirror_pad_us": {"label": "Mirror Padding (us)", "min": 0.0, "max": 10_000.0, "decimals": 3, "step": 1.0},
    "pre_sg_window": {"label": "Pre-smoothing SG Window", "min": 3, "max": 9999},
    "sigma_k": {"label": "Threshold Sigma K", "min": 0.0, "max": 100.0, "decimals": 4, "step": 0.1},
    "min_peak_distance_samples": {"label": "Min Peak Distance", "min": 1, "max": 100_000},
    "cross_consecutive": {"label": "Cross Consecutive", "min": 1, "max": 10_000},
    "pick_mode": {"label": "Head Pick Mode", "choices": ["first_cross", "max", "first_sig_slope"]},
    "slope_polarity": {"label": "Slope Polarity", "choices": ["abs", "positive", "negative"]},
    "slope_smooth_win": {"label": "Slope Smooth Window", "min": 3, "max": 9999},
    "edge_ignore_samples": {"label": "Edge Ignore Samples", "min": 0, "max": 100_000},
}


KEY_FIELDS_RDP = [
    "pre_n",
    "rough_k",
    "rdp_epsilon",
    "search_left_us",
    "search_right_us",
    "amp_k",
]

KEY_FIELDS_ICE = [
    "rdp_epsilon",
    "ensemble_size",
    "noise_strength",
    "sigma_k",
    "pick_mode",
    "cross_consecutive",
]


def _infer_field_type(default_value: Any, field_meta: dict[str, Any]) -> str:
    if "choices" in field_meta:
        return "choice"
    if isinstance(default_value, bool):
        return "bool"
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return "int"
    if isinstance(default_value, float):
        return "float"
    return "str"


def build_schema(defaults: dict[str, Any], meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for key, value in defaults.items():
        field_meta = dict(meta.get(key, {}))
        schema.append(
            {
                "key": key,
                "label": field_meta.pop("label", _pretty_label(key)),
                "default": value,
                "type": field_meta.pop("type", _infer_field_type(value, field_meta)),
                **field_meta,
            }
        )
    return schema


ALGORITHM_DEFINITIONS: dict[str, dict[str, Any]] = {
    "rdp_local_aic": {
        "label": "RDP + Local AIC",
        "defaults": dict(DEFAULT_RDP_LOCAL_AIC_PARAMS),
        "schema": build_schema(DEFAULT_RDP_LOCAL_AIC_PARAMS, _RDP_META),
        "key_fields": list(KEY_FIELDS_RDP),
        "dialog_title": "RDP + Local AIC Full Parameter Table",
    },
    "rdp_global_iceemdan_teo": {
        "label": "RDP + Global ICEEMDAN-TEO",
        "defaults": dict(DEFAULT_ICEEMDAN_TEO_PARAMS),
        "schema": build_schema(DEFAULT_ICEEMDAN_TEO_PARAMS, _ICE_META),
        "key_fields": list(KEY_FIELDS_ICE),
        "dialog_title": "RDP + Global ICEEMDAN-TEO Full Parameter Table",
    },
}


def get_algorithm_ids() -> list[str]:
    return list(ALGORITHM_DEFINITIONS.keys())


__all__ = [
    "ALGORITHM_DEFINITIONS",
    "KEY_FIELDS_ICE",
    "KEY_FIELDS_RDP",
    "build_schema",
    "get_algorithm_ids",
]
