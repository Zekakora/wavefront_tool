"""Wavefront data loading helpers for PyQt-friendly integration.

This module keeps file I/O and CSV pairing logic separate from the detection
algorithms, so the UI layer can stay thin.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import DefaultDict, Iterable, Optional

import numpy as np
import pandas as pd

FS = 4.2e6

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
    re.VERBOSE | re.IGNORECASE,
)


def load_csv_no_header(
    csv_path: str,
    *,
    index_col: int = 0,
    value_col: int = 1,
    dtype: type = float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read a two-column headerless CSV.

    Expected default format:
    - column 0: sample index
    - column 1: signal value
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    max_col = max(index_col, value_col)
    if df.shape[1] <= max_col:
        raise ValueError(
            f"CSV 列数不足。当前列数 = {df.shape[1]}，至少需要到第 {max_col + 1} 列。"
        )

    idx = df.iloc[:, index_col].to_numpy(dtype=dtype)
    x = df.iloc[:, value_col].to_numpy(dtype=dtype)
    return idx, x, df


def load_signal_only(
    csv_path: str,
    *,
    value_col: int = 1,
    dtype: type = float,
) -> np.ndarray:
    """Read only the signal column for direct algorithm calls."""
    _, x, _ = load_csv_no_header(csv_path, value_col=value_col, dtype=dtype)
    return x


def load_ab_signals(
    file_a: str,
    file_b: str,
    *,
    index_col: int = 0,
    value_col: int = 1,
    dtype: type = float,
) -> dict:
    """Read A/B end signals together."""
    idx_a, x_a, df_a = load_csv_no_header(
        file_a, index_col=index_col, value_col=value_col, dtype=dtype
    )
    idx_b, x_b, df_b = load_csv_no_header(
        file_b, index_col=index_col, value_col=value_col, dtype=dtype
    )
    return {
        "file_a": file_a,
        "file_b": file_b,
        "idx_a": idx_a,
        "idx_b": idx_b,
        "x_a": x_a,
        "x_b": x_b,
        "df_a": df_a,
        "df_b": df_b,
    }


def time_vector(n: int, fs: float = FS) -> np.ndarray:
    return np.arange(int(n), dtype=float) / float(fs)


def extract_match_key(filename: str) -> Optional[str]:
    """Extract the pairing key from a filename.

    Example:
        2026-03-13_11-56-36_321-686-374.csv -> 11-56-36_321
    """
    name = os.path.basename(filename)
    match = FILENAME_PATTERN.match(name)
    if not match:
        return None
    return f"{match.group('time')}_{match.group('head')}"


def scan_and_group_csvs(folder: str) -> dict[str, list[str]]:
    """Group CSV file names by match key."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"不是有效文件夹: {folder}")

    groups: DefaultDict[str, list[str]] = defaultdict(list)

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".csv"):
            continue
        key = extract_match_key(name)
        if key is not None:
            groups[key].append(name)

    return dict(groups)


def build_pairs(folder: str, target_key: Optional[str] = None) -> list[tuple[str, str, str]]:
    """Build A/B file pairs using the filename rule from the original script."""
    groups = scan_and_group_csvs(folder)
    pairs: list[tuple[str, str, str]] = []

    for key, names in groups.items():
        if target_key is not None and key != target_key:
            continue

        if len(names) < 2:
            continue

        if len(names) == 2:
            pairs.append((key, os.path.join(folder, names[0]), os.path.join(folder, names[1])))
            continue

        for i in range(0, len(names) - 1, 2):
            sub_key = key if len(names) == 2 else f"{key}__pair{i // 2 + 1}"
            pairs.append(
                (
                    sub_key,
                    os.path.join(folder, names[i]),
                    os.path.join(folder, names[i + 1]),
                )
            )

    return pairs


def list_csv_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"不是有效文件夹: {folder}")
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(".csv")
    ]


__all__ = [
    "FS",
    "FILENAME_PATTERN",
    "build_pairs",
    "extract_match_key",
    "list_csv_files",
    "load_ab_signals",
    "load_csv_no_header",
    "load_signal_only",
    "scan_and_group_csvs",
    "time_vector",
]
