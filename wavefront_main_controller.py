from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

# Matplotlib Qt backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei', 'SimHei', 'DejaVu Sans']

# -----------------------------------------------------------------------------
# 路径准备：main_modified.py 位于 /mnt/data，模块位于 /mnt/data/wavefront_modular
# 实际迁移时，建议把本文件、main_modified.py、wavefront_modular 放在同一工程目录
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
MODULAR_DIR = CURRENT_DIR / "wavefront_modular"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if MODULAR_DIR.exists() and str(MODULAR_DIR) not in sys.path:
    sys.path.insert(0, str(MODULAR_DIR))

from main_m import Ui_MainWindow
from wavelet_transform_panel import WaveletTransformPanel
from wavefront_algo_iceemdan_teo import detect_wavefront_rdp_global_iceemdan_teo
from wavefront_algo_rdp_aic import detect_wavefront_rdp
from wavefront_data_io import build_pairs, extract_match_key, list_csv_files, load_ab_signals
from wavefront_param_dialog import FullParameterDialog
from wavefront_param_schema import ALGORITHM_DEFINITIONS
from wavefront_param_store import ParameterStore
from wavefront_plot_save import create_result_figure_ab, save_figure, save_result_summary_json


ALGORITHM_CALLABLES: dict[str, Callable[..., dict[str, Any]]] = {
    "rdp_local_aic": detect_wavefront_rdp,
    "rdp_global_iceemdan_teo": detect_wavefront_rdp_global_iceemdan_teo,
}

ALGORITHM_ORDER = [
    "rdp_local_aic",
    "rdp_global_iceemdan_teo",
]

PAIR_MODE_AUTO = "auto"
PAIR_MODE_KEYWORD = "keyword"


class WavefrontMainController(QMainWindow):
    """主窗口逻辑控制器。

    设计思路：
    1. main_modified.py 保持纯 UI 壳，不做手工逻辑修改。
    2. 本文件专门负责 UI 控件与算法/参数/绘图的映射。
    3. 关键参数区使用 UI 文件里已经定义好的动态属性：
       - algorithmId
       - paramKey
       - paramType
       - fieldKey
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.param_store = ParameterStore()
        self.settings = QSettings("WavefrontLocator", "WavefrontLocatorUI")

        self.canvas: FigureCanvas | None = None
        self.nav_toolbar: NavigationToolbar | None = None
        self.current_figure = None
        self.current_result_bundle: dict[str, Any] | None = None
        self.last_output_dir_used: str = ""

        self._key_param_widgets: dict[str, dict[str, QtWidgets.QWidget]] = {}

        self.wavelet_panel: WaveletTransformPanel | None = None

        self._prepare_algorithm_combo()
        self._prepare_plot_area()
        self._prepare_wavelet_page()
        self._setup_module_toolbar()
        self._collect_key_param_widgets()
        self._load_general_settings()
        self._load_key_params_from_store()
        self._wire_signals()
        self._sync_pair_keyword_enabled()
        self._sync_algorithm_page()
        self._update_result_labels_idle()

    # ------------------------------------------------------------------
    # 初始化与控件装配
    # ------------------------------------------------------------------
    def _prepare_algorithm_combo(self) -> None:
        """给算法下拉框补充 userData，避免后续仅靠文本判断。"""
        combo = self.ui.comboAlgorithm
        combo.blockSignals(True)
        for index, algorithm_id in enumerate(ALGORITHM_ORDER):
            if index < combo.count():
                combo.setItemData(index, algorithm_id)
        combo.blockSignals(False)

    def _prepare_plot_area(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.ui.framePlotContainer)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        placeholder = QtWidgets.QLabel(
            "Plot area\n\nRun detection to display the A/B result figure here."
        )
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #666; border: 1px dashed #BBB; padding: 24px;")
        layout.addWidget(placeholder)
        self._plot_placeholder = placeholder


    def _prepare_wavelet_page(self) -> None:
        if self.ui.pageReserved.layout() is None:
            layout = QtWidgets.QVBoxLayout(self.ui.pageReserved)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        else:
            layout = self.ui.pageReserved.layout()

        self.wavelet_panel = WaveletTransformPanel(self.ui.pageReserved)
        layout.addWidget(self.wavelet_panel)

    def _setup_module_toolbar(self) -> None:
        # 只保留两个切页动作
        self.actionGroupModules = QtGui.QActionGroup(self)
        self.actionGroupModules.setExclusive(True)

        self.actionShowWavefront = QtGui.QAction("波头识别", self)
        self.actionShowWavefront.setCheckable(True)

        self.actionShowWavelet = QtGui.QAction("小波变换", self)
        self.actionShowWavelet.setCheckable(True)

        self.actionGroupModules.addAction(self.actionShowWavefront)
        self.actionGroupModules.addAction(self.actionShowWavelet)

        self.actionShowWavefront.triggered.connect(lambda: self._switch_main_page(0))
        self.actionShowWavelet.triggered.connect(lambda: self._switch_main_page(1))

        # 菜单栏直接放 QAction，不使用下拉子菜单
        self.ui.menubar.clear()
        self.ui.menubar.addAction(self.actionShowWavefront)
        self.ui.menubar.addAction(self.actionShowWavelet)

        # 工具栏隐藏
        self.ui.toolBar.hide()

    def _switch_main_page(self, page_index: int, *, save: bool = True) -> None:
        page_index = 0 if int(page_index) <= 0 else 1
        self.ui.mainStackedWidget.setCurrentIndex(page_index)
        self.actionShowWavefront.setChecked(page_index == 0)
        self.actionShowWavelet.setChecked(page_index == 1)
        if save:
            self.settings.setValue("main_page_index", page_index)

    def _collect_key_param_widgets(self) -> None:
        self._key_param_widgets = {algo_id: {} for algo_id in ALGORITHM_ORDER}
        pages = [self.ui.pageRdpLocalAicKeyParams, self.ui.pageIceemdanTeoKeyParams]

        for page in pages:
            widgets = page.findChildren(QtWidgets.QWidget)
            for widget in widgets:
                algorithm_id = widget.property("algorithmId")
                param_key = widget.property("paramKey")
                if not algorithm_id or not param_key:
                    continue
                self._key_param_widgets[str(algorithm_id)][str(param_key)] = widget

    def _wire_signals(self) -> None:
        # 算法切换
        self.ui.comboAlgorithm.currentIndexChanged.connect(self._on_algorithm_changed)

        # 通用输入项持久化/联动
        self.ui.btnBrowseInputDirA.clicked.connect(lambda: self._browse_dir_for(self.ui.editInputDirA))
        self.ui.btnBrowseInputDirB.clicked.connect(lambda: self._browse_dir_for(self.ui.editInputDirB))
        self.ui.btnBrowseOutputDir.clicked.connect(lambda: self._browse_dir_for(self.ui.editOutputDir))

        self.ui.editInputDirA.editingFinished.connect(self._save_general_settings)
        self.ui.editInputDirB.editingFinished.connect(self._save_general_settings)
        self.ui.editOutputDir.editingFinished.connect(self._save_general_settings)
        self.ui.editPairKeyword.editingFinished.connect(self._save_general_settings)
        self.ui.editSensorDistanceM.editingFinished.connect(self._save_general_settings)
        self.ui.editSamplingFreqMHz.editingFinished.connect(self._on_wave_speed_changed)
        self.ui.comboMatchMode.currentIndexChanged.connect(self._on_pair_mode_changed)

        # 关键参数区 -> 参数存储
        for algorithm_id, widget_map in self._key_param_widgets.items():
            for param_key, widget in widget_map.items():
                self._connect_param_widget(widget, algorithm_id, param_key)

        # 完整参数表
        self.ui.btnOpenRdpLocalAicParamDialog.clicked.connect(
            lambda: self._open_full_param_dialog("rdp_local_aic")
        )
        self.ui.btnOpenIceemdanTeoParamDialog.clicked.connect(
            lambda: self._open_full_param_dialog("rdp_global_iceemdan_teo")
        )

        # 功能按钮
        self.ui.btnRunDetection.clicked.connect(self.run_detection)
        self.ui.btnSaveResult.clicked.connect(self.save_current_result_figure)
        self.ui.btnClearResult.clicked.connect(self.clear_current_result)

    def _connect_param_widget(self, widget: QtWidgets.QWidget, algorithm_id: str, param_key: str) -> None:
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.editingFinished.connect(
                lambda aid=algorithm_id, key=param_key, w=widget: self._save_single_key_param(aid, key, w)
            )
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(
                lambda _=0, aid=algorithm_id, key=param_key, w=widget: self._save_single_key_param(aid, key, w)
            )
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.toggled.connect(
                lambda _=False, aid=algorithm_id, key=param_key, w=widget: self._save_single_key_param(aid, key, w)
            )

    # ------------------------------------------------------------------
    # 通用设置与参数同步
    # ------------------------------------------------------------------
    def _load_general_settings(self) -> None:
        self.ui.editInputDirA.setText(self.settings.value("input_dir_a", "", str))
        self.ui.editInputDirB.setText(self.settings.value("input_dir_b", "", str))
        self.ui.editOutputDir.setText(self.settings.value("output_dir", "", str))
        self.ui.editPairKeyword.setText(self.settings.value("pair_keyword", "", str))
        self.ui.editSensorDistanceM.setText(self.settings.value("sensor_distance_m", "", str))
        self.ui.editSamplingFreqMHz.setText(self.settings.value("wave_speed_mps", "299792458", str))

        saved_pair_mode = self.settings.value("pair_mode", PAIR_MODE_AUTO, str)
        self.ui.comboMatchMode.setCurrentIndex(0 if saved_pair_mode == PAIR_MODE_AUTO else 1)

        saved_algorithm_id = self.settings.value("algorithm_id", ALGORITHM_ORDER[0], str)
        target_index = 0
        for i, algorithm_id in enumerate(ALGORITHM_ORDER):
            if algorithm_id == saved_algorithm_id:
                target_index = i
                break
        self.ui.comboAlgorithm.setCurrentIndex(target_index)

        saved_page_index = self.settings.value("main_page_index", 0, int)
        self._switch_main_page(saved_page_index, save=False)

    def _save_general_settings(self) -> None:
        config = self.collect_general_config()
        self.settings.setValue("input_dir_a", config["input_dir_a"])
        self.settings.setValue("input_dir_b", config["input_dir_b"])
        self.settings.setValue("output_dir", config["output_dir"])
        self.settings.setValue("pair_keyword", config["pair_keyword"])
        self.settings.setValue("sensor_distance_m", config["sensor_distance_m"])
        self.settings.setValue("wave_speed_mps", str(config["wave_speed_mps"]))
        self.settings.setValue("pair_mode", config["pair_mode"])
        self.settings.setValue("algorithm_id", config["algorithm_id"])
        self.settings.setValue("main_page_index", self.ui.mainStackedWidget.currentIndex())

    def _on_pair_mode_changed(self) -> None:
        self._sync_pair_keyword_enabled()
        self._save_general_settings()

    def _sync_pair_keyword_enabled(self) -> None:
        is_keyword_mode = self.current_pair_mode() == PAIR_MODE_KEYWORD
        self.ui.editPairKeyword.setEnabled(is_keyword_mode)
        self.ui.labelPairKeyword.setEnabled(is_keyword_mode)

    def _on_wave_speed_changed(self) -> None:
        self._save_general_settings()

    def _load_key_params_from_store(self) -> None:
        for algorithm_id, widget_map in self._key_param_widgets.items():
            params = self.param_store.get_params(algorithm_id)
            for param_key, widget in widget_map.items():
                if param_key in params:
                    self._set_widget_value(widget, params[param_key])

    def _refresh_key_params_for_algorithm(self, algorithm_id: str) -> None:
        params = self.param_store.get_params(algorithm_id)
        widget_map = self._key_param_widgets.get(algorithm_id, {})
        for param_key, widget in widget_map.items():
            if param_key in params:
                self._set_widget_value(widget, params[param_key])

    def _save_single_key_param(self, algorithm_id: str, param_key: str, widget: QtWidgets.QWidget) -> None:
        try:
            value = self._read_widget_value(widget)
            self.param_store.set_param(algorithm_id, param_key, value, save=True)
        except Exception as exc:
            QMessageBox.warning(self, "Parameter Error", f"{algorithm_id}.{param_key}\n\n{exc}")
            self._refresh_key_params_for_algorithm(algorithm_id)

    def _open_full_param_dialog(self, algorithm_id: str) -> None:
        dialog = FullParameterDialog(algorithm_id, self.param_store, self)
        dialog.paramsApplied.connect(lambda _values, aid=algorithm_id: self._on_full_params_applied(aid))
        dialog.exec()

    def _on_full_params_applied(self, algorithm_id: str) -> None:
        # 关键参数页刷新
        self._refresh_key_params_for_algorithm(algorithm_id)

        # 波速为主页面通用参数；完整参数表中的 fs 仅作为算法采样率使用，不再回写主页面。
        self._save_general_settings()

    def _on_algorithm_changed(self) -> None:
        self._sync_algorithm_page()
        self._save_general_settings()

    def _sync_algorithm_page(self) -> None:
        self.ui.stackedAlgorithmKeyParams.setCurrentIndex(self.ui.comboAlgorithm.currentIndex())
        algorithm_id = self.current_algorithm_id()
        algorithm_label = ALGORITHM_DEFINITIONS[algorithm_id]["label"]
        self.ui.valueCurrentAlgorithm.setText(algorithm_label)

    # ------------------------------------------------------------------
    # 通用配置读取
    # ------------------------------------------------------------------
    def current_algorithm_id(self) -> str:
        data = self.ui.comboAlgorithm.currentData()
        if data:
            return str(data)
        return ALGORITHM_ORDER[self.ui.comboAlgorithm.currentIndex()]

    def current_pair_mode(self) -> str:
        return PAIR_MODE_AUTO if self.ui.comboMatchMode.currentIndex() == 0 else PAIR_MODE_KEYWORD

    def current_wave_speed_mps(self) -> float:
        text = self.ui.editSamplingFreqMHz.text().strip() or "299792458"
        wave_speed = float(text)
        if wave_speed <= 0:
            raise ValueError("Wave speed must be > 0 m/s")
        return wave_speed

    def current_algorithm_fs_hz(self) -> float:
        algorithm_id = self.current_algorithm_id()
        params = self.param_store.get_params(algorithm_id)
        fs_hz = float(params.get("fs", 4.2e6))
        if fs_hz <= 0:
            raise ValueError("Algorithm sampling frequency must be > 0 Hz")
        return fs_hz

    def collect_general_config(self) -> dict[str, Any]:
        sensor_distance_text = self.ui.editSensorDistanceM.text().strip()
        sensor_distance_m = float(sensor_distance_text) if sensor_distance_text else None
        wave_speed_mps = self.current_wave_speed_mps()
        sampling_freq_hz = self.current_algorithm_fs_hz()

        return {
            "input_dir_a": self.ui.editInputDirA.text().strip(),
            "input_dir_b": self.ui.editInputDirB.text().strip(),
            "output_dir": self.ui.editOutputDir.text().strip(),
            "pair_mode": self.current_pair_mode(),
            "pair_keyword": self.ui.editPairKeyword.text().strip(),
            "sensor_distance_m": sensor_distance_m,
            "wave_speed_mps": wave_speed_mps,
            "sampling_freq_hz": sampling_freq_hz,
            "algorithm_id": self.current_algorithm_id(),
            "algorithm_label": ALGORITHM_DEFINITIONS[self.current_algorithm_id()]["label"],
        }

    def collect_run_config(self) -> dict[str, Any]:
        general = self.collect_general_config()
        algorithm_id = general["algorithm_id"]
        params = self.param_store.get_params(algorithm_id)
        params["fs"] = general["sampling_freq_hz"]
        return {
            **general,
            "params": params,
            "callable": ALGORITHM_CALLABLES[algorithm_id],
        }

    # ------------------------------------------------------------------
    # 文件配对逻辑
    # ------------------------------------------------------------------
    def _resolve_pairs(self, input_dir_a: str, input_dir_b: str, pair_mode: str, pair_keyword: str) -> list[tuple[str, str, str]]:
        if not input_dir_a:
            raise ValueError("Please select Data A directory first.")

        # 单目录模式：A 目录内自己配对
        if not input_dir_b or os.path.abspath(input_dir_a) == os.path.abspath(input_dir_b):
            pairs = build_pairs(input_dir_a, target_key=None)
            if pair_mode == PAIR_MODE_KEYWORD and pair_keyword:
                pairs = [
                    pair for pair in pairs
                    if pair_keyword in pair[0]
                    or pair_keyword in os.path.basename(pair[1])
                    or pair_keyword in os.path.basename(pair[2])
                ]
            return pairs

        files_a = list_csv_files(input_dir_a)
        files_b = list_csv_files(input_dir_b)

        if pair_mode == PAIR_MODE_KEYWORD:
            if not pair_keyword:
                raise ValueError("Keyword mode is selected, but Pair Keyword is empty.")
            files_a = [p for p in files_a if pair_keyword in os.path.basename(p)]
            files_b = [p for p in files_b if pair_keyword in os.path.basename(p)]

        if not files_a:
            raise ValueError("No CSV files found in Data A directory after filtering.")
        if not files_b:
            raise ValueError("No CSV files found in Data B directory after filtering.")

        pairs = self._pair_two_dirs_by_match_key(files_a, files_b)
        if pairs:
            return pairs

        # 回退：如果无法按 match key 配对，但两边数量一致，则按文件名排序一一对应。
        if len(files_a) == len(files_b):
            pairs = []
            files_a = sorted(files_a)
            files_b = sorted(files_b)
            for i, (fa, fb) in enumerate(zip(files_a, files_b), start=1):
                key = extract_match_key(os.path.basename(fa)) or extract_match_key(os.path.basename(fb)) or f"pair_{i:03d}"
                pairs.append((key, fa, fb))
            return pairs

        raise ValueError("Unable to pair files between A/B directories. Check filename rules or pair mode.")

    @staticmethod
    def _pair_two_dirs_by_match_key(files_a: list[str], files_b: list[str]) -> list[tuple[str, str, str]]:
        grouped_a: dict[str, list[str]] = {}
        grouped_b: dict[str, list[str]] = {}

        for path in files_a:
            key = extract_match_key(os.path.basename(path))
            if key:
                grouped_a.setdefault(key, []).append(path)
        for path in files_b:
            key = extract_match_key(os.path.basename(path))
            if key:
                grouped_b.setdefault(key, []).append(path)

        common_keys = sorted(set(grouped_a) & set(grouped_b))
        pairs: list[tuple[str, str, str]] = []
        for key in common_keys:
            list_a = sorted(grouped_a[key])
            list_b = sorted(grouped_b[key])
            n = min(len(list_a), len(list_b))
            for i in range(n):
                sub_key = key if n == 1 else f"{key}__pair{i + 1}"
                pairs.append((sub_key, list_a[i], list_b[i]))
        return pairs

    # ------------------------------------------------------------------
    # 检测主流程
    # ------------------------------------------------------------------
    def run_detection(self) -> None:
        try:
            run_cfg = self.collect_run_config()
            pairs = self._resolve_pairs(
                run_cfg["input_dir_a"],
                run_cfg["input_dir_b"],
                run_cfg["pair_mode"],
                run_cfg["pair_keyword"],
            )
            if not pairs:
                raise ValueError("No valid A/B pairs were found.")
        except Exception as exc:
            QMessageBox.warning(self, "Run Configuration Error", str(exc))
            return

        self._set_busy(True)
        self.ui.valueDetectionStatus.setText(f"Running... 0/{len(pairs)}")
        QtWidgets.QApplication.processEvents()

        successes = 0
        failures: list[str] = []
        last_success_bundle: dict[str, Any] | None = None

        try:
            for index, (pair_key, file_a, file_b) in enumerate(pairs, start=1):
                self.ui.valueCurrentFile.setText(f"{pair_key}\n{os.path.basename(file_a)}\n{os.path.basename(file_b)}")
                self.ui.valueDetectionStatus.setText(f"Running... {index}/{len(pairs)}")
                QtWidgets.QApplication.processEvents()

                try:
                    bundle = self._run_single_pair(pair_key, file_a, file_b, run_cfg)
                    last_success_bundle = bundle
                    successes += 1
                except Exception as pair_exc:
                    failures.append(f"{pair_key}: {pair_exc}")
                    continue
        finally:
            self._set_busy(False)

        if last_success_bundle is not None:
            self.current_result_bundle = last_success_bundle
            self._display_result_bundle(last_success_bundle)

        if failures and successes == 0:
            self.ui.valueDetectionStatus.setText("All failed")
            QMessageBox.critical(self, "Detection Failed", "\n".join(failures[:10]))
            return

        if failures:
            self.ui.valueDetectionStatus.setText(f"Finished with warnings: {successes} success / {len(failures)} failed")
            QMessageBox.warning(
                self,
                "Detection Finished with Warnings",
                f"Success: {successes}\nFailed: {len(failures)}\n\n" + "\n".join(failures[:10]),
            )
        else:
            self.ui.valueDetectionStatus.setText(f"Completed: {successes} pair(s)")

    def _run_single_pair(self, pair_key: str, file_a: str, file_b: str, run_cfg: dict[str, Any]) -> dict[str, Any]:
        signals = load_ab_signals(file_a, file_b)
        algo_func = run_cfg["callable"]
        params = dict(run_cfg["params"])

        result_a = algo_func(signals["x_a"], **params)
        result_b = algo_func(signals["x_b"], **params)

        dt_signed_s = float(result_b["t_head"] - result_a["t_head"])
        dt_us = abs(dt_signed_s) * 1e6
        distance_info = self._compute_fault_distance_info(
            run_cfg.get("sensor_distance_m"),
            run_cfg.get("wave_speed_mps"),
            result_a["t_head"],
            result_b["t_head"],
        )
        output_dir = run_cfg["output_dir"].strip()
        saved_paths: dict[str, str] = {}

        # 保存：全局图 + 局部图 + 两端 JSON + meta JSON
        if output_dir:
            save_dir = self._build_save_dir_with_dt(output_dir, pair_key, dt_us)
            os.makedirs(save_dir, exist_ok=True)
            saved_paths = self._save_pair_outputs(
                save_dir=save_dir,
                pair_key=pair_key,
                file_a=file_a,
                file_b=file_b,
                result_a=result_a,
                result_b=result_b,
                fs=run_cfg["sampling_freq_hz"],
                algorithm_label=run_cfg["algorithm_label"],
                general_config=run_cfg,
                distance_info=distance_info,
            )
            self.last_output_dir_used = save_dir

        return {
            "pair_key": pair_key,
            "file_a": file_a,
            "file_b": file_b,
            "result_a": result_a,
            "result_b": result_b,
            "dt_us": dt_us,
            "dt_signed_s": dt_signed_s,
            "distance_info": distance_info,
            "saved_paths": saved_paths,
            "run_config": run_cfg,
        }

    def _build_save_dir_with_dt(self, output_dir: str, pair_key: str, dt_us: float) -> str:
        folder_name = f"{pair_key}_{dt_us:.2f}us"
        return os.path.join(output_dir, folder_name)

    def _compute_fault_distance_info(self, sensor_distance_m: float | None, wave_speed_mps: float | None, t_head_a: float, t_head_b: float) -> dict[str, Any]:
        if sensor_distance_m is None or wave_speed_mps is None:
            return {
                "dt_signed_s": None,
                "distance_to_a_m": None,
                "distance_to_b_m": None,
            }

        dt_signed_s = float(t_head_b - t_head_a)
        distance_to_a_m = float((sensor_distance_m - wave_speed_mps * dt_signed_s) / 2.0)
        distance_to_b_m = float(sensor_distance_m - distance_to_a_m)
        return {
            "dt_signed_s": dt_signed_s,
            "distance_to_a_m": distance_to_a_m,
            "distance_to_b_m": distance_to_b_m,
        }

    @staticmethod
    def _format_distance_text(distance_m: float | None) -> str:
        if distance_m is None:
            return "-"
        return f"{distance_m:.3f} m"

    def _save_pair_outputs(
        self,
        *,
        save_dir: str,
        pair_key: str,
        file_a: str,
        file_b: str,
        result_a: dict[str, Any],
        result_b: dict[str, Any],
        fs: float,
        algorithm_label: str,
        general_config: dict[str, Any],
        distance_info: dict[str, Any],
    ) -> dict[str, str]:
        base_a = Path(file_a).stem
        base_b = Path(file_b).stem
        title_prefix = algorithm_label

        fig_global, _ = create_result_figure_ab(
            result_a,
            result_b,
            file_a=file_a,
            file_b=file_b,
            fs=fs,
            local_zoom=False,
            title_prefix=title_prefix,
        )
        global_path = os.path.join(save_dir, f"{base_a}__{base_b}_AB_global.png")
        save_figure(fig_global, global_path, dpi=200, close_fig=True)

        fig_local, _ = create_result_figure_ab(
            result_a,
            result_b,
            file_a=file_a,
            file_b=file_b,
            fs=fs,
            local_zoom=True,
            title_prefix=title_prefix,
        )
        local_path = os.path.join(save_dir, f"{base_a}__{base_b}_AB_local.png")
        save_figure(fig_local, local_path, dpi=200, close_fig=True)

        json_a_path = os.path.join(save_dir, f"{base_a}_summary.json")
        json_b_path = os.path.join(save_dir, f"{base_b}_summary.json")
        save_result_summary_json(result_a, json_a_path)
        save_result_summary_json(result_b, json_b_path)

        meta_path = os.path.join(save_dir, f"{pair_key}_run_meta.json")
        meta_payload = {
            "pair_key": pair_key,
            "file_a": file_a,
            "file_b": file_b,
            "algorithm_id": general_config["algorithm_id"],
            "algorithm_label": algorithm_label,
            "dt_us": abs(result_a["t_head"] - result_b["t_head"]) * 1e6,
            "dt_signed_s": distance_info.get("dt_signed_s"),
            "distance_to_a_m": distance_info.get("distance_to_a_m"),
            "distance_to_b_m": distance_info.get("distance_to_b_m"),
            "t_head_a_us": result_a["t_head"] * 1e6,
            "t_head_b_us": result_b["t_head"] * 1e6,
            "general_config": {
                "input_dir_a": general_config["input_dir_a"],
                "input_dir_b": general_config["input_dir_b"],
                "output_dir": general_config["output_dir"],
                "pair_mode": general_config["pair_mode"],
                "pair_keyword": general_config["pair_keyword"],
                "sensor_distance_m": general_config["sensor_distance_m"],
                "wave_speed_mps": general_config["wave_speed_mps"],
            },
            "algorithm_params": general_config["params"],
            "saved_paths": {
                "global_plot": global_path,
                "local_plot": local_path,
                "result_a_json": json_a_path,
                "result_b_json": json_b_path,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        return {
            "global_plot": global_path,
            "local_plot": local_path,
            "result_a_json": json_a_path,
            "result_b_json": json_b_path,
            "meta_json": meta_path,
        }

    # ------------------------------------------------------------------
    # 结果显示与手动保存
    # ------------------------------------------------------------------
    def _display_result_bundle(self, bundle: dict[str, Any]) -> None:
        result_a = bundle["result_a"]
        result_b = bundle["result_b"]
        run_cfg = bundle["run_config"]

        fig, _ = create_result_figure_ab(
            result_a,
            result_b,
            file_a=bundle["file_a"],
            file_b=bundle["file_b"],
            fs=run_cfg["sampling_freq_hz"],
            local_zoom=True,
            title_prefix=run_cfg["algorithm_label"],
        )
        self._set_plot_figure(fig)

        self.ui.valueCurrentFile.setText(
            f"{bundle['pair_key']}\n{os.path.basename(bundle['file_a'])}\n{os.path.basename(bundle['file_b'])}"
        )
        self.ui.valueCurrentAlgorithm.setText(run_cfg["algorithm_label"])
        distance_info = bundle.get("distance_info", {})
        self.ui.valueChannelAResult.setText(f"{result_a['t_head'] * 1e6:.3f} us")
        self.ui.valueChannelBResult.setText(f"{result_b['t_head'] * 1e6:.3f} us")
        self.ui.valueTimeDifference.setText(f"{bundle['dt_us']:.3f}")
        self.ui.valueDistanceToA.setText(self._format_distance_text(distance_info.get("distance_to_a_m")))
        self.ui.valueDistanceToB.setText(self._format_distance_text(distance_info.get("distance_to_b_m")))
        self.ui.valueDetectionStatus.setText("Completed")

    def save_current_result_figure(self) -> None:
        if self.current_figure is None or self.current_result_bundle is None:
            QMessageBox.information(self, "No Result", "There is no displayed figure to save.")
            return

        default_name = f"{self.current_result_bundle['pair_key']}_display.png"
        default_dir = self.last_output_dir_used or self.ui.editOutputDir.text().strip() or str(CURRENT_DIR)
        default_path = os.path.join(default_dir, default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Display Figure",
            default_path,
            "PNG Files (*.png);;JPG Files (*.jpg);;PDF Files (*.pdf)",
        )
        if not file_path:
            return

        try:
            self.current_figure.savefig(file_path, dpi=200, bbox_inches="tight")
            QMessageBox.information(self, "Saved", f"Figure saved to:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def clear_current_result(self) -> None:
        self.current_result_bundle = None
        self._clear_plot_area()
        self._update_result_labels_idle()

    def _update_result_labels_idle(self) -> None:
        self.ui.valueCurrentFile.setText("-")
        self.ui.valueCurrentAlgorithm.setText(ALGORITHM_DEFINITIONS[self.current_algorithm_id()]["label"])
        self.ui.valueChannelAResult.setText("-")
        self.ui.valueChannelBResult.setText("-")
        self.ui.valueTimeDifference.setText("-")
        self.ui.valueDistanceToA.setText("-")
        self.ui.valueDistanceToB.setText("-")
        self.ui.valueDetectionStatus.setText("Ready")

    # ------------------------------------------------------------------
    # Plot 容器管理
    # ------------------------------------------------------------------
    def _set_plot_figure(self, fig) -> None:
        self._clear_plot_area()

        self.current_figure = fig
        self.canvas = FigureCanvas(fig)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)

        layout = self.ui.framePlotContainer.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.ui.framePlotContainer)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def _clear_plot_area(self) -> None:
        layout = self.ui.framePlotContainer.layout()
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
        self.canvas = None
        self.nav_toolbar = None

        placeholder = QtWidgets.QLabel(
            "Plot area\n\nRun detection to display the A/B result figure here."
        )
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #666; border: 1px dashed #BBB; padding: 24px;")
        layout.addWidget(placeholder)
        self._plot_placeholder = placeholder

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    def _browse_dir_for(self, target_edit: QtWidgets.QLineEdit) -> None:
        start_dir = target_edit.text().strip() or str(CURRENT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if folder:
            target_edit.setText(folder)
            self._save_general_settings()

    def _set_busy(self, busy: bool) -> None:
        self.ui.btnRunDetection.setEnabled(not busy)
        self.ui.btnSaveResult.setEnabled(not busy)
        self.ui.btnClearResult.setEnabled(not busy)
        self.ui.btnBrowseInputDirA.setEnabled(not busy)
        self.ui.btnBrowseInputDirB.setEnabled(not busy)
        self.ui.btnBrowseOutputDir.setEnabled(not busy)
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    @staticmethod
    def _format_number(value: float) -> str:
        if math.isfinite(value):
            text = f"{value:.8f}".rstrip("0").rstrip(".")
            return text or "0"
        return str(value)

    def _set_widget_value(self, widget: QtWidgets.QWidget, value: Any) -> None:
        block = widget.blockSignals(True)
        try:
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(self._format_number(float(value)) if isinstance(value, (int, float)) else str(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                idx = widget.findData(value)
                if idx < 0:
                    idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
        finally:
            widget.blockSignals(block)

    def _read_widget_value(self, widget: QtWidgets.QWidget) -> Any:
        param_type = str(widget.property("paramType") or "str")
        if isinstance(widget, QtWidgets.QLineEdit):
            text = widget.text().strip()
            if param_type == "int":
                return int(float(text))
            if param_type == "float":
                return float(text)
            return text
        if isinstance(widget, QtWidgets.QComboBox):
            data = widget.currentData()
            if data is not None:
                return data
            text = widget.currentText().strip()
            if param_type == "int":
                return int(float(text))
            if param_type == "float":
                return float(text)
            return text
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        raise TypeError(f"Unsupported widget type: {type(widget)}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = WavefrontMainController()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
