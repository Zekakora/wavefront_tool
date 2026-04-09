from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy import signal

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


COMMON_WAVELETS = [
    "db2", "db4", "db6", "sym4", "sym8", "coif3", "bior3.5", "haar",
]
COMMON_MODES = ["symmetric", "periodization", "zero", "constant", "reflect"]


@dataclass
class SignalState:
    file_path: str = ""
    df: pd.DataFrame | None = None
    raw_signal: np.ndarray | None = None
    analysis_signal: np.ndarray | None = None
    coeffs: list[np.ndarray] | None = None
    component_signals: dict[str, np.ndarray] = field(default_factory=dict)
    denoised_signal: np.ndarray | None = None
    reconstructed_signal: np.ndarray | None = None
    sigma: float | None = None
    threshold: float | None = None
    level: int | None = None


class MplFigureCard(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def clear_with_message(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#666666", fontsize=11)
        self.canvas.draw_idle()


class WaveletTransformPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.signal_states: dict[str, SignalState] = {
            "A": SignalState(),
            "B": SignalState(),
        }
        self.current_status: str = "Ready"
        self.last_used_level: int | None = None

        self._build_ui()
        self._wire_signals()
        self._reset_views()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        root_layout.addWidget(splitter)

        title_font = self.font()
        title_font.setBold(True)
        title_font.setPointSize(11)

        # Left
        self.leftScrollArea = QtWidgets.QScrollArea(splitter)
        self.leftScrollArea.setWidgetResizable(True)
        self.leftScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.leftScrollArea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.leftScrollContents = QtWidgets.QWidget()
        self.leftScrollArea.setWidget(self.leftScrollContents)
        left_layout = QtWidgets.QVBoxLayout(self.leftScrollContents)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(12)

        self.inputCard = QtWidgets.QFrame(self.leftScrollContents)
        self.inputCard.setObjectName("inputCard")
        input_layout = QtWidgets.QVBoxLayout(self.inputCard)
        input_layout.setContentsMargins(16, 16, 16, 16)
        input_layout.setSpacing(12)

        self.labelInputTitle = QtWidgets.QLabel("小波变换对比分析", self.inputCard)
        self.labelInputTitle.setObjectName("labelInputSectionTitle")
        self.labelInputTitle.setFont(title_font)
        input_layout.addWidget(self.labelInputTitle)

        self.labelInputTip = QtWidgets.QLabel(
            "支持同时导入 A/B 两个 CSV，使用同一组小波参数进行分解、去噪与对比分析。",
            self.inputCard,
        )
        self.labelInputTip.setWordWrap(True)
        input_layout.addWidget(self.labelInputTip)

        self.formInput = QtWidgets.QFormLayout()
        self.formInput.setHorizontalSpacing(12)
        self.formInput.setVerticalSpacing(10)

        file_a_row = QtWidgets.QHBoxLayout()
        file_a_row.setSpacing(8)
        self.editCsvPathA = QtWidgets.QLineEdit(self.inputCard)
        self.editCsvPathA.setPlaceholderText("请选择 A 文件 CSV")
        self.btnBrowseCsvA = QtWidgets.QToolButton(self.inputCard)
        self.btnBrowseCsvA.setText("...")
        file_a_row.addWidget(self.editCsvPathA, 1)
        file_a_row.addWidget(self.btnBrowseCsvA)
        self.formInput.addRow("CSV A", file_a_row)

        file_b_row = QtWidgets.QHBoxLayout()
        file_b_row.setSpacing(8)
        self.editCsvPathB = QtWidgets.QLineEdit(self.inputCard)
        self.editCsvPathB.setPlaceholderText("请选择 B 文件 CSV")
        self.btnBrowseCsvB = QtWidgets.QToolButton(self.inputCard)
        self.btnBrowseCsvB.setText("...")
        file_b_row.addWidget(self.editCsvPathB, 1)
        file_b_row.addWidget(self.btnBrowseCsvB)
        self.formInput.addRow("CSV B", file_b_row)

        self.spinSignalColumn = QtWidgets.QSpinBox(self.inputCard)
        self.spinSignalColumn.setRange(0, 999)
        self.spinSignalColumn.setValue(1)
        self.spinSignalColumn.setToolTip("按 pandas 的列索引读取信号列，默认第 2 列")
        self.formInput.addRow("信号列索引", self.spinSignalColumn)

        self.editFsHz = QtWidgets.QLineEdit(self.inputCard)
        self.editFsHz.setText("4200000")
        self.editFsHz.setPlaceholderText("仅用于时间轴显示")
        self.formInput.addRow("采样率 (Hz)", self.editFsHz)

        self.comboWavelet = QtWidgets.QComboBox(self.inputCard)
        self.comboWavelet.addItems(COMMON_WAVELETS)
        self.comboWavelet.setCurrentText("db4")
        self.formInput.addRow("小波基", self.comboWavelet)

        self.spinLevel = QtWidgets.QSpinBox(self.inputCard)
        self.spinLevel.setRange(1, 12)
        self.spinLevel.setValue(4)
        self.formInput.addRow("分解层数", self.spinLevel)

        self.comboMode = QtWidgets.QComboBox(self.inputCard)
        self.comboMode.addItems(COMMON_MODES)
        self.comboMode.setCurrentText("symmetric")
        self.formInput.addRow("边界模式", self.comboMode)

        self.comboThresholdMode = QtWidgets.QComboBox(self.inputCard)
        self.comboThresholdMode.addItems(["soft", "hard"])
        self.formInput.addRow("阈值方式", self.comboThresholdMode)

        self.spinThresholdScale = QtWidgets.QDoubleSpinBox(self.inputCard)
        self.spinThresholdScale.setRange(0.10, 20.0)
        self.spinThresholdScale.setDecimals(2)
        self.spinThresholdScale.setSingleStep(0.10)
        self.spinThresholdScale.setValue(1.00)
        self.formInput.addRow("阈值倍率", self.spinThresholdScale)

        self.checkDetrend = QtWidgets.QCheckBox("去趋势", self.inputCard)
        self.checkDetrend.setChecked(True)
        self.formInput.addRow("预处理", self.checkDetrend)

        self.checkNormalize = QtWidgets.QCheckBox("归一化", self.inputCard)
        self.checkNormalize.setChecked(False)
        self.formInput.addRow("", self.checkNormalize)

        input_layout.addLayout(self.formInput)

        self.buttonRow1 = QtWidgets.QHBoxLayout()
        self.buttonRow1.setSpacing(8)
        self.btnLoadSignals = QtWidgets.QPushButton("加载对比数据", self.inputCard)
        self.btnDecompose = QtWidgets.QPushButton("执行分解", self.inputCard)
        self.buttonRow1.addWidget(self.btnLoadSignals)
        self.buttonRow1.addWidget(self.btnDecompose)
        input_layout.addLayout(self.buttonRow1)

        self.buttonRow2 = QtWidgets.QHBoxLayout()
        self.buttonRow2.setSpacing(8)
        self.btnDenoise = QtWidgets.QPushButton("执行去噪", self.inputCard)
        self.btnReconstructSelected = QtWidgets.QPushButton("重构所选分量", self.inputCard)
        self.buttonRow2.addWidget(self.btnDenoise)
        self.buttonRow2.addWidget(self.btnReconstructSelected)
        input_layout.addLayout(self.buttonRow2)

        self.btnExport = QtWidgets.QPushButton("导出对比结果 CSV", self.inputCard)
        input_layout.addWidget(self.btnExport)

        left_layout.addWidget(self.inputCard)

        self.componentCard = QtWidgets.QFrame(self.leftScrollContents)
        self.componentCard.setObjectName("inputCard")
        comp_layout = QtWidgets.QVBoxLayout(self.componentCard)
        comp_layout.setContentsMargins(16, 16, 16, 16)
        comp_layout.setSpacing(10)

        self.labelComponents = QtWidgets.QLabel("分量选择", self.componentCard)
        self.labelComponents.setObjectName("labelAlgorithmSectionTitle")
        self.labelComponents.setFont(title_font)
        comp_layout.addWidget(self.labelComponents)

        self.listComponents = QtWidgets.QListWidget(self.componentCard)
        self.listComponents.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.listComponents.setMinimumHeight(220)
        comp_layout.addWidget(self.listComponents)

        self.labelComponentTip = QtWidgets.QLabel(
            "勾选 A / D 分量后，可同时重构两个文件中对应分量并进行对比。",
            self.componentCard,
        )
        self.labelComponentTip.setWordWrap(True)
        comp_layout.addWidget(self.labelComponentTip)

        left_layout.addWidget(self.componentCard)
        left_layout.addStretch(1)

        # Center
        self.centerPanel = QtWidgets.QWidget(splitter)
        center_layout = QtWidgets.QVBoxLayout(self.centerPanel)
        center_layout.setContentsMargins(0, 0, 10, 0)
        center_layout.setSpacing(12)

        self.plotCard = QtWidgets.QFrame(self.centerPanel)
        self.plotCard.setObjectName("plotCard")
        plot_layout = QtWidgets.QVBoxLayout(self.plotCard)
        plot_layout.setContentsMargins(16, 16, 16, 16)
        plot_layout.setSpacing(10)

        self.labelPlotTitle = QtWidgets.QLabel("A/B 对比图形区", self.plotCard)
        self.labelPlotTitle.setObjectName("labelResultSectionTitle")
        self.labelPlotTitle.setFont(title_font)
        plot_layout.addWidget(self.labelPlotTitle)

        self.tabPlots = QtWidgets.QTabWidget(self.plotCard)
        plot_layout.addWidget(self.tabPlots, 1)

        self.tabTime = QtWidgets.QWidget()
        self.tabReconstruct = QtWidgets.QWidget()
        self.tabCoeff = QtWidgets.QWidget()
        self.tabEnergy = QtWidgets.QWidget()
        self.tabPlots.addTab(self.tabTime, "时域对比")
        self.tabPlots.addTab(self.tabReconstruct, "重构结果")
        self.tabPlots.addTab(self.tabCoeff, "分解系数")
        self.tabPlots.addTab(self.tabEnergy, "能量对比")

        time_layout = QtWidgets.QVBoxLayout(self.tabTime)
        reconstruct_layout = QtWidgets.QVBoxLayout(self.tabReconstruct)
        coeff_layout = QtWidgets.QVBoxLayout(self.tabCoeff)
        energy_layout = QtWidgets.QVBoxLayout(self.tabEnergy)
        time_layout.setContentsMargins(0, 0, 0, 0)
        reconstruct_layout.setContentsMargins(0, 0, 0, 0)
        coeff_layout.setContentsMargins(0, 0, 0, 0)
        energy_layout.setContentsMargins(0, 0, 0, 0)

        self.timeFigureCard = MplFigureCard(self.tabTime)
        self.reconstructFigureCard = MplFigureCard(self.tabReconstruct)
        self.coeffFigureCard = MplFigureCard(self.tabCoeff)
        self.energyFigureCard = MplFigureCard(self.tabEnergy)
        time_layout.addWidget(self.timeFigureCard)
        reconstruct_layout.addWidget(self.reconstructFigureCard)
        coeff_layout.addWidget(self.coeffFigureCard)
        energy_layout.addWidget(self.energyFigureCard)

        center_layout.addWidget(self.plotCard, 1)

        # Right
        self.rightScrollArea = QtWidgets.QScrollArea(splitter)
        self.rightScrollArea.setWidgetResizable(True)
        self.rightScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.rightScrollArea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.rightScrollContents = QtWidgets.QWidget()
        self.rightScrollArea.setWidget(self.rightScrollContents)
        right_layout = QtWidgets.QVBoxLayout(self.rightScrollContents)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.resultCard = QtWidgets.QFrame(self.rightScrollContents)
        self.resultCard.setObjectName("resultCard")
        result_layout = QtWidgets.QVBoxLayout(self.resultCard)
        result_layout.setContentsMargins(16, 16, 16, 16)
        result_layout.setSpacing(10)

        self.labelResultTitle = QtWidgets.QLabel("结果摘要", self.resultCard)
        self.labelResultTitle.setObjectName("labelResultSectionTitle")
        self.labelResultTitle.setFont(title_font)
        result_layout.addWidget(self.labelResultTitle)

        self.formResult = QtWidgets.QFormLayout()
        self.formResult.setHorizontalSpacing(12)
        self.formResult.setVerticalSpacing(10)
        self.formResult.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.valueFileA = self._make_value_label(self.resultCard)
        self.valueFileB = self._make_value_label(self.resultCard)
        self.valueSamplesA = self._make_value_label(self.resultCard)
        self.valueSamplesB = self._make_value_label(self.resultCard)
        self.valueFs = self._make_value_label(self.resultCard)
        self.valueDurationA = self._make_value_label(self.resultCard)
        self.valueDurationB = self._make_value_label(self.resultCard)
        self.valueWavelet = self._make_value_label(self.resultCard)
        self.valueLevel = self._make_value_label(self.resultCard)
        self.valueSigmaA = self._make_value_label(self.resultCard)
        self.valueSigmaB = self._make_value_label(self.resultCard)
        self.valueThresholdA = self._make_value_label(self.resultCard)
        self.valueThresholdB = self._make_value_label(self.resultCard)
        self.valueStatus = self._make_value_label(self.resultCard)

        self.formResult.addRow("文件 A", self.valueFileA)
        self.formResult.addRow("文件 B", self.valueFileB)
        self.formResult.addRow("A 样本数", self.valueSamplesA)
        self.formResult.addRow("B 样本数", self.valueSamplesB)
        self.formResult.addRow("采样率", self.valueFs)
        self.formResult.addRow("A 时长", self.valueDurationA)
        self.formResult.addRow("B 时长", self.valueDurationB)
        self.formResult.addRow("小波基", self.valueWavelet)
        self.formResult.addRow("分解层数", self.valueLevel)
        self.formResult.addRow("A 噪声 σ", self.valueSigmaA)
        self.formResult.addRow("B 噪声 σ", self.valueSigmaB)
        self.formResult.addRow("A 阈值", self.valueThresholdA)
        self.formResult.addRow("B 阈值", self.valueThresholdB)
        self.formResult.addRow("状态", self.valueStatus)
        result_layout.addLayout(self.formResult)
        right_layout.addWidget(self.resultCard)

        self.energyCard = QtWidgets.QFrame(self.rightScrollContents)
        self.energyCard.setObjectName("resultCard")
        energy_card_layout = QtWidgets.QVBoxLayout(self.energyCard)
        energy_card_layout.setContentsMargins(16, 16, 16, 16)
        energy_card_layout.setSpacing(10)

        self.labelEnergyTitle = QtWidgets.QLabel("各层能量摘要", self.energyCard)
        self.labelEnergyTitle.setObjectName("labelAlgorithmSectionTitle")
        self.labelEnergyTitle.setFont(title_font)
        energy_card_layout.addWidget(self.labelEnergyTitle)

        self.textEnergySummary = QtWidgets.QPlainTextEdit(self.energyCard)
        self.textEnergySummary.setReadOnly(True)
        self.textEnergySummary.setMinimumHeight(240)
        energy_card_layout.addWidget(self.textEnergySummary)
        right_layout.addWidget(self.energyCard)
        right_layout.addStretch(1)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 700, 320])

    def _make_value_label(self, parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel("-", parent)
        label.setWordWrap(True)
        label.setObjectName("valueDetectionStatus")
        return label

    def _wire_signals(self) -> None:
        self.btnBrowseCsvA.clicked.connect(lambda: self._browse_csv_for(self.editCsvPathA))
        self.btnBrowseCsvB.clicked.connect(lambda: self._browse_csv_for(self.editCsvPathB))
        self.btnLoadSignals.clicked.connect(self.load_signals)
        self.btnDecompose.clicked.connect(self.run_decomposition)
        self.btnDenoise.clicked.connect(self.run_denoise)
        self.btnReconstructSelected.clicked.connect(self.reconstruct_selected_components)
        self.btnExport.clicked.connect(self.export_current_result_csv)

    # ------------------------------------------------------------------
    # Data / processing
    # ------------------------------------------------------------------
    def _browse_csv_for(self, target_edit: QtWidgets.QLineEdit) -> None:
        start_dir = str(Path(target_edit.text().strip()).parent) if target_edit.text().strip() else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", start_dir, "CSV Files (*.csv)")
        if file_path:
            target_edit.setText(file_path)

    def _sampling_hz(self) -> float:
        text = self.editFsHz.text().strip() or "4200000"
        hz = float(text)
        if hz <= 0:
            raise ValueError("采样率必须大于 0")
        return hz

    def _preprocess_signal(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        if self.checkDetrend.isChecked():
            y = signal.detrend(y, type="constant")
        if self.checkNormalize.isChecked():
            scale = np.max(np.abs(y)) + 1e-12
            y = y / scale
        return y

    def _axis_x(self, n: int) -> tuple[np.ndarray, str]:
        try:
            hz = self._sampling_hz()
            return np.arange(n, dtype=float) / hz * 1e6, "Time (us)"
        except Exception:
            return np.arange(n, dtype=float), "Sample index"

    @staticmethod
    def _pad_array(arr: np.ndarray | None, length: int) -> np.ndarray:
        if arr is None:
            return np.full(length, np.nan, dtype=float)
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= length:
            return arr[:length]
        out = np.full(length, np.nan, dtype=float)
        out[: len(arr)] = arr
        return out

    def _loaded_keys(self) -> list[str]:
        return [key for key, state in self.signal_states.items() if state.raw_signal is not None]

    def _reset_state(self, key: str) -> None:
        self.signal_states[key] = SignalState()

    def _load_single_signal(self, file_path: str) -> tuple[pd.DataFrame, np.ndarray]:
        df = pd.read_csv(file_path, header=None)
        if df.empty:
            raise ValueError(f"文件为空: {file_path}")

        col = int(self.spinSignalColumn.value())
        if col < 0 or col >= df.shape[1]:
            raise ValueError(f"信号列索引超出范围，当前文件共有 {df.shape[1]} 列")

        series = pd.to_numeric(df.iloc[:, col], errors="coerce")
        series = series.interpolate(limit_direction="both").fillna(0.0)
        x = series.to_numpy(dtype=float)
        if x.size < 8:
            raise ValueError("信号长度过短，至少需要 8 个采样点")
        return df, x

    def load_signals(self) -> None:
        try:
            file_map = {
                "A": self.editCsvPathA.text().strip(),
                "B": self.editCsvPathB.text().strip(),
            }
            if not file_map["A"] and not file_map["B"]:
                raise ValueError("请至少选择一个 CSV 文件，建议同时选择 A/B 两个文件进行对比")

            loaded_count = 0
            for key, file_path in file_map.items():
                if not file_path:
                    self._reset_state(key)
                    continue

                df, raw = self._load_single_signal(file_path)
                state = SignalState(
                    file_path=file_path,
                    df=df,
                    raw_signal=raw,
                    analysis_signal=self._preprocess_signal(raw),
                )
                self.signal_states[key] = state
                loaded_count += 1

            self.last_used_level = None
            self.listComponents.clear()
            status = "已加载 A/B 对比数据" if loaded_count >= 2 else "已加载单个文件"
            self.current_status = status
            self._update_summary(status=status)
            self._refresh_all_plots()
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", str(exc))

    def _ensure_signals_ready(self) -> dict[str, SignalState]:
        loaded = self._loaded_keys()
        if not loaded:
            raise ValueError("请先加载数据")
        ready: dict[str, SignalState] = {}
        for key in loaded:
            state = self.signal_states[key]
            if state.raw_signal is None:
                continue
            state.analysis_signal = self._preprocess_signal(state.raw_signal)
            ready[key] = state
        return ready

    def _current_wavelet_config(self, states: dict[str, SignalState]) -> tuple[str, str, int]:
        wavelet = self.comboWavelet.currentText().strip()
        mode = self.comboMode.currentText().strip()
        requested_level = int(self.spinLevel.value())

        max_levels = []
        for state in states.values():
            if state.analysis_signal is None:
                continue
            max_level = pywt.dwt_max_level(len(state.analysis_signal), pywt.Wavelet(wavelet).dec_len)
            max_levels.append(max(max_level, 1))

        if not max_levels:
            raise ValueError("当前没有可分解的数据")

        common_max_level = min(max_levels)
        level = max(1, min(requested_level, common_max_level))
        if level != requested_level:
            self.spinLevel.setValue(level)
        return wavelet, mode, level

    @staticmethod
    def _estimate_noise_sigma_from_coeffs(coeffs: list[np.ndarray] | None) -> float:
        if not coeffs or len(coeffs) < 2:
            return 0.0
        detail = np.asarray(coeffs[-1], dtype=float)
        med = np.median(detail)
        mad = np.median(np.abs(detail - med)) + 1e-12
        return float(mad / 0.6745)

    def run_decomposition(self) -> None:
        try:
            states = self._ensure_signals_ready()
            wavelet, mode, level = self._current_wavelet_config(states)

            for state in states.values():
                x = np.asarray(state.analysis_signal, dtype=float)
                state.coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode, level=level)
                state.component_signals = self._build_component_signals(state.coeffs, wavelet, mode, len(x), level)
                state.denoised_signal = None
                state.reconstructed_signal = None
                state.sigma = self._estimate_noise_sigma_from_coeffs(state.coeffs)
                state.threshold = None
                state.level = level

            self.last_used_level = level
            self._populate_component_list(level)
            self.current_status = "分解完成"
            self._update_summary(status="分解完成")
            self._refresh_all_plots()
        except Exception as exc:
            QMessageBox.warning(self, "分解失败", str(exc))

    def run_denoise(self) -> None:
        try:
            states = self._ensure_signals_ready()
            if any(state.coeffs is None for state in states.values()):
                self.run_decomposition()
                states = self._ensure_signals_ready()

            wavelet, mode, level = self._current_wavelet_config(states)
            thr_mode = self.comboThresholdMode.currentText().strip()
            scale = float(self.spinThresholdScale.value())

            for state in states.values():
                if state.coeffs is None or state.analysis_signal is None:
                    continue
                sigma = self._estimate_noise_sigma_from_coeffs(state.coeffs)
                threshold = scale * sigma * np.sqrt(2.0 * np.log(max(len(state.analysis_signal), 2)))
                new_coeffs: list[np.ndarray] = [state.coeffs[0]]
                for detail in state.coeffs[1:]:
                    new_coeffs.append(pywt.threshold(detail, threshold, mode=thr_mode))
                denoised = pywt.waverec(new_coeffs, wavelet=wavelet, mode=mode)
                state.denoised_signal = np.asarray(denoised[: len(state.analysis_signal)], dtype=float)
                state.sigma = sigma
                state.threshold = float(threshold)
                state.level = level

            self.last_used_level = level
            self.current_status = "去噪完成"
            self._update_summary(status="去噪完成")
            self._refresh_all_plots()
        except Exception as exc:
            QMessageBox.warning(self, "去噪失败", str(exc))

    def reconstruct_selected_components(self) -> None:
        try:
            states = self._ensure_signals_ready()
            if any(state.coeffs is None for state in states.values()):
                raise ValueError("请先执行小波分解")

            checked_names = self._checked_component_names()
            if not checked_names:
                raise ValueError("请先勾选至少一个分量")

            wavelet, mode, level = self._current_wavelet_config(states)
            name_to_index = self._component_name_to_index(level)

            for state in states.values():
                if state.coeffs is None or state.analysis_signal is None:
                    continue
                selected = [np.zeros_like(c) for c in state.coeffs]
                for name in checked_names:
                    if name in name_to_index:
                        idx = name_to_index[name]
                        selected[idx] = state.coeffs[idx]
                rec = pywt.waverec(selected, wavelet=wavelet, mode=mode)
                state.reconstructed_signal = np.asarray(rec[: len(state.analysis_signal)], dtype=float)
                state.level = level

            self.last_used_level = level
            self.current_status = f"已重构 {', '.join(checked_names)}"
            self._update_summary(status=self.current_status)
            self._refresh_all_plots()
        except Exception as exc:
            QMessageBox.warning(self, "重构失败", str(exc))

    def export_current_result_csv(self) -> None:
        try:
            loaded = self._loaded_keys()
            if not loaded:
                raise ValueError("当前没有可导出的数据")

            base_dir = str(Path(self.signal_states[loaded[0]].file_path).parent) if self.signal_states[loaded[0]].file_path else str(Path.home())
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出小波对比结果 CSV",
                str(Path(base_dir) / "wavelet_compare_result.csv"),
                "CSV Files (*.csv)",
            )
            if not file_path:
                return

            max_len = max(len(self.signal_states[key].raw_signal) for key in loaded if self.signal_states[key].raw_signal is not None)
            data: dict[str, np.ndarray] = {"sample_index": np.arange(max_len, dtype=float)}
            try:
                data["time_us"] = np.arange(max_len, dtype=float) / self._sampling_hz() * 1e6
            except Exception:
                pass

            for key in loaded:
                state = self.signal_states[key]
                data[f"raw_{key}"] = self._pad_array(state.raw_signal, max_len)
                data[f"analysis_{key}"] = self._pad_array(state.analysis_signal, max_len)
                if state.denoised_signal is not None:
                    data[f"denoised_{key}"] = self._pad_array(state.denoised_signal, max_len)
                if state.reconstructed_signal is not None:
                    data[f"reconstructed_{key}"] = self._pad_array(state.reconstructed_signal, max_len)

            pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "导出成功", f"结果已保存到:\n{file_path}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _build_component_signals(
        self,
        coeffs: list[np.ndarray],
        wavelet: str,
        mode: str,
        signal_len: int,
        level: int,
    ) -> dict[str, np.ndarray]:
        component_signals: dict[str, np.ndarray] = {}
        names = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
        for coeff_index, name in enumerate(names):
            selected = [np.zeros_like(c) for c in coeffs]
            selected[coeff_index] = coeffs[coeff_index]
            rec = pywt.waverec(selected, wavelet=wavelet, mode=mode)
            component_signals[name] = np.asarray(rec[:signal_len], dtype=float)
        return component_signals

    def _component_name_to_index(self, level: int) -> dict[str, int]:
        names = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
        return {name: index for index, name in enumerate(names)}

    def _populate_component_list(self, level: int) -> None:
        self.listComponents.clear()
        names = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
        for row, name in enumerate(names):
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if row == 0 else QtCore.Qt.CheckState.Unchecked)
            self.listComponents.addItem(item)

    def _checked_component_names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.listComponents.count()):
            item = self.listComponents.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                names.append(item.text())
        return names

    # ------------------------------------------------------------------
    # Summary / plots
    # ------------------------------------------------------------------
    def _format_duration(self, n: int | None) -> str:
        if not n:
            return "-"
        try:
            return f"{n / self._sampling_hz() * 1e6:.3f} us"
        except Exception:
            return f"{n} samples"

    def _update_summary(self, status: str = "") -> None:
        state_a = self.signal_states["A"]
        state_b = self.signal_states["B"]

        self.valueFileA.setText(state_a.file_path or "-")
        self.valueFileB.setText(state_b.file_path or "-")
        self.valueSamplesA.setText(str(len(state_a.raw_signal)) if state_a.raw_signal is not None else "-")
        self.valueSamplesB.setText(str(len(state_b.raw_signal)) if state_b.raw_signal is not None else "-")

        try:
            self.valueFs.setText(f"{self._sampling_hz():.3f} Hz")
        except Exception:
            self.valueFs.setText("-")

        self.valueDurationA.setText(self._format_duration(len(state_a.raw_signal) if state_a.raw_signal is not None else None))
        self.valueDurationB.setText(self._format_duration(len(state_b.raw_signal) if state_b.raw_signal is not None else None))
        self.valueWavelet.setText(self.comboWavelet.currentText())
        self.valueLevel.setText(str(self.last_used_level or self.spinLevel.value()))
        self.valueSigmaA.setText("-" if state_a.sigma is None else f"{state_a.sigma:.6e}")
        self.valueSigmaB.setText("-" if state_b.sigma is None else f"{state_b.sigma:.6e}")
        self.valueThresholdA.setText("-" if state_a.threshold is None else f"{state_a.threshold:.6e}")
        self.valueThresholdB.setText("-" if state_b.threshold is None else f"{state_b.threshold:.6e}")
        self.valueStatus.setText(status or self.current_status or "Ready")
        self.textEnergySummary.setPlainText(self._energy_summary_text())

    def _energy_summary_text(self) -> str:
        loaded = self._loaded_keys()
        if not loaded:
            return "尚未加载数据。"

        chunks: list[str] = []
        for key in loaded:
            state = self.signal_states[key]
            if not state.coeffs:
                chunks.append(f"{key}: 尚未执行分解")
                continue
            level = state.level or max(len(state.coeffs) - 1, 1)
            names = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
            energies = [float(np.sum(np.square(np.asarray(c, dtype=float)))) for c in state.coeffs]
            total = sum(energies) + 1e-12
            lines = [f"{key} 文件:"]
            for name, energy in zip(names, energies):
                lines.append(f"  {name}: {energy / total * 100.0:.2f}%")
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks)

    def _refresh_all_plots(self) -> None:
        self._plot_time_domain()
        self._plot_reconstructed_signals()
        self._plot_coefficients()
        self._plot_energy_distribution()

    def _plot_time_domain(self) -> None:
        loaded = self._loaded_keys()
        if not loaded:
            self.timeFigureCard.clear_with_message("请先加载 A/B CSV 数据")
            return

        fig = self.timeFigureCard.figure
        fig.clear()
        axes = fig.subplots(len(loaded), 1, squeeze=False)
        axes = axes.flatten()

        for ax_raw, key in zip(axes, loaded):
            state = self.signal_states[key]
            assert state.raw_signal is not None
            x_axis, x_label = self._axis_x(len(state.raw_signal))
            ax_proc = ax_raw.twinx()

            raw_line = ax_raw.plot(
                x_axis,
                state.raw_signal,
                label=f"{key} Raw",
                linewidth=1.0,
                alpha=0.75,
            )
            proc_lines = []
            if state.analysis_signal is not None and not np.array_equal(state.analysis_signal, state.raw_signal):
                proc_lines += ax_proc.plot(
                    x_axis,
                    state.analysis_signal,
                    label=f"{key} Preprocessed",
                    linewidth=1.0,
                    alpha=0.85,
                )
            if state.denoised_signal is not None:
                proc_lines += ax_proc.plot(
                    x_axis,
                    state.denoised_signal,
                    label=f"{key} Denoised",
                    linewidth=1.15,
                )
            if state.reconstructed_signal is not None:
                proc_lines += ax_proc.plot(
                    x_axis,
                    state.reconstructed_signal,
                    label=f"{key} Reconstructed",
                    linewidth=1.15,
                )

            ax_raw.set_title(f"{key} 文件时域对比（原始 / 处理后双坐标轴）")
            ax_raw.set_xlabel(x_label)
            ax_raw.set_ylabel("Raw amplitude")
            ax_proc.set_ylabel("Processed amplitude")
            ax_raw.grid(alpha=0.25)

            handles = raw_line + proc_lines
            if handles:
                labels = [line.get_label() for line in handles]
                ax_raw.legend(handles, labels, loc="best")

        self.timeFigureCard.canvas.draw_idle()

    def _plot_reconstructed_signals(self) -> None:
        loaded = self._loaded_keys()
        states_with_reconstruction = [key for key in loaded if self.signal_states[key].reconstructed_signal is not None]
        if not loaded:
            self.reconstructFigureCard.clear_with_message("请先加载 A/B CSV 数据")
            return
        if not states_with_reconstruction:
            self.reconstructFigureCard.clear_with_message("请先勾选分量并执行重构")
            return

        fig = self.reconstructFigureCard.figure
        fig.clear()
        axes = fig.subplots(len(loaded), 1, squeeze=False)
        axes = axes.flatten()

        for ax, key in zip(axes, loaded):
            state = self.signal_states[key]
            assert state.raw_signal is not None
            x_axis, x_label = self._axis_x(len(state.raw_signal))

            ax.plot(x_axis, state.raw_signal, label=f"{key} Raw", linewidth=0.9, alpha=0.35)
            if state.reconstructed_signal is not None:
                ax.plot(x_axis, state.reconstructed_signal, label=f"{key} Reconstructed", linewidth=1.25)
            if state.denoised_signal is not None:
                ax.plot(x_axis, state.denoised_signal, label=f"{key} Denoised", linewidth=1.0, alpha=0.85)

            checked_names = self._checked_component_names()
            suffix = f"（分量: {', '.join(checked_names)}）" if checked_names else ""
            ax.set_title(f"{key} 文件重构结果{suffix}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")

        self.reconstructFigureCard.canvas.draw_idle()

    def _plot_coefficients(self) -> None:
        loaded = [key for key in self._loaded_keys() if self.signal_states[key].coeffs]
        if not loaded:
            self.coeffFigureCard.clear_with_message("请先执行小波分解")
            return

        fig = self.coeffFigureCard.figure
        fig.clear()

        states = [self.signal_states[key] for key in loaded]
        nrows = max(len(state.coeffs or []) for state in states)
        ncols = len(loaded)
        axes = fig.subplots(nrows, ncols, squeeze=False)

        for col, key in enumerate(loaded):
            state = self.signal_states[key]
            coeffs = state.coeffs or []
            level = state.level or max(len(coeffs) - 1, 1)
            names = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
            for row in range(nrows):
                ax = axes[row, col]
                if row < len(coeffs):
                    coeff = np.asarray(coeffs[row], dtype=float)
                    name = names[row] if row < len(names) else f"C{row}"
                    ax.plot(np.arange(len(coeff)), coeff, linewidth=0.9)
                    ax.set_ylabel(name)
                    ax.grid(alpha=0.20)
                    if row == 0:
                        ax.set_title(f"{key} 文件系数")
                    if row == nrows - 1:
                        ax.set_xlabel("Coefficient index")
                else:
                    ax.axis("off")

        self.coeffFigureCard.canvas.draw_idle()

    def _plot_energy_distribution(self) -> None:
        loaded = [key for key in self._loaded_keys() if self.signal_states[key].coeffs]
        if not loaded:
            self.energyFigureCard.clear_with_message("请先执行小波分解")
            return

        fig = self.energyFigureCard.figure
        fig.clear()
        ax = fig.add_subplot(111)

        level = self.last_used_level or self.spinLevel.value()
        names = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
        x = np.arange(len(names), dtype=float)

        if len(loaded) == 1:
            key = loaded[0]
            coeffs = self.signal_states[key].coeffs or []
            energies = np.array([float(np.sum(np.square(np.asarray(c, dtype=float)))) for c in coeffs], dtype=float)
            ratios = energies / (np.sum(energies) + 1e-12) * 100.0
            ax.bar(names[: len(ratios)], ratios)
        else:
            width = 0.36
            for offset, key in [(-width / 2, loaded[0]), (width / 2, loaded[1])]:
                coeffs = self.signal_states[key].coeffs or []
                energies = np.array([float(np.sum(np.square(np.asarray(c, dtype=float)))) for c in coeffs], dtype=float)
                ratios = energies / (np.sum(energies) + 1e-12) * 100.0
                ax.bar(x[: len(ratios)] + offset, ratios, width=width, label=f"{key} 文件")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.legend(loc="best")

        ax.set_title("A/B 各层能量占比对比")
        ax.set_ylabel("Energy ratio (%)")
        ax.grid(axis="y", alpha=0.25)
        self.energyFigureCard.canvas.draw_idle()

    def _reset_views(self) -> None:
        self.current_status = "Ready"
        self._update_summary(status="Ready")
        self.timeFigureCard.clear_with_message("请先加载 A/B CSV 数据")
        self.reconstructFigureCard.clear_with_message("请先勾选分量并执行重构")
        self.coeffFigureCard.clear_with_message("请先执行小波分解")
        self.energyFigureCard.clear_with_message("请先执行小波分解")
