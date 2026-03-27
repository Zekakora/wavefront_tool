from __future__ import annotations

from typing import Any

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"
except ImportError:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt5"

try:
    from .wavefront_param_schema import ALGORITHM_DEFINITIONS
    from .wavefront_param_store import ParameterStore
except ImportError:
    from wavefront_param_schema import ALGORITHM_DEFINITIONS
    from wavefront_param_store import ParameterStore


def _section_specs_for_algorithm(algorithm_id: str) -> list[dict[str, Any]]:
    if algorithm_id == "rdp_local_aic":
        return [
            {
                "title": "1. Basic / Data Settings",
                "description": "Sampling and baseline related parameters.",
                "keys": ["fs", "pre_n"],
            },
            {
                "title": "2. Wavelet Denoising",
                "description": "Pre-denoising options before rough localization and refinement.",
                "keys": [
                    "wavelet_enabled",
                    "wavelet",
                    "wavelet_level",
                    "wavelet_beta",
                    "wavelet_threshold_scale",
                    "wavelet_use_level_dependent_sigma",
                    "wavelet_mode",
                ],
            },
            {
                "title": "3. Rough Threshold Localization",
                "description": "Initial coarse trigger and smoothing used to narrow the event range.",
                "keys": ["rough_k", "rough_consecutive", "threshold_sg_window", "threshold_sg_poly"],
            },
            {
                "title": "4. RDP Simplification",
                "description": "RDP window, simplification strength, and segment screening parameters.",
                "keys": [
                    "rdp_left_us",
                    "rdp_right_us",
                    "rdp_epsilon",
                    "rdp_sg_window",
                    "rdp_sg_poly",
                    "rdp_seg_slope_k",
                    "rdp_seg_amp_k",
                    "rdp_weak_factor",
                ],
            },
            {
                "title": "5. Local Search / Trigger Detection",
                "description": "Search window, amplitude/slope thresholds, and tangent-fit related settings.",
                "keys": [
                    "search_left_us",
                    "search_right_us",
                    "noise_guard_us",
                    "noise_win_us",
                    "slope_win",
                    "slope_poly",
                    "amp_k",
                    "slope_k",
                    "min_consecutive",
                    "fit_n",
                    "polarity",
                ],
            },
            {
                "title": "6. AIC Refinement",
                "description": "Local AIC window and smoothing parameters for the final refined pick.",
                "keys": ["aic_left_us", "aic_right_us", "aic_min_split", "aic_smooth_win", "aic_smooth_poly"],
            },
        ]

    if algorithm_id == "rdp_global_iceemdan_teo":
        return [
            {
                "title": "1. Basic / Data Settings",
                "description": "Sampling and overall input related parameters.",
                "keys": ["fs"],
            },
            {
                "title": "2. RDP Rough Localization",
                "description": "Coarse event localization based on RDP preprocessing and threshold crossing.",
                "keys": [
                    "rdp_preprocess",
                    "rdp_sg_window",
                    "rdp_sg_polyorder",
                    "rdp_epsilon",
                    "rdp_pre_n",
                    "rdp_post_check_n",
                    "rdp_thr_method",
                    "rdp_k",
                    "rdp_use_abs",
                    "rdp_min_consecutive",
                ],
            },
            {
                "title": "3. Search Window / Global Preprocess",
                "description": "Search interval and preprocessing before ICEEMDAN decomposition.",
                "keys": ["search_left_us", "search_right_us", "noise_guard_us", "min_pre_noise_us", "global_preprocess"],
            },
            {
                "title": "4. ICEEMDAN Decomposition",
                "description": "IMF selection and ICEEMDAN decomposition hyperparameters.",
                "keys": [
                    "use_imf_mode",
                    "alpha2",
                    "ensemble_size",
                    "noise_strength",
                    "max_imfs",
                    "random_state",
                    "use_mirror_pad",
                    "mirror_pad_us",
                ],
            },
            {
                "title": "5. TEO Energy / Head Picking",
                "description": "TEO smoothing, thresholding, crossing rules, and wavehead pick strategy.",
                "keys": [
                    "pre_sg_window",
                    "sigma_k",
                    "min_peak_distance_samples",
                    "cross_consecutive",
                    "pick_mode",
                    "slope_polarity",
                    "slope_smooth_win",
                    "edge_ignore_samples",
                ],
            },
        ]

    return []


class ParameterEditorWidget(QWidget):
    valuesChanged = pyqtSignal(dict)

    def __init__(
        self,
        schema: list[dict[str, Any]],
        parent: QWidget | None = None,
        section_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.schema = schema
        self.fields: dict[str, Any] = {}
        self.section_specs = section_specs or []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        if self.section_specs:
            self._build_grouped_layout(root)
        else:
            layout = self._create_form_layout()
            root.addLayout(layout)
            for field in schema:
                widget = self._create_widget(field)
                self.fields[field["key"]] = widget
                layout.addRow(QLabel(field["label"]), widget)
            root.addStretch(1)

    def _create_form_layout(self) -> QFormLayout:
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow if QT_API == "PyQt6" else QFormLayout.ExpandingFieldsGrow
        )
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight if QT_API == "PyQt6" else Qt.AlignRight)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop if QT_API == "PyQt6" else Qt.AlignTop)
        layout.setSpacing(10)
        return layout

    def _build_grouped_layout(self, root: QVBoxLayout) -> None:
        field_map = {field["key"]: field for field in self.schema}
        added_keys: set[str] = set()

        for section in self.section_specs:
            section_keys = [key for key in section.get("keys", []) if key in field_map]
            if not section_keys:
                continue

            root.addWidget(self._create_section_title(section["title"]))
            description = section.get("description", "").strip()
            if description:
                root.addWidget(self._create_section_description(description))

            form = self._create_form_layout()
            root.addLayout(form)
            for key in section_keys:
                field = field_map[key]
                widget = self._create_widget(field)
                self.fields[key] = widget
                form.addRow(QLabel(field["label"]), widget)
                added_keys.add(key)

        remaining = [field for field in self.schema if field["key"] not in added_keys]
        if remaining:
            root.addWidget(self._create_section_title("Other Parameters"))
            root.addWidget(self._create_section_description("Parameters not explicitly assigned to a feature group."))
            form = self._create_form_layout()
            root.addLayout(form)
            for field in remaining:
                widget = self._create_widget(field)
                self.fields[field["key"]] = widget
                form.addRow(QLabel(field["label"]), widget)

        root.addStretch(1)

    def _create_section_title(self, text: str) -> QLabel:
        label = QLabel(f"<b>{text}</b><hr>")
        label.setWordWrap(True)
        return label

    def _create_section_description(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #666666; padding-left: 2px; padding-bottom: 4px;")
        return label

    def _create_widget(self, field: dict[str, Any]):
        field_type = field["type"]
        default = field["default"]

        if field_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
            widget.toggled.connect(lambda _=False: self.valuesChanged.emit(self.get_values()))
            return widget

        if field_type == "choice":
            widget = QComboBox()
            for choice in field.get("choices", []):
                widget.addItem(str(choice), choice)
            index = widget.findData(default)
            if index >= 0:
                widget.setCurrentIndex(index)
            widget.currentIndexChanged.connect(lambda _=0: self.valuesChanged.emit(self.get_values()))
            return widget

        if field_type == "int":
            widget = QSpinBox()
            widget.setRange(int(field.get("min", -2_147_483_648)), int(field.get("max", 2_147_483_647)))
            widget.setSingleStep(int(field.get("step", 1)))
            widget.setValue(int(default))
            widget.valueChanged.connect(lambda _=0: self.valuesChanged.emit(self.get_values()))
            return widget

        if field_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(field.get("min", -1e18)), float(field.get("max", 1e18)))
            widget.setDecimals(int(field.get("decimals", 6)))
            widget.setSingleStep(float(field.get("step", 0.1)))
            widget.setValue(float(default))
            widget.valueChanged.connect(lambda _=0.0: self.valuesChanged.emit(self.get_values()))
            return widget

        widget = QLineEdit(str(default))
        widget.editingFinished.connect(lambda: self.valuesChanged.emit(self.get_values()))
        return widget

    def set_values(self, values: dict[str, Any]) -> None:
        for field in self.schema:
            key = field["key"]
            if key not in values:
                continue
            widget = self.fields[key]
            value = values[key]
            field_type = field["type"]
            if field_type == "bool":
                widget.setChecked(bool(value))
            elif field_type == "choice":
                index = widget.findData(value)
                if index < 0:
                    index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif field_type == "int":
                widget.setValue(int(value))
            elif field_type == "float":
                widget.setValue(float(value))
            else:
                widget.setText(str(value))

    def get_values(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in self.schema:
            key = field["key"]
            widget = self.fields[key]
            field_type = field["type"]
            if field_type == "bool":
                result[key] = widget.isChecked()
            elif field_type == "choice":
                result[key] = widget.currentData()
            elif field_type == "int":
                result[key] = int(widget.value())
            elif field_type == "float":
                result[key] = float(widget.value())
            else:
                result[key] = widget.text().strip()
        return result


class FullParameterDialog(QDialog):
    paramsApplied = pyqtSignal(dict)

    def __init__(self, algorithm_id: str, store: ParameterStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.algorithm_id = algorithm_id
        self.store = store
        self.algorithm_info = ALGORITHM_DEFINITIONS[algorithm_id]
        self.setWindowTitle(self.algorithm_info["dialog_title"])
        self.resize(760, 720)

        root = QVBoxLayout(self)
        title = QLabel(
            f"<b>{self.algorithm_info['label']}</b><br>"
            "Default values are displayed at startup. You can modify them and they will be preserved."
        )
        title.setWordWrap(True)
        root.addWidget(title)

        toolbar = QHBoxLayout()
        self.btn_import = QPushButton("Import")
        self.btn_export = QPushButton("Export")
        self.btn_reset = QPushButton("Restore Defaults")
        toolbar.addWidget(self.btn_import)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_reset)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

        self.editor = ParameterEditorWidget(
            self.algorithm_info["schema"],
            section_specs=_section_specs_for_algorithm(self.algorithm_id),
        )
        self.editor.set_values(self.store.get_params(self.algorithm_id))
        self.scroll.setWidget(self.editor)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            if QT_API == "PyQt6"
            else QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        root.addWidget(self.button_box)

        self.btn_import.clicked.connect(self.import_json)
        self.btn_export.clicked.connect(self.export_json)
        self.btn_reset.clicked.connect(self.restore_defaults)
        self.button_box.accepted.connect(self.apply_and_accept)
        self.button_box.rejected.connect(self.reject)

    def restore_defaults(self) -> None:
        self.editor.set_values(self.store.algorithm_defaults(self.algorithm_id))

    def import_json(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Parameter JSON", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            imported = self.store.import_algorithm_json(self.algorithm_id, file_path, save=False)
            if "params" in imported and isinstance(imported["params"], dict):
                imported = imported["params"]
            self.editor.set_values(imported)
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))

    def export_json(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Parameter JSON",
            f"{self.algorithm_id}_params.json",
            "JSON Files (*.json)",
        )
        if not file_path:
            return
        try:
            self.store.update_params(self.algorithm_id, self.editor.get_values(), save=False)
            self.store.export_algorithm_json(self.algorithm_id, file_path)
            QMessageBox.information(self, "Export Succeeded", f"Saved to:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def apply_and_accept(self) -> None:
        values = self.editor.get_values()
        self.store.update_params(self.algorithm_id, values, save=True)
        self.paramsApplied.emit(values)
        self.accept()


__all__ = ["FullParameterDialog", "ParameterEditorWidget", "QT_API"]
