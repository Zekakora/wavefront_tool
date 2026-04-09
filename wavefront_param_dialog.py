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
        QFrame,
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
        QFrame,
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
    from .wavefront_param_translations import PARAM_LABELS_ZH, PARAM_TOOLTIPS_ZH, CHOICE_TEXT_ZH
except ImportError:
    from wavefront_param_schema import ALGORITHM_DEFINITIONS
    from wavefront_param_store import ParameterStore
    from wavefront_param_translations import PARAM_LABELS_ZH, PARAM_TOOLTIPS_ZH, CHOICE_TEXT_ZH


def _algorithm_label_zh(algorithm_id: str, original: str) -> str:
    mapping = {
        "rdp_local_aic": "RDP + 局部 AIC 参数表",
        "rdp_global_iceemdan_teo": "RDP + 全局 ICEEMDAN-TEO 参数表",
    }
    return mapping.get(algorithm_id, original)


def _section_specs_for_algorithm(algorithm_id: str) -> list[dict[str, Any]]:
    if algorithm_id == "rdp_local_aic":
        return [
            {
                "title": "1. 基础与数据设置",
                "description": "采样频率、基线估计等基础参数。",
                "keys": ["fs", "pre_n"],
            },
            {
                "title": "2. 小波降噪",
                "description": "粗定位与精细定位前的小波预降噪配置。",
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
                "title": "3. 粗阈值定位",
                "description": "用于缩小候选区域的初始触发与平滑参数。",
                "keys": ["rough_k", "rough_consecutive", "threshold_sg_window", "threshold_sg_poly"],
            },
            {
                "title": "4. RDP 化简",
                "description": "RDP 窗口、化简强度与候选分段筛选参数。",
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
                "title": "5. 局部搜索与触发检测",
                "description": "搜索窗、幅值/斜率阈值以及切线拟合相关参数。",
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
                "title": "6. AIC 精修",
                "description": "最终波头精修时使用的局部 AIC 窗口与平滑参数。",
                "keys": ["aic_left_us", "aic_right_us", "aic_min_split", "aic_smooth_win", "aic_smooth_poly"],
            },
        ]

    if algorithm_id == "rdp_global_iceemdan_teo":
        return [
            {
                "title": "1. 基础与数据设置",
                "description": "采样频率等整体输入相关参数。",
                "keys": ["fs"],
            },
            {
                "title": "2. RDP 粗定位",
                "description": "基于 RDP 预处理与阈值越界规则的粗定位参数。",
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
                "title": "3. 搜索窗与全局预处理",
                "description": "ICEEMDAN 分解前的搜索范围与整体预处理配置。",
                "keys": ["search_left_us", "search_right_us", "noise_guard_us", "min_pre_noise_us",
                         "global_preprocess"],
            },
            {
                "title": "4. ICEEMDAN 分解",
                "description": "IMF 选取方式与 ICEEMDAN 分解超参数。",
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
                "title": "5. TEO 能量与波头拾取",
                "description": "TEO 平滑、阈值规则、越阈判定与波头拾取方式。",
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
        self.field_map = {field["key"]: field for field in self.schema}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        if self.section_specs:
            self._build_grouped_layout(root)
        else:
            card = self._create_section_card("参数设置", "可在此修改当前算法的全部参数。")
            form = self._create_form_layout()
            card.layout().addLayout(form)
            for field in schema:
                widget = self._create_widget(field)
                self.fields[field["key"]] = widget
                form.addRow(self._create_label(field), widget)
            root.addWidget(card)
            root.addStretch(1)

    def _create_form_layout(self) -> QFormLayout:
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            if QT_API == "PyQt6"
            else QFormLayout.AllNonFixedFieldsGrow
        )
        if QT_API == "PyQt6":
            layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        else:
            layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
            layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        return layout

    def _create_section_card(self, title: str, description: str = "") -> QFrame:
        card = QFrame(self)
        card.setProperty("card", "true")
        card.setProperty("sectionCard", "true")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitleLabel")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setObjectName("sectionDescriptionLabel")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        return card

    def _build_grouped_layout(self, root: QVBoxLayout) -> None:
        added_keys: set[str] = set()

        for section in self.section_specs:
            section_keys = [key for key in section.get("keys", []) if key in self.field_map]
            if not section_keys:
                continue

            card = self._create_section_card(section["title"], section.get("description", "").strip())
            form = self._create_form_layout()
            card.layout().addLayout(form)

            for key in section_keys:
                field = self.field_map[key]
                widget = self._create_widget(field)
                self.fields[key] = widget
                form.addRow(self._create_label(field), widget)
                added_keys.add(key)

            root.addWidget(card)

        remaining = [field for field in self.schema if field["key"] not in added_keys]
        if remaining:
            card = self._create_section_card("其他参数", "未显式分组的参数会显示在这里。")
            form = self._create_form_layout()
            card.layout().addLayout(form)
            for field in remaining:
                widget = self._create_widget(field)
                self.fields[field["key"]] = widget
                form.addRow(self._create_label(field), widget)
            root.addWidget(card)

        root.addStretch(1)

    def _translated_label_text(self, field: dict[str, Any]) -> str:
        return PARAM_LABELS_ZH.get(field["key"], field.get("label", field["key"]))

    def _create_label(self, field: dict[str, Any]) -> QLabel:
        label = QLabel(self._translated_label_text(field))
        label.setWordWrap(True)

        key = field["key"]
        original = field.get("label", "")
        desc = PARAM_TOOLTIPS_ZH.get(key, "")

        tooltip_lines = []
        if desc:
            tooltip_lines.append(f"💡 作用：{desc}\n")
            cursor_shape = Qt.CursorShape.WhatsThisCursor if QT_API == "PyQt6" else Qt.WhatsThisCursor
            label.setCursor(cursor_shape)

        tooltip_lines.append(f"参数键：{key}")
        if original and original != label.text():
            tooltip_lines.append(f"原始英文：{original}")

        label.setToolTip("\n".join(tooltip_lines))
        return label

    def _choice_display_text(self, choice: Any) -> str:
        return CHOICE_TEXT_ZH.get(str(choice), str(choice))

    def _create_widget(self, field: dict[str, Any]):
        field_type = field["type"]
        default = field["default"]

        if field_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
            widget.setText("启用")
            widget.toggled.connect(lambda _=False: self.valuesChanged.emit(self.get_values()))
            return widget

        if field_type == "choice":
            widget = QComboBox()
            for choice in field.get("choices", []):
                widget.addItem(self._choice_display_text(choice), choice)
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
            if key not in values or key not in self.fields:
                continue
            widget = self.fields[key]
            value = values[key]
            field_type = field["type"]
            if field_type == "bool":
                widget.setChecked(bool(value))
            elif field_type == "choice":
                index = widget.findData(value)
                if index < 0:
                    index = widget.findText(self._choice_display_text(value))
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
        self.setWindowTitle(_algorithm_label_zh(self.algorithm_id, self.algorithm_info["dialog_title"]))
        self.resize(880, 760)
        self.setMinimumSize(720, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.header_card = QFrame(self)
        self.header_card.setProperty("card", "true")
        header_layout = QVBoxLayout(self.header_card)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(6)

        self.title_label = QLabel(_algorithm_label_zh(self.algorithm_id, self.algorithm_info["label"]))
        self.title_label.setObjectName("dialogTitleLabel")
        self.title_label.setWordWrap(True)
        header_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("启动时会显示当前默认参数。你可以修改、导入、导出或恢复默认值，保存后会自动持久化。")
        self.subtitle_label.setObjectName("dialogSubtitleLabel")
        self.subtitle_label.setWordWrap(True)
        header_layout.addWidget(self.subtitle_label)
        root.addWidget(self.header_card)

        self.toolbar_card = QFrame(self)
        self.toolbar_card.setProperty("card", "true")
        toolbar_wrap = QVBoxLayout(self.toolbar_card)
        toolbar_wrap.setContentsMargins(12, 12, 12, 12)
        toolbar_wrap.setSpacing(0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        self.btn_import = QPushButton("导入参数")
        self.btn_export = QPushButton("导出参数")
        self.btn_reset = QPushButton("恢复默认")
        self.btn_import.setToolTip("从 JSON 文件导入当前算法参数")
        self.btn_export.setToolTip("将当前算法参数导出到 JSON 文件")
        self.btn_reset.setToolTip("恢复当前算法的默认参数")
        toolbar.addWidget(self.btn_import)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_reset)
        toolbar.addStretch(1)
        toolbar_wrap.addLayout(toolbar)
        root.addWidget(self.toolbar_card)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame if QT_API == "PyQt6" else QFrame.NoFrame)
        root.addWidget(self.scroll, 1)

        self.editor = ParameterEditorWidget(
            self.algorithm_info["schema"],
            section_specs=_section_specs_for_algorithm(self.algorithm_id),
        )
        self.editor.set_values(self.store.get_params(self.algorithm_id))
        self.scroll.setWidget(self.editor)

        self.button_card = QFrame(self)
        self.button_card.setProperty("card", "true")
        button_layout = QVBoxLayout(self.button_card)
        button_layout.setContentsMargins(12, 12, 12, 12)
        button_layout.setSpacing(0)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            if QT_API == "PyQt6"
            else QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        ok_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Ok if QT_API == "PyQt6" else QDialogButtonBox.Ok
        )
        cancel_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel if QT_API == "PyQt6" else QDialogButtonBox.Cancel
        )
        if ok_button is not None:
            ok_button.setText("保存并应用")
            ok_button.setObjectName("primaryActionButton")
        if cancel_button is not None:
            cancel_button.setText("取消")
        button_layout.addWidget(self.button_box)
        root.addWidget(self.button_card)

        self.btn_import.clicked.connect(self.import_json)
        self.btn_export.clicked.connect(self.export_json)
        self.btn_reset.clicked.connect(self.restore_defaults)
        self.button_box.accepted.connect(self.apply_and_accept)
        self.button_box.rejected.connect(self.reject)

        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: #f2f2f2;
                color: #1c1c1c;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }
            QLabel {
                background: transparent;
                color: #1c1c1c;
            }
            QFrame[card="true"] {
                background: #ffffff;
                border: 1px solid #dadada;
                border-radius: 12px;
            }
            QLabel#dialogTitleLabel {
                font-size: 16px;
                font-weight: 700;
                color: #111111;
            }
            QLabel#dialogSubtitleLabel {
                color: #666666;
                font-size: 12px;
            }
            QLabel#sectionTitleLabel {
                font-size: 14px;
                font-weight: 700;
                color: #111111;
            }
            QLabel#sectionDescriptionLabel {
                color: #666666;
                font-size: 12px;
                padding-bottom: 2px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                padding: 8px 10px;
                min-height: 20px;
                selection-background-color: #1c1c1c;
                selection-color: #ffffff;
            }
            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #a9a9a9;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #111111;
                background: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QPushButton {
                background: #f5f5f5;
                border: 1px solid #d5d5d5;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
                color: #1c1c1c;
                min-height: 20px;
            }
            QPushButton:hover {
                background: #ebebeb;
                border-color: #bfbfbf;
            }
            QPushButton:pressed {
                background: #e1e1e1;
            }
            QPushButton#primaryActionButton {
                background: #111111;
                color: #ffffff;
                border: 1px solid #111111;
                font-weight: 700;
                min-width: 110px;
            }
            QPushButton#primaryActionButton:hover {
                background: #000000;
                border-color: #000000;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #bfbfbf;
                background: #ffffff;
            }
            QCheckBox::indicator:hover {
                border-color: #8f8f8f;
            }
            QCheckBox::indicator:checked {
                background: #111111;
                border-color: #111111;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #bcbcbc;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9f9f9f;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0px;
            }
            """
        )

    def restore_defaults(self) -> None:
        self.editor.set_values(self.store.algorithm_defaults(self.algorithm_id))
        QMessageBox.information(self, "已恢复默认参数", "当前算法参数已恢复为默认值。")

    def import_json(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "导入参数 JSON", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            imported = self.store.import_algorithm_json(self.algorithm_id, file_path, save=False)
            if "params" in imported and isinstance(imported["params"], dict):
                imported = imported["params"]
            self.editor.set_values(imported)
            QMessageBox.information(self, "导入成功", "参数已成功导入到当前对话框。")
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def export_json(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出参数 JSON",
            f"{self.algorithm_id}_params.json",
            "JSON Files (*.json)",
        )
        if not file_path:
            return
        try:
            self.store.update_params(self.algorithm_id, self.editor.get_values(), save=False)
            self.store.export_algorithm_json(self.algorithm_id, file_path)
            QMessageBox.information(self, "导出成功", f"参数文件已保存到：\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def apply_and_accept(self) -> None:
        values = self.editor.get_values()
        self.store.update_params(self.algorithm_id, values, save=True)
        self.paramsApplied.emit(values)
        self.accept()


__all__ = ["FullParameterDialog", "ParameterEditorWidget", "QT_API"]