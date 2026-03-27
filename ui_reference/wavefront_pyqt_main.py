from __future__ import annotations

import sys
from typing import Any, Callable

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt5"

try:
    from .wavefront_algo_iceemdan_teo import detect_wavefront_rdp_global_iceemdan_teo
    from .wavefront_algo_rdp_aic import detect_wavefront_rdp
    from .wavefront_param_dialog import FullParameterDialog, ParameterEditorWidget
    from .wavefront_param_schema import ALGORITHM_DEFINITIONS
    from .wavefront_param_store import ParameterStore
except ImportError:
    from wavefront_algo_iceemdan_teo import detect_wavefront_rdp_global_iceemdan_teo
    from wavefront_algo_rdp_aic import detect_wavefront_rdp
    from wavefront_param_dialog import FullParameterDialog, ParameterEditorWidget
    from wavefront_param_schema import ALGORITHM_DEFINITIONS
    from wavefront_param_store import ParameterStore


ALGORITHM_CALLABLES: dict[str, Callable[..., dict[str, Any]]] = {
    "rdp_local_aic": detect_wavefront_rdp,
    "rdp_global_iceemdan_teo": detect_wavefront_rdp_global_iceemdan_teo,
}


class AlgorithmParamPage(QWidget):
    def __init__(self, algorithm_id: str, store: ParameterStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.algorithm_id = algorithm_id
        self.store = store
        self.info = ALGORITHM_DEFINITIONS[algorithm_id]

        layout = QVBoxLayout(self)
        group = QGroupBox("Key Parameters")
        group_layout = QVBoxLayout(group)

        key_schema = [
            field for field in self.info["schema"]
            if field["key"] in self.info["key_fields"]
        ]
        self.editor = ParameterEditorWidget(key_schema)
        self.editor.set_values(self.store.get_params(self.algorithm_id))
        group_layout.addWidget(self.editor)

        btn_row = QHBoxLayout()
        self.btn_full = QPushButton("Open Full Parameter Table")
        btn_row.addWidget(self.btn_full)
        btn_row.addStretch(1)
        group_layout.addLayout(btn_row)

        layout.addWidget(group)
        layout.addStretch(1)

        self.editor.valuesChanged.connect(self._save_key_values)
        self.btn_full.clicked.connect(self.open_full_dialog)

    def _save_key_values(self, values: dict[str, Any]) -> None:
        self.store.update_params(self.algorithm_id, values, save=True)

    def refresh_from_store(self) -> None:
        self.editor.set_values(self.store.get_params(self.algorithm_id))

    def open_full_dialog(self) -> None:
        dialog = FullParameterDialog(self.algorithm_id, self.store, self)
        dialog.paramsApplied.connect(lambda _values: self.refresh_from_store())
        dialog.exec() if QT_API == "PyQt6" else dialog.exec_()


class WavefrontMainWindow(QMainWindow):
    def __init__(self, store: ParameterStore | None = None) -> None:
        super().__init__()
        self.store = store or ParameterStore()
        self.setWindowTitle("Wavefront Locator - Parameter UI")
        self.resize(980, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        top_card = QFrame()
        top_card.setFrameShape(QFrame.Shape.StyledPanel if QT_API == "PyQt6" else QFrame.StyledPanel)
        top_layout = QGridLayout(top_card)
        top_layout.addWidget(QLabel("Algorithm"), 0, 0)
        self.algorithm_combo = QComboBox()
        for algorithm_id, info in ALGORITHM_DEFINITIONS.items():
            self.algorithm_combo.addItem(info["label"], algorithm_id)
        top_layout.addWidget(self.algorithm_combo, 0, 1)
        top_layout.setColumnStretch(1, 1)
        root.addWidget(top_card)

        self.param_stack = QStackedWidget()
        self.pages: dict[str, AlgorithmParamPage] = {}
        for algorithm_id in ALGORITHM_DEFINITIONS:
            page = AlgorithmParamPage(algorithm_id, self.store, self)
            self.pages[algorithm_id] = page
            self.param_stack.addWidget(page)
        root.addWidget(self.param_stack, 1)

        # summary_box = QGroupBox("Current Parameter Snapshot")
        # summary_layout = QVBoxLayout(summary_box)
        # self.summary_text = QTextEdit()
        # self.summary_text.setReadOnly(True)
        # summary_layout.addWidget(self.summary_text)
        # root.addWidget(summary_box, 1)

        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        # for page in self.pages.values():
        #     page.editor.valuesChanged.connect(lambda _values: self.refresh_summary())

        self.on_algorithm_changed(0)

    def current_algorithm_id(self) -> str:
        return str(self.algorithm_combo.currentData())

    def current_params(self) -> dict[str, Any]:
        return self.store.get_params(self.current_algorithm_id())

    def current_algorithm_callable(self) -> Callable[..., dict[str, Any]]:
        return ALGORITHM_CALLABLES[self.current_algorithm_id()]

    def on_algorithm_changed(self, index: int) -> None:
        self.param_stack.setCurrentIndex(index)
        # self.refresh_summary()

    # def refresh_summary(self) -> None:
    #     algorithm_id = self.current_algorithm_id()
    #     params = self.store.get_params(algorithm_id)
    #     label = ALGORITHM_DEFINITIONS[algorithm_id]["label"]
    #     lines = [f"Algorithm: {label}", "", "Current parameters:"]
    #     for key, value in params.items():
    #         lines.append(f"- {key} = {value}")
    #     self.summary_text.setPlainText("\n".join(lines))

    def collect_run_config(self) -> dict[str, Any]:
        algorithm_id = self.current_algorithm_id()
        return {
            "algorithm_id": algorithm_id,
            "algorithm_label": ALGORITHM_DEFINITIONS[algorithm_id]["label"],
            "params": self.store.get_params(algorithm_id),
            "callable": ALGORITHM_CALLABLES[algorithm_id],
        }

    def show_current_config(self) -> None:
        config = self.collect_run_config()
        QMessageBox.information(
            self,
            "Current Run Config",
            f"{config['algorithm_label']}\n\nParameter count: {len(config['params'])}",
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = WavefrontMainWindow()
    window.show()
    return app.exec() if QT_API == "PyQt6" else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
