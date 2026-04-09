from __future__ import annotations

import os
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from wavefront_main_controller_modified import CURRENT_DIR, WavefrontMainController
from wavefront_plot_save import WavefrontInteractiveViewer, export_plot_widget_image


class WavefrontMainControllerPG(WavefrontMainController):
    """PyQtGraph-based controller.

    Keeps the original UI / parameter / algorithm wiring from WavefrontMainController,
    but replaces the embedded matplotlib canvas with an interactive pyqtgraph viewer.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.plot_viewer: WavefrontInteractiveViewer | None = None
        super().__init__(parent)

    def _prepare_plot_area(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.ui.framePlotContainer)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.plot_viewer = WavefrontInteractiveViewer(self.ui.framePlotContainer)
        layout.addWidget(self.plot_viewer, 1)

    def _display_result_bundle(self, bundle: dict) -> None:
        result_a = bundle["result_a"]
        result_b = bundle["result_b"]
        run_cfg = bundle["run_config"]

        if self.plot_viewer is not None:
            self.plot_viewer.set_results(
                result_a,
                result_b,
                file_a=bundle["file_a"],
                file_b=bundle["file_b"],
                fs=run_cfg["sampling_freq_hz"],
                title_prefix=run_cfg["algorithm_label"],
            )

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
        if self.current_result_bundle is None or self.plot_viewer is None:
            QMessageBox.information(self, "No Result", "There is no displayed interactive plot to save.")
            return

        default_name = f"{self.current_result_bundle['pair_key']}_interactive.png"
        default_dir = self.last_output_dir_used or self.ui.editOutputDir.text().strip() or str(CURRENT_DIR)
        default_path = os.path.join(default_dir, default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Interactive Plot",
            default_path,
            "PNG Files (*.png);;JPG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;SVG Files (*.svg)",
        )
        if not file_path:
            return

        try:
            export_plot_widget_image(self.plot_viewer.plot_widget, file_path, width=1800)
            QMessageBox.information(self, "Saved", f"Interactive plot saved to:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def clear_current_result(self) -> None:
        self.current_result_bundle = None
        if self.plot_viewer is not None:
            self.plot_viewer.clear_results()
        self._update_result_labels_idle()

    def _clear_plot_area(self) -> None:
        if self.plot_viewer is not None:
            self.plot_viewer.clear_results()

    def _set_plot_figure(self, fig) -> None:
        # Retained only for compatibility with any inherited calls.
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = WavefrontMainControllerPG()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
