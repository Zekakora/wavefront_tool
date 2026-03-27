新增的 PyQt 参数界面文件：

1. wavefront_param_schema.py
   - 定义两个算法的默认参数、完整参数 schema、主窗口关键参数 schema。

2. wavefront_param_store.py
   - 参数持久化层。
   - 默认保存到：用户主目录 /.wavefront_locator/user_params.json
   - 用户修改关键参数页或完整参数表后，都会自动保存。

3. wavefront_param_dialog.py
   - 完整参数表 QDialog。
   - 支持：Import / Export / Restore Defaults / OK / Cancel。
   - OK 时写回持久化存储。

4. wavefront_pyqt_main.py
   - 主窗口示例。
   - 通过下拉框选择算法。
   - 选择后 QStackedWidget 自动切换到对应的关键参数页。
   - 每个关键参数页都带“Open Full Parameter Table”按钮。

当前已经满足的交互要求：
- 主窗口用下拉框切换算法。
- QStackedWidget 自动显示该算法对应的关键参数区域。
- 每个算法都有独立完整参数表。
- 参数表支持导入、导出、恢复默认值。
- 参数默认值会显示出来。
- 用户修改后会持久化保存，下次打开仍保持上次修改值。

如何运行：
- 确保安装 PyQt5 或 PyQt6。
- 将这些文件与 wavefront_algo_rdp_aic.py、wavefront_algo_iceemdan_teo.py 放在同一目录。
- 运行：python wavefront_pyqt_main.py

如何在你自己的主界面中接入：
- 可以直接复用 ParameterStore、FullParameterDialog、AlgorithmParamPage 三层。
- 若你已经有自己的 MainWindow，只需要：
  1) 建一个算法选择下拉框
  2) 建一个 QStackedWidget
  3) 对每个算法放一个 AlgorithmParamPage
  4) 在运行按钮里调用 collect_run_config() 或 store.get_params(algo_id)

算法调用方式：
- RDP + Local AIC:
    detect_wavefront_rdp(x, **params)
- RDP + Global ICEEMDAN-TEO:
    detect_wavefront_rdp_global_iceemdan_teo(x, **params)
