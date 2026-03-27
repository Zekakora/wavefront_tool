模块拆分说明
====================

文件结构
1. wavefront_data_io.py
   - 只负责 CSV 读取、A/B 双端读取、按文件名规则配对。

2. wavefront_algo_rdp_aic.py
   - 只负责 RDP + 前置改进小波降噪 + threshold/slope 触发 + 局部 AIC 的算法流程。
   - 主入口函数：detect_wavefront_rdp(x, **params)

3. wavefront_algo_iceemdan_teo.py
   - 只负责 RDP 粗定位 + 全局 ICEEMDAN + TEO 精定位。
   - 主入口函数：detect_wavefront_iceemdan_teo(x, **params)
     （它是 detect_wavefront_rdp_global_iceemdan_teo 的别名）

4. wavefront_plot_save.py
   - 只负责绘图和保存。
   - 可直接返回 matplotlib Figure，方便嵌入 PyQt 的 FigureCanvas。
   - 支持单端图和 A/B 双端图。

推荐 PyQt 调用方式
--------------------
单文件：
    from wavefront_data_io import load_csv_no_header
    from wavefront_algo_rdp_aic import detect_wavefront_rdp
    from wavefront_plot_save import create_result_figure_single

    _, x, _ = load_csv_no_header(csv_path)
    result = detect_wavefront_rdp(x, fs=4.2e6, wavelet_enabled=True)
    fig, axes = create_result_figure_single(result, end_label="A端", local_zoom=True)

双文件：
    from wavefront_data_io import load_ab_signals
    from wavefront_algo_iceemdan_teo import detect_wavefront_iceemdan_teo
    from wavefront_plot_save import create_result_figure_ab

    data = load_ab_signals(file_a, file_b)
    result_a = detect_wavefront_iceemdan_teo(data["x_a"], fs=4.2e6)
    result_b = detect_wavefront_iceemdan_teo(data["x_b"], fs=4.2e6)
    fig, axes = create_result_figure_ab(result_a, result_b, local_zoom=True)

保存图和结果摘要
--------------------
    from wavefront_plot_save import save_figure, save_result_summary_json

    save_figure(fig, "fig/result.png", close_fig=False)
    save_result_summary_json(result, "fig/result_summary.json")

依赖说明
--------------------
RDP + AIC 模块：
    numpy, scipy, matplotlib
    若开启 wavelet_enabled=True，还需要 pywt

ICEEMDAN-TEO 模块：
    numpy, scipy, matplotlib, PyEMD, fastrdp

备注
--------------------
在我重构时，顺手修正了 ICEEMDAN-TEO 原脚本里一个很可能的索引问题：
    use_imf_mode="iimf1" 时，原脚本实际返回的是 imfs[1]。
现在已经改成真正的 IMF1，即 imfs[0]。
