"""
IdeaSearch Streamlit App - Tab切换版本
支持绘图拟合和文件上传拟合两种模式
"""

import sys
import streamlit as st
import numpy as np
from pathlib import Path
import time

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.components.canvas import render_drawing_canvas, process_canvas_data, canvas_to_base64
from src.components.config import render_sidebar_config, load_default_config
from src.components.results import (
    render_progress_section, 
    render_fitting_comparison,
    render_pareto_frontier,
    render_score_history,
    render_api_calls_log
)
from src.core.fitting import FittingEngine
import plotly.graph_objects as go

# 页面配置
default_config = load_default_config()
st.set_page_config(
    page_title=default_config['app']['title'],
    page_icon=default_config['app']['page_icon'],
    layout=default_config['app']['layout'],
    initial_sidebar_state=default_config['app']['initial_sidebar_state']
)


def init_session_state():
    """初始化 session state"""
    if 'canvas_data' not in st.session_state:
        st.session_state.canvas_data = None
    if 'npz_data' not in st.session_state:
        st.session_state.npz_data = None
    if 'fitting_engine' not in st.session_state:
        st.session_state.fitting_engine = None
    if 'fitting_running' not in st.session_state:
        st.session_state.fitting_running = False


def tab_canvas_fitting():
    """Tab 1: 画布绘图拟合"""
    st.markdown("## 🎨 绘制曲线拟合")
    
    col_left, col_right = st.columns([1, 1])
    
    # 左侧：绘图画布
    with col_left:
        canvas_config = default_config['canvas']
        canvas_result = render_drawing_canvas(canvas_config)
        
        if canvas_result.image_data is not None:
            config = st.session_state.config
            curve_data = process_canvas_data(
                canvas_result,
                width=canvas_config['width'],
                height=canvas_config['height'],
                smooth=config['smooth_data'],
                interpolate=config['interpolate_data'],
                num_points=config['num_points']
            )
            
            if curve_data is not None:
                x, y = curve_data
                st.session_state.canvas_data = (x, y)
                st.success(f"✅ 已提取 {len(x)} 个数据点")
            else:
                st.info("请在画布上绘制曲线")
    
    # 右侧：数据预览
    with col_right:
        st.markdown("### 📊 数据预览")
        
        if st.session_state.canvas_data is not None:
            x, y = st.session_state.canvas_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("X 范围", f"[{x.min():.2f}, {x.max():.2f}]")
            with col2:
                st.metric("Y 范围", f"[{y.min():.2f}, {y.max():.2f}]")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=6, color='#1f77b4')))
            fig.update_layout(title="原始数据点", xaxis_title="x", yaxis_title="y", height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("等待绘制数据...")
    
    st.markdown("---")
    
    # 拟合控制
    st.markdown("### 🚀 拟合控制")
    
    config = st.session_state.config
    can_start = (
        st.session_state.canvas_data is not None and
        len(config['models']) > 0 and
        len(config['functions']) > 0 and
        not st.session_state.fitting_running
    )
    
    if st.button("▶️ 开始拟合", disabled=not can_start, use_container_width=True):
        x, y = st.session_state.canvas_data
        
        # 创建实时更新的UI占位符
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()
        
        log_placeholder = st.empty()
        
        # 创建拟合引擎
        engine = FittingEngine(config)
        canvas_image = canvas_to_base64(canvas_result) if canvas_result.image_data is not None else None
        
        # 准备数据
        x_2d = x.reshape(-1, 1)
        
        # 初始化fitter和searcher
        status_placeholder.info("🔧 初始化拟合引擎...")
        engine.initialize_fitter(x_2d, y, yerr=None)
        engine.initialize_searcher(canvas_image)
        
        # 设置状态
        engine.is_running = True
        engine.start_time = time.time()
        
        status_placeholder.success("✅ 初始化完成，开始拟合...")
        
        # 主循环
        logs = []
        total_cycles = config['cycle_num']
        unit_epochs = config['unit_interaction_num']
        
        for cycle in range(total_cycles):
            engine.current_cycle = cycle + 1
            
            # 动态设置 filter_func
            if cycle % 3 == 0:
                engine.ideasearcher.set_filter_func(lambda idea: idea)
            else:
                engine.ideasearcher.set_filter_func(lambda idea: "")
            
            # 重新填充岛屿
            if cycle != 0:
                engine.ideasearcher.repopulate_islands()
            
            status_placeholder.info(f"🔄 循环 {engine.current_cycle}/{total_cycles} 进行中...")
            
            # 运行 epochs
            for epoch in range(unit_epochs):
                # 运行一个 epoch
                engine.ideasearcher.run(1)
                
                # 立即获取结果
                try:
                    engine.best_expression = engine.fitter.get_best_fit()
                    engine.best_score = engine.ideasearcher.get_best_score()
                except:
                    engine.best_expression = ""
                    engine.best_score = 0.0
                
                # 更新进度条
                progress = (epoch + 1) / unit_epochs
                progress_placeholder.progress(progress, text=f"循环 {engine.current_cycle}/{total_cycles} - Epoch {epoch + 1}/{unit_epochs}")
                
                # 更新日志
                log_msg = f"🔹 Cycle {engine.current_cycle} · Epoch {epoch + 1}/{unit_epochs} · 分数: {engine.best_score:.4f}"
                if engine.best_expression:
                    log_msg += f" · 表达式: {engine.best_expression[:50]}{'...' if len(engine.best_expression) > 50 else ''}"
                logs.append(log_msg)
                log_placeholder.text_area("📝 实时日志", "\n".join(logs[-20:]), height=200)
                
                # 实时更新图表
                if engine.best_expression and engine.best_score > 0:
                    try:
                        x_plot = np.linspace(x.min(), x.max(), 300)
                        y_plot = engine.evaluate_expression(engine.best_expression, x_plot)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='原始数据', marker=dict(size=8)))
                        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='拟合曲线', line=dict(width=3, color='red')))
                        fig.update_layout(title=f"实时拟合结果 (分数: {engine.best_score:.4f})", height=400)
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # 显示指标
                        with metrics_placeholder.container():
                            st.metric("当前分数", f"{engine.best_score:.4f}")
                            st.metric("循环进度", f"{engine.current_cycle}/{total_cycles}")
                            st.metric("运行时间", f"{time.time() - engine.start_time:.1f}s")
                            st.code(engine.best_expression, language='python')
                    except Exception as e:
                        pass
                
                # 检查是否达到目标
                if engine.best_score >= config['shutdown_score']:
                    status_placeholder.success(f"🎯 达到目标分数 {config['shutdown_score']}！")
                    break
            
            # 记录分数历史
            if engine.best_score > 0:
                engine.score_history.append(engine.best_score)
            
            # 循环完成
            if engine.best_score >= config['shutdown_score']:
                break
        
        # 完成
        engine.is_running = False
        st.session_state.fitting_engine = engine
        status_placeholder.success(f"✨ 拟合完成！最终分数: {engine.best_score:.4f}")
        progress_placeholder.progress(1.0, text="已完成")
    
    # 显示拟合结果
    if st.session_state.fitting_engine is not None:
        engine = st.session_state.fitting_engine
        state = engine.get_state()
        
        if state['best_expression']:
            st.markdown("---")
            
            # 拟合结果对比
            x, y = st.session_state.canvas_data
            x_plot = np.linspace(x.min(), x.max(), 300)
            y_plot = engine.evaluate_expression(state['best_expression'], x_plot)
            
            render_fitting_comparison(
                original_x=x,
                original_y=y,
                fitted_x=x_plot,
                fitted_y=y_plot,
                expression=state['best_expression']
            )
            
            # 进度和结果
            render_progress_section(state)
            render_score_history(state['score_history'], key_suffix="_canvas")
            render_pareto_frontier(engine.get_pareto_frontier())
            render_api_calls_log(state['api_calls_log'])


def tab_npz_fitting():
    """Tab 2: NPZ文件上传拟合"""
    st.markdown("## 📁 上传NPZ文件拟合")
    
    # 文件上传
    st.markdown("### 📤 上传数据文件")
    uploaded_file = st.file_uploader(
        "选择 NPZ 文件",
        type=['npz'],
        help="上传包含 'x', 'y' 和可选 'error' 键的 NPZ 文件"
    )
    
    if uploaded_file is not None:
        try:
            npz_data = np.load(uploaded_file)
            
            if 'x' not in npz_data or 'y' not in npz_data:
                st.error("❌ NPZ文件必须包含 'x' 和 'y' 键！")
                return
            
            x = npz_data['x']
            y = npz_data['y']
            yerr = npz_data['error'] if 'error' in npz_data else None
            
            if x.ndim != 2:
                st.error("❌ 输入数据 'x' 必须是2维数组 (n_samples, n_features)！")
                return
            if y.ndim != 1:
                st.error("❌ 输出数据 'y' 必须是1维数组 (n_samples,)！")
                return
            
            st.session_state.npz_data = (x, y, yerr)
            st.success(f"✅ 成功加载数据：{x.shape[0]} 个样本，{x.shape[1]} 个特征")
            
            # 数据预览
            st.markdown("### 📊 数据预览")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("样本数", x.shape[0])
            with col2:
                st.metric("特征数", x.shape[1])
            with col3:
                st.metric("包含误差", "是" if yerr is not None else "否")
            
            # 对于1维特征,显示数据散点图
            if x.shape[1] == 1:
                st.markdown("### 📈 数据可视化")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x[:, 0], y=y, mode='markers', name='数据点', marker=dict(size=6)))
                if yerr is not None:
                    fig.add_trace(go.Scatter(
                        x=x[:, 0], y=y, 
                        error_y=dict(type='data', array=yerr, visible=True),
                        mode='markers', name='带误差', marker=dict(size=4, opacity=0.5)
                    ))
                fig.update_layout(title="数据散点图", xaxis_title="x1", yaxis_title="y", height=300, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True, key="npz_preview_chart")
            
            # 显示数据范围
            st.markdown("### 📋 数据范围")
            for i in range(min(x.shape[1], 5)):  # 最多显示前5个特征
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"x{i+1} 范围", f"[{x[:, i].min():.4f}, {x[:, i].max():.4f}]")
                with col2:
                    st.metric(f"x{i+1} 均值", f"{x[:, i].mean():.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("y 范围", f"[{y.min():.4f}, {y.max():.4f}]")
            with col2:
                st.metric("y 均值", f"{y.mean():.4f}")
            
        except Exception as e:
            st.error(f"❌ 加载文件出错: {str(e)}")
            return
    
    st.markdown("---")
    
    # 拟合控制
    st.markdown("### 🚀 拟合控制")
    
    config = st.session_state.config
    can_start = (
        st.session_state.npz_data is not None and
        len(config['models']) > 0 and
        len(config['functions']) > 0 and
        not st.session_state.fitting_running
    )
    
    if st.button("▶️ 开始拟合 (NPZ)", disabled=not can_start, use_container_width=True, key="start_npz_fitting"):
        x, y, yerr = st.session_state.npz_data
        
        # 创建实时更新的UI占位符
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()
        
        log_placeholder = st.empty()
        api_log_placeholder = st.empty()
        
        # 创建拟合引擎
        engine = FittingEngine(config)
        
        # 初始化fitter和searcher
        status_placeholder.info("🔧 初始化拟合引擎...")
        engine.initialize_fitter(x, y, yerr)
        engine.initialize_searcher(canvas_image=None)  # NPZ模式不使用图片
        
        # 设置状态
        engine.is_running = True
        engine.start_time = time.time()
        
        status_placeholder.success("✅ 初始化完成，开始拟合...")
        
        # 主循环
        logs = []
        api_logs = []
        total_cycles = config['cycle_num']
        unit_epochs = config['unit_interaction_num']
        
        for cycle in range(total_cycles):
            engine.current_cycle = cycle + 1
            
            # 动态设置 filter_func
            if cycle % 3 == 0:
                engine.ideasearcher.set_filter_func(lambda idea: idea)
            else:
                engine.ideasearcher.set_filter_func(lambda idea: "")
            
            # 重新填充岛屿
            if cycle != 0:
                engine.ideasearcher.repopulate_islands()
            
            status_placeholder.info(f"🔄 循环 {engine.current_cycle}/{total_cycles} 进行中...")
            
            # 运行 epochs
            for epoch in range(unit_epochs):
                try:
                    # 运行一个 epoch
                    engine.ideasearcher.run(1)
                    
                    # 立即获取结果
                    try:
                        engine.best_expression = engine.fitter.get_best_fit()
                        engine.best_score = engine.ideasearcher.get_best_score()
                    except:
                        engine.best_expression = ""
                        engine.best_score = 0.0
                    
                    # 更新进度条
                    progress = (epoch + 1) / unit_epochs
                    progress_placeholder.progress(progress, text=f"循环 {engine.current_cycle}/{total_cycles} - Epoch {epoch + 1}/{unit_epochs}")
                    
                    # 更新日志
                    log_msg = f"🔹 Cycle {engine.current_cycle} · Epoch {epoch + 1}/{unit_epochs} · 分数: {engine.best_score:.4f}"
                    if engine.best_expression:
                        log_msg += f" · 表达式: {engine.best_expression[:50]}{'...' if len(engine.best_expression) > 50 else ''}"
                    logs.append(log_msg)
                    log_placeholder.text_area("📝 实时日志", "\n".join(logs[-20:]), height=200, key=f"npz_log_{cycle}_{epoch}")
                    
                    # 记录API调用
                    if engine.best_expression and engine.best_score > 0:
                        api_log_entry = f"[{time.strftime('%H:%M:%S')}] Cycle {engine.current_cycle} Epoch {epoch+1}: Score={engine.best_score:.4f}"
                        api_logs.append(api_log_entry)
                        api_log_placeholder.text_area("📞 API调用记录", "\n".join(api_logs[-15:]), height=150, key=f"npz_api_log_{cycle}_{epoch}")
                    
                    # 实时更新图表(仅1维特征)
                    if x.shape[1] == 1 and engine.best_expression and engine.best_score > 0:
                        try:
                            x_plot = np.linspace(x[:, 0].min(), x[:, 0].max(), 300).reshape(-1, 1)
                            y_plot = engine.evaluate_expression(engine.best_expression, x_plot)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x[:, 0], y=y, mode='markers', name='原始数据', marker=dict(size=8)))
                            if yerr is not None:
                                fig.add_trace(go.Scatter(
                                    x=x[:, 0], y=y,
                                    error_y=dict(type='data', array=yerr, visible=True),
                                    mode='markers', name='误差', marker=dict(size=4, opacity=0.3)
                                ))
                            fig.add_trace(go.Scatter(x=x_plot[:, 0], y=y_plot, mode='lines', name='拟合曲线', line=dict(width=3, color='red')))
                            fig.update_layout(title=f"实时拟合结果 (分数: {engine.best_score:.4f})", height=400)
                            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"npz_chart_{cycle}_{epoch}")
                        except Exception as e:
                            pass
                    
                    # 显示指标
                    with metrics_placeholder.container():
                        st.metric("当前分数", f"{engine.best_score:.4f}")
                        st.metric("循环进度", f"{engine.current_cycle}/{total_cycles}")
                        st.metric("运行时间", f"{time.time() - engine.start_time:.1f}s")
                        if engine.best_expression:
                            st.code(engine.best_expression, language='python')
                    
                    # 强制刷新控制台输出
                    print(f"[NPZ Fitting - Cycle {engine.current_cycle} - Epoch {epoch + 1}] Score: {engine.best_score:.4f}")
                    sys.stdout.flush()
                    
                except Exception as e:
                    error_msg = f"❌ Epoch {epoch + 1} 执行出错: {str(e)}"
                    logs.append(error_msg)
                    log_placeholder.text_area("📝 实时日志", "\n".join(logs[-20:]), height=200, key=f"npz_log_err_{cycle}_{epoch}")
                    print(error_msg)
                    sys.stdout.flush()
                    # 不中断，继续下一个epoch
                    continue
                
                # 检查是否达到目标
                if engine.best_score >= config['shutdown_score']:
                    status_placeholder.success(f"🎯 达到目标分数 {config['shutdown_score']}！")
                    break
            
            # 记录分数历史
            if engine.best_score > 0:
                engine.score_history.append(engine.best_score)
            
            # 循环完成
            if engine.best_score >= config['shutdown_score']:
                break
        
        # 完成
        engine.is_running = False
        st.session_state.fitting_engine = engine
        status_placeholder.success(f"✨ 拟合完成！最终分数: {engine.best_score:.4f}")
        progress_placeholder.progress(1.0, text="已完成")
    
    # 显示拟合结果
    if st.session_state.fitting_engine is not None:
        engine = st.session_state.fitting_engine
        state = engine.get_state()
        
        if state['best_expression']:
            st.markdown("---")
            st.markdown("### 📊 拟合结果")
            
            # 如果是1维特征，显示拟合对比
            if st.session_state.npz_data is not None:
                x, y, yerr = st.session_state.npz_data
                
                if x.shape[1] == 1:
                    x_plot = np.linspace(x[:, 0].min(), x[:, 0].max(), 300).reshape(-1, 1)
                    y_plot = engine.evaluate_expression(state['best_expression'], x_plot)
                    
                    render_fitting_comparison(
                        original_x=x[:, 0],
                        original_y=y,
                        fitted_x=x_plot[:, 0],
                        fitted_y=y_plot,
                        expression=state['best_expression']
                    )
            
            # 进度和结果
            render_progress_section(state)
            render_score_history(state['score_history'], key_suffix="_npz")
            render_pareto_frontier(engine.get_pareto_frontier())
            render_api_calls_log(state['api_calls_log'])


def tab_npz_advanced_config():
    """Tab 3: NPZ高级配置 (原有的完整NPZ配置界面)"""
    st.markdown("## 🔧 NPZ文件高级配置拟合")
    
    # 文件上传
    st.markdown("### 📤 上传数据文件")
    uploaded_file = st.file_uploader(
        "选择 NPZ 文件",
        type=['npz'],
        help="上传包含 'x', 'y' 和可选 'error' 键的 NPZ 文件",
        key="advanced_npz_uploader"
    )
    
    if uploaded_file is not None:
        try:
            npz_data = np.load(uploaded_file)
            
            if 'x' not in npz_data or 'y' not in npz_data:
                st.error("❌ NPZ文件必须包含 'x' 和 'y' 键！")
                return
            
            x = npz_data['x']
            y = npz_data['y']
            yerr = npz_data['error'] if 'error' in npz_data else None
            
            if x.ndim != 2:
                st.error("❌ 输入数据 'x' 必须是2维数组 (n_samples, n_features)！")
                return
            if y.ndim != 1:
                st.error("❌ 输出数据 'y' 必须是1维数组 (n_samples,)！")
                return
            
            st.session_state.npz_data = (x, y, yerr)
            st.success(f"✅ 成功加载数据：{x.shape[0]} 个样本，{x.shape[1]} 个特征")
            
            # 数据预览
            st.markdown("### 📊 数据预览")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("样本数", x.shape[0])
            with col2:
                st.metric("特征数", x.shape[1])
            with col3:
                st.metric("包含误差", "是" if yerr is not None else "否")
            
            st.markdown("#### 数据范围")
            for i in range(min(x.shape[1], 5)):  # 最多显示5个特征
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"x{i+1} 范围", f"[{x[:, i].min():.4f}, {x[:, i].max():.4f}]")
            if x.shape[1] > 5:
                st.info(f"还有 {x.shape[1] - 5} 个特征未显示")
            st.metric("y 范围", f"[{y.min():.4f}, {y.max():.4f}]")
            
        except Exception as e:
            st.error(f"❌ 读取文件失败：{e}")
            return
    
    if st.session_state.npz_data is None:
        st.info("👆 请上传NPZ数据文件")
        return
    
    x, y, yerr = st.session_state.npz_data
    n_features = x.shape[1]
    
    st.markdown("---")
    
    # 配置区域
    st.markdown("### ⚙️ 拟合配置")
    
    # 初始化session state中的配置值（如果不存在）
    if 'npz_input_description' not in st.session_state:
        st.session_state.npz_input_description = "使用物理量的测量数据来推导目标物理量"
    if 'npz_output_name' not in st.session_state:
        st.session_state.npz_output_name = "F"
    if 'npz_output_description' not in st.session_state:
        st.session_state.npz_output_description = "力"
    if 'npz_output_unit' not in st.session_state:
        st.session_state.npz_output_unit = "kg*m/s^2"
    if 'npz_perform_unit_validation' not in st.session_state:
        st.session_state.npz_perform_unit_validation = True
    
    with st.expander("📋 基本信息", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            input_description = st.text_area(
                "输入描述",
                value=st.session_state.npz_input_description,
                height=100,
                key="npz_input_desc_widget"
            )
            st.session_state.npz_input_description = input_description
        with col2:
            output_name = st.text_input("输出变量名称", value=st.session_state.npz_output_name, key="npz_out_name")
            st.session_state.npz_output_name = output_name
            
            output_description = st.text_input("输出变量描述", value=st.session_state.npz_output_description, key="npz_out_desc")
            st.session_state.npz_output_description = output_description
            
            output_unit = st.text_input("输出变量单位", value=st.session_state.npz_output_unit, key="npz_out_unit")
            st.session_state.npz_output_unit = output_unit
    
    with st.expander("📊 特征变量配置", expanded=True):
        variable_names = []
        variable_units = []
        variable_descriptions = {}
        
        st.markdown("为每个特征配置名称、单位和描述：")
        for i in range(n_features):
            st.markdown(f"**特征 {i+1}:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                default_name = f"m" if i == 0 else f"a" if i == 1 else f"x{i+1}"
                var_name = st.text_input(f"变量名", value=default_name, key=f"var_name_{i}")
                variable_names.append(var_name)
            with col2:
                default_unit = "kg" if i == 0 else "m/s^2"
                var_unit = st.text_input(f"单位", value=default_unit, key=f"var_unit_{i}")
                variable_units.append(var_unit)
            with col3:
                default_desc = "质量" if i == 0 else "加速度"
                var_desc = st.text_input(f"描述", value=default_desc, key=f"var_desc_{i}")
                variable_descriptions[var_name] = var_desc
    
    with st.expander("🔬 高级选项"):
        perform_unit_validation = st.checkbox(
            "启用单位验证",
            value=st.session_state.npz_perform_unit_validation,
            help="确保生成的表达式在量纲上正确"
        )
        st.session_state.npz_perform_unit_validation = perform_unit_validation
    
    # 显示配置摘要
    with st.expander("📋 配置摘要"):
        st.write("**输入变量:**")
        for name, unit, desc in zip(variable_names, variable_units, variable_descriptions.values()):
            st.write(f"- {name} ({unit}): {desc}")
        st.write(f"**输出变量:** {output_name} ({output_unit}): {output_description}")
        st.write(f"**单位验证:** {'启用' if perform_unit_validation else '禁用'}")
    
    st.markdown("---")
    st.markdown("### 🚀 开始拟合")
    
    config = st.session_state.config
    can_start = (
        len(config['models']) > 0 and
        len(config['functions']) > 0 and
        output_name and
        all(variable_names) and
        all(variable_units) and
        output_unit and
        not st.session_state.fitting_running
    )
    
    if not can_start:
        if not output_name or not output_unit:
            st.warning("⚠️ 请填写输出变量名称和单位")
        elif not all(variable_names) or not all(variable_units):
            st.warning("⚠️ 请为所有特征配置变量名和单位")
        elif len(config['models']) == 0:
            st.warning("⚠️ 请在侧边栏选择至少一个模型")
        elif len(config['functions']) == 0:
            st.warning("⚠️ 请在侧边栏选择至少一个函数")
    
    if st.button("▶️ 开始拟合", disabled=not can_start, use_container_width=True, key="npz_fit"):
        # 创建实时更新的UI占位符
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()
        
        log_placeholder = st.empty()
        
        # 创建修改后的配置（包含NPZ特定的fitter参数）
        npz_config = config.copy()
        npz_config['fitter_params'] = {
            'perform_unit_validation': perform_unit_validation,
            'variable_names': variable_names,
            'variable_units': variable_units if perform_unit_validation else None,
            'output_name': output_name,
            'output_unit': output_unit if perform_unit_validation else None,
            'variable_descriptions': variable_descriptions,
            'output_description': output_description,
            'input_description': input_description,
        }
        
        # 创建拟合引擎
        engine = FittingEngine(npz_config)
        
        # 初始化fitter和searcher
        status_placeholder.info("🔧 初始化拟合引擎...")
        engine.initialize_fitter(x, y, yerr)
        engine.initialize_searcher(canvas_image=None)
        
        # 设置状态
        engine.is_running = True
        engine.start_time = time.time()
        
        status_placeholder.success("✅ 初始化完成，开始拟合...")
        
        # 主循环
        logs = []
        total_cycles = config['cycle_num']
        unit_epochs = config['unit_interaction_num']
        
        for cycle in range(total_cycles):
            engine.current_cycle = cycle + 1
            
            # 动态设置 filter_func
            if cycle % 3 == 0:
                engine.ideasearcher.set_filter_func(lambda idea: idea)
            else:
                engine.ideasearcher.set_filter_func(lambda idea: "")
            
            # 重新填充岛屿
            if cycle != 0:
                engine.ideasearcher.repopulate_islands()
            
            status_placeholder.info(f"🔄 循环 {engine.current_cycle}/{total_cycles} 进行中...")
            
            # 运行 epochs
            for epoch in range(unit_epochs):
                # 运行一个 epoch
                engine.ideasearcher.run(1)
                
                # 立即获取结果
                try:
                    engine.best_expression = engine.fitter.get_best_fit()
                    engine.best_score = engine.ideasearcher.get_best_score()
                except:
                    engine.best_expression = ""
                    engine.best_score = 0.0
                
                # 更新进度条
                progress = (epoch + 1) / unit_epochs
                progress_placeholder.progress(progress, text=f"循环 {engine.current_cycle}/{total_cycles} - Epoch {epoch + 1}/{unit_epochs}")
                
                # 更新日志
                log_msg = f"🔹 Cycle {engine.current_cycle} · Epoch {epoch + 1}/{unit_epochs} · 分数: {engine.best_score:.4f}"
                if engine.best_expression:
                    log_msg += f" · 表达式: {engine.best_expression[:50]}{'...' if len(engine.best_expression) > 50 else ''}"
                logs.append(log_msg)
                log_placeholder.text_area("📝 实时日志", "\n".join(logs[-20:]), height=200)
                
                # 实时更新图表
                if engine.best_expression and engine.best_score > 0:
                    try:
                        # 对于多维输入，显示预测vs实际值
                        x_plot = x[:, 0]
                        y_pred = engine.evaluate_expression(engine.best_expression, x)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_plot, y=y, mode='markers', name='原始数据', marker=dict(size=8)))
                        fig.add_trace(go.Scatter(x=x_plot, y=y_pred, mode='markers', name='拟合结果', marker=dict(size=8, color='red')))
                        fig.update_layout(title=f"实时拟合结果 (分数: {engine.best_score:.4f})", height=400)
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # 显示指标
                        with metrics_placeholder.container():
                            st.metric("当前分数", f"{engine.best_score:.4f}")
                            st.metric("循环进度", f"{engine.current_cycle}/{total_cycles}")
                            st.metric("运行时间", f"{time.time() - engine.start_time:.1f}s")
                            st.code(engine.best_expression, language='python')
                    except Exception as e:
                        pass
                
                # 检查是否达到目标
                if engine.best_score >= config['shutdown_score']:
                    status_placeholder.success(f"🎯 达到目标分数 {config['shutdown_score']}！")
                    break
            
            # 记录分数历史
            if engine.best_score > 0:
                engine.score_history.append(engine.best_score)
            
            # 循环完成
            if engine.best_score >= config['shutdown_score']:
                break
        
        # 完成
        engine.is_running = False
        st.session_state.fitting_engine = engine
        status_placeholder.success(f"✨ 拟合完成！最终分数: {engine.best_score:.4f}")
        progress_placeholder.progress(1.0, text="已完成")
    
    # 显示拟合结果
    if st.session_state.fitting_engine is not None:
        engine = st.session_state.fitting_engine
        state = engine.get_state()
        
        if state['best_expression']:
            st.markdown("---")
            
            # 对于多维输入，显示预测vs实际值
            x_plot = x[:, 0] if x.shape[1] > 1 else x.flatten()
            y_pred = engine.evaluate_expression(state['best_expression'], x)
            
            render_fitting_comparison(
                original_x=x_plot,
                original_y=y,
                fitted_x=x_plot,
                fitted_y=y_pred,
                expression=state['best_expression']
            )
            
            # 进度和结果
            render_progress_section(state)
            render_score_history(state['score_history'], key_suffix="_npz_advanced")
            render_pareto_frontier(engine.get_pareto_frontier())
            render_api_calls_log(state['api_calls_log'])


def main():
    """主函数"""
    init_session_state()
    
    # 标题
    st.title("🔬 IdeaSearch 符号回归系统")
    st.markdown("基于大语言模型的智能符号回归 - 支持绘图拟合和文件上传拟合")
    st.markdown("---")
    
    # 侧边栏配置
    config = render_sidebar_config()
    st.session_state.config = config
    
    # Tab切换 - 3个标签页
    tab1, tab2, tab3 = st.tabs(["🎨 绘制曲线拟合", "📁 上传文件拟合", "⚙️ 高级配置拟合"])
    
    with tab1:
        tab_canvas_fitting()
    
    with tab2:
        tab_npz_fitting()
    
    with tab3:
        tab_npz_advanced_config()
    
    # 页脚
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>🔬 Powered by IdeaSearch Framework | Built with Streamlit</p>
            <p>基于大语言模型的智能符号回归系统</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
