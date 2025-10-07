"""
结果展示组件
展示拟合结果、进度和调用记录
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def render_progress_section(state: Dict[str, Any]) -> None:
    """
    渲染进度监控区域（增强版）
    
    Args:
        state: 拟合状态字典
    """
    st.markdown("### 📊 拟合进度")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_cycle = state.get('current_cycle', 0)
        total_cycles = state.get('total_cycles', 0)
        st.metric("当前循环", f"{current_cycle}/{total_cycles}")
    
    with col2:
        best_score = state.get('best_score', 0.0)
        # 显示分数变化趋势
        score_history = state.get('score_history', [])
        delta = None
        if len(score_history) >= 2:
            delta = score_history[-1] - score_history[-2]
        st.metric("最佳分数", f"{best_score:.4f}", delta=f"{delta:.4f}" if delta is not None else None)
    
    with col3:
        total_api_calls = state.get('total_api_calls', 0)
        # 估算平均每循环的API调用数
        avg_calls = total_api_calls / current_cycle if current_cycle > 0 else 0
        st.metric("API 调用", f"{total_api_calls}", delta=f"平均 {avg_calls:.1f}/循环" if avg_calls > 0 else None, delta_color="off")
    
    with col4:
        elapsed_time = state.get('elapsed_time', 0.0)
        # 估算剩余时间
        if current_cycle > 0 and total_cycles > current_cycle:
            avg_time_per_cycle = elapsed_time / current_cycle
            remaining_time = avg_time_per_cycle * (total_cycles - current_cycle)
            st.metric("已用时间", f"{elapsed_time:.1f}s", delta=f"预计剩余 {remaining_time:.1f}s", delta_color="off")
        else:
            st.metric("已用时间", f"{elapsed_time:.1f}s")
    
    # 进度条（带百分比）
    if total_cycles > 0:
        progress = current_cycle / total_cycles
        progress_text = f"进度: {progress*100:.1f}% ({current_cycle}/{total_cycles} 循环)"
        st.progress(progress, text=progress_text)


def render_api_calls_log(api_calls: List[Dict[str, Any]], max_display: int = 50) -> None:
    """
    渲染 API 调用记录
    
    Args:
        api_calls: API 调用记录列表
        max_display: 最大显示记录数
    """
    st.markdown("### 📝 模型调用记录")
    
    if not api_calls:
        st.info("暂无调用记录")
        return
    
    # 显示最新的记录
    display_calls = api_calls[-max_display:]
    
    # 创建表格数据
    records = []
    for i, call in enumerate(display_calls, 1):
        records.append({
            '序号': len(api_calls) - max_display + i if len(api_calls) > max_display else i,
            '模型': call.get('model', 'N/A'),
            '表达式': call.get('expression', 'N/A')[:50] + '...' if len(call.get('expression', '')) > 50 else call.get('expression', 'N/A'),
            '分数': f"{call.get('score', 0.0):.2f}",
            '时间': call.get('timestamp', 'N/A'),
        })
    
    df = pd.DataFrame(records)
    st.dataframe(df, width='stretch', height=300)
    
    # 统计信息
    with st.expander("📈 调用统计"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**模型使用次数:**")
            model_counts = {}
            for call in api_calls:
                model = call.get('model', 'Unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {model}: {count} 次")
        
        with col2:
            st.write("**分数分布:**")
            scores = [call.get('score', 0) for call in api_calls]
            if scores:
                st.write(f"- 平均分数: {np.mean(scores):.2f}")
                st.write(f"- 最高分数: {np.max(scores):.2f}")
                st.write(f"- 最低分数: {np.min(scores):.2f}")


def render_fitting_comparison(
    original_x: np.ndarray,
    original_y: np.ndarray,
    fitted_x: Optional[np.ndarray] = None,
    fitted_y: Optional[np.ndarray] = None,
    expression: Optional[str] = None
) -> None:
    """
    渲染原始曲线与拟合结果对比
    
    Args:
        original_x: 原始 x 数据
        original_y: 原始 y 数据
        fitted_x: 拟合 x 数据
        fitted_y: 拟合 y 数据
        expression: 拟合表达式
    """
    st.markdown("### 📈 拟合结果对比")
    
    # 创建图表
    fig = go.Figure()
    
    # 添加原始曲线
    fig.add_trace(go.Scatter(
        x=original_x,
        y=original_y,
        mode='markers',
        name='原始数据',
        marker=dict(size=8, color='#1f77b4', opacity=0.6)
    ))
    
    # 添加拟合曲线
    if fitted_x is not None and fitted_y is not None:
        fig.add_trace(go.Scatter(
            x=fitted_x,
            y=fitted_y,
            mode='lines',
            name='拟合结果',
            line=dict(color='#ff7f0e', width=3)
        ))
    
    # 更新布局
    fig.update_layout(
        title="原始曲线 vs 拟合曲线",
        xaxis_title="x",
        yaxis_title="y",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示拟合表达式
    if expression:
        st.markdown("### 🔢 拟合表达式")
        st.code(expression, language='python')
        
        # 计算拟合误差
        if fitted_x is not None and fitted_y is not None:
            # 在原始数据点上插值拟合结果
            from scipy import interpolate
            f = interpolate.interp1d(fitted_x, fitted_y, kind='linear', fill_value='extrapolate')
            fitted_at_original = f(original_x)
            
            mse = np.mean((original_y - fitted_at_original) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(original_y - fitted_at_original))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("均方误差 (MSE)", f"{mse:.4f}")
            with col2:
                st.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
            with col3:
                st.metric("平均绝对误差 (MAE)", f"{mae:.4f}")


def render_pareto_frontier(pareto_data: Dict[int, Dict]) -> None:
    """
    渲染 Pareto 前沿图
    
    Args:
        pareto_data: Pareto 前沿数据字典
    """
    st.markdown("### 🎯 Pareto 前沿")
    
    if not pareto_data:
        st.info("暂无 Pareto 前沿数据")
        return
    
    # 提取数据
    complexities = []
    scores = []
    expressions = []
    
    for complexity, info in pareto_data.items():
        complexities.append(complexity)
        scores.append(info.get('score', 0))
        expr = info.get('ansatz', 'N/A')
        expressions.append(expr[:30] + '...' if len(expr) > 30 else expr)
    
    # 创建散点图
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=complexities,
        y=scores,
        mode='markers+lines',
        marker=dict(size=10, color=scores, colorscale='Viridis', showscale=True),
        text=expressions,
        hovertemplate='<b>复杂度:</b> %{x}<br><b>分数:</b> %{y:.2f}<br><b>表达式:</b> %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="复杂度 vs 分数 (Pareto 前沿)",
        xaxis_title="表达式复杂度",
        yaxis_title="拟合分数",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示详细表格
    with st.expander("📋 查看详细数据"):
        records = []
        for complexity, info in sorted(pareto_data.items()):
            records.append({
                '复杂度': complexity,
                '分数': f"{info.get('score', 0):.2f}",
                '表达式': info.get('ansatz', 'N/A'),
                '创建时间': info.get('created_at', 'N/A')
            })
        df = pd.DataFrame(records)
        st.dataframe(df, width='stretch')


def render_score_history(score_history: List[float], key_suffix: str = "") -> None:
    """
    渲染分数历史曲线
    
    Args:
        score_history: 分数历史列表
        key_suffix: 用于生成唯一 key 的后缀
    """
    st.markdown("### 📉 拟合历史")
    
    if not score_history:
        st.info("暂无历史数据")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(score_history) + 1)),
        y=score_history,
        mode='lines+markers',
        name='最佳分数',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="最佳分数随循环变化",
        xaxis_title="循环次数",
        yaxis_title="最佳分数",
        template='plotly_white',
        height=300
    )
    
    # 使用唯一的 key
    chart_key = f"score_history_chart{key_suffix}" if key_suffix else "score_history_chart"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
