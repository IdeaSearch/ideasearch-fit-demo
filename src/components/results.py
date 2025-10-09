"""
结果展示组件
展示拟合结果、进度和调用记录
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from ..utils import t


def render_progress_section(state: Dict[str, Any]) -> None:
    """
    渲染进度监控区域（增强版）
    
    Args:
        state: 拟合状态字典
    """
    st.markdown(f"### {t('progress.title')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_cycle = state.get('current_cycle', 0)
        total_cycles = state.get('total_cycles', 0)
        st.metric(t('progress.current_cycle'), f"{current_cycle}/{total_cycles}")
    
    with col2:
        best_score = state.get('best_score', 0.0)
        # 显示分数变化趋势
        score_history = state.get('score_history', [])
        delta = None
        if len(score_history) >= 2:
            delta = score_history[-1] - score_history[-2]
        st.metric(t('progress.best_score'), f"{best_score:.4f}", delta=f"{delta:.4f}" if delta is not None else None)
    
    with col3:
        total_api_calls = state.get('total_api_calls', 0)
        # 估算平均每循环的API调用数
        avg_calls = total_api_calls / current_cycle if current_cycle > 0 else 0
        st.metric(t('progress.api_calls'), f"{total_api_calls}", delta=t('progress.avg_per_cycle', avg=avg_calls) if avg_calls > 0 else None, delta_color="off")
    
    with col4:
        elapsed_time = state.get('elapsed_time', 0.0)
        # 估算剩余时间
        if current_cycle > 0 and total_cycles > current_cycle:
            avg_time_per_cycle = elapsed_time / current_cycle
            remaining_time = avg_time_per_cycle * (total_cycles - current_cycle)
            st.metric(t('progress.elapsed_time'), f"{elapsed_time:.1f}s", delta=t('progress.remaining_time', time=remaining_time), delta_color="off")
        else:
            st.metric(t('progress.elapsed_time'), f"{elapsed_time:.1f}s")
    
    # 进度条（带百分比）
    if total_cycles > 0:
        progress = current_cycle / total_cycles
        progress_text = t('progress.progress_text', progress=progress*100, current=current_cycle, total=total_cycles)
        st.progress(progress, text=progress_text)


def render_api_calls_log(api_calls: List[Dict[str, Any]], max_display: int = 50) -> None:
    """
    渲染 API 调用记录
    
    Args:
        api_calls: API 调用记录列表
        max_display: 最大显示记录数
    """
    st.markdown(f"### {t('api_calls.title')}")
    
    if not api_calls:
        st.info(t('api_calls.no_records'))
        return
    
    # 显示最新的记录
    display_calls = api_calls[-max_display:]
    
    # 创建表格数据
    records = []
    for i, call in enumerate(display_calls, 1):
        # 根据状态设置不同的显示风格
        status = call.get('status', 'unknown')
        expression_display = call.get('expression', 'N/A')
        
        # 对于过长的表达式，进行截断处理
        if len(expression_display) > 50:
            expression_display = expression_display[:47] + '...'
        
        # 根据状态添加状态标识
        if status == 'no_expression':
            expression_display = "⏳ " + expression_display
        elif status == 'success' and call.get('score', 0) > 0:
            expression_display = "✅ " + expression_display
        else:
            expression_display = "🔍 " + expression_display
        
        records.append({
            t('api_calls.sequence'): len(api_calls) - max_display + i if len(api_calls) > max_display else i,
            t('api_calls.cycle'): t('api_calls.cycle_format', cycle=call.get('cycle', 'N/A')),
            t('api_calls.model'): call.get('model', 'N/A'),
            t('api_calls.expression_status'): expression_display,
            t('api_calls.fitting_score'): f"{call.get('score', 0.0):.4f}",
            t('api_calls.call_count'): call.get('total_api_calls', 'N/A'),
            t('api_calls.timestamp'): call.get('timestamp', 'N/A')
        })
    
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, height=300)
    
    # 状态说明
    with st.expander(t('api_calls.status_legend')):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(t('api_calls.status_success'))
            st.write(t('api_calls.status_success_desc'))
        with col2:
            st.write(t('api_calls.status_searching'))
            st.write(t('api_calls.status_searching_desc'))
        with col3:
            st.write(t('api_calls.status_exploring'))
            st.write(t('api_calls.status_exploring_desc'))
    
    # 统计信息
    with st.expander(t('api_calls.call_statistics')):
        col1, col2 = st.columns(2)
        with col1:
            st.write(t('api_calls.model_usage'))
            model_counts = {}
            for call in api_calls:
                model = call.get('model', 'Unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {model}: {count} 次")
            
            # 状态统计
            st.write(t('api_calls.task_status'))
            status_counts = {}
            for call in api_calls:
                status = call.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            status_labels = {
                'success': t('api_calls.status_success_count'),
                'no_expression': t('api_calls.status_searching_count'),
                'unknown': t('api_calls.status_exploring_count')
            }
            
            for status, count in status_counts.items():
                label = status_labels.get(status, t('api_calls.unknown_status', status=status))
                st.write(f"- {label}: {count} 次")
        
        with col2:
            st.write(t('api_calls.score_distribution'))
            scores = [call.get('score', 0) for call in api_calls]
            if scores:
                valid_scores = [s for s in scores if s > 0]
                if valid_scores:
                    st.write(f"- {t('api_calls.avg_score', score=np.mean(scores))}")
                    st.write(f"- {t('api_calls.max_score', score=np.max(scores))}")
                    st.write(f"- {t('api_calls.min_score', score=np.min(scores))}")
                    st.write(f"- {t('api_calls.valid_scores', valid=len(valid_scores), total=len(scores))}")
                else:
                    st.write(f"- {t('api_calls.no_valid_scores')}")
                    st.write(f"- {t('api_calls.total_records', count=len(scores))}")
            
            # API调用效率
            st.write(t('api_calls.api_efficiency'))
            if api_calls:
                latest_call = api_calls[-1]
                total_calls = latest_call.get('total_api_calls', 0)
                cycles = latest_call.get('cycle', 0)
                if cycles > 0 and total_calls > 0:
                    avg_calls_per_cycle = total_calls / cycles
                    st.write(f"- {t('api_calls.total_api_calls', count=total_calls)}")
                    st.write(f"- {t('api_calls.avg_calls_per_cycle', avg=avg_calls_per_cycle)}")
                else:
                    st.write(f"- {t('api_calls.calculating')}")


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
    st.markdown(f"### {t('results.comparison_title')}")
    
    # 创建图表
    fig = go.Figure()
    
    # 添加原始曲线
    fig.add_trace(go.Scatter(
        x=original_x,
        y=original_y,
        mode='markers',
        name=t('results.original_data'),
        marker=dict(size=8, color='#1f77b4', opacity=0.6)
    ))
    
    # 添加拟合曲线
    if fitted_x is not None and fitted_y is not None:
        fig.add_trace(go.Scatter(
            x=fitted_x,
            y=fitted_y,
            mode='lines',
            name=t('results.fitted_result'),
            line=dict(color='#ff7f0e', width=3)
        ))
    
    # 更新布局
    fig.update_layout(
        title=t('results.comparison_chart_title'),
        xaxis_title=t('results.x_axis'),
        yaxis_title=t('results.y_axis'),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True, key="fitting_comparison_chart")
    
    # 显示拟合表达式
    if expression:
        st.markdown(f"### {t('results.fitting_expression')}")
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
                st.metric(t('results.mse'), f"{mse:.4f}")
            with col2:
                st.metric(t('results.rmse'), f"{rmse:.4f}")
            with col3:
                st.metric(t('results.mae'), f"{mae:.4f}")


def render_pareto_frontier(pareto_data: Dict[int, Dict]) -> None:
    """
    渲染 Pareto 前沿图
    
    Args:
        pareto_data: Pareto 前沿数据字典
    """
    st.markdown(f"### {t('pareto.title')}")
    
    if not pareto_data:
        st.info(t('pareto.no_data'))
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
        title=t('pareto.chart_title'),
        xaxis_title=t('pareto.complexity'),
        yaxis_title=t('pareto.score'),
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key="pareto_frontier_chart")
    
    # 显示详细表格
    with st.expander(t('pareto.detail_data')):
        records = []
        for complexity, info in sorted(pareto_data.items()):
            records.append({
                t('pareto.complexity_col'): complexity,
                t('pareto.score_col'): f"{info.get('score', 0):.2f}",
                t('pareto.expression_col'): info.get('ansatz', 'N/A'),
                t('pareto.created_time'): info.get('created_at', 'N/A')
            })
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)


def render_score_history(score_history: List[float], key_suffix: str = "") -> None:
    """
    渲染分数历史曲线
    
    Args:
        score_history: 分数历史列表
        key_suffix: 用于生成唯一 key 的后缀
    """
    st.markdown(f"### {t('score_history.title')}")
    
    with st.expander(t('score_history.score_calculation'), expanded=False):
        st.write(t('score_history.calculation_formula'))
        st.code(t('score_history.formula'), language="text")
        st.write(t('score_history.formula_desc'))
        st.write(f"- {t('score_history.mse_desc')}")
        st.write(f"- {t('score_history.linear_mse_desc')}")
    
    if not score_history:
        st.info(t('score_history.no_history'))
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(score_history) + 1)),
        y=score_history,
        mode='lines+markers',
        name=t('score_history.best_score'),
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=t('score_history.chart_title'),
        xaxis_title=t('score_history.cycle_axis'),
        yaxis_title=t('score_history.score_axis'),
        template='plotly_white',
        height=300
    )
    
    # 使用唯一的 key
    chart_key = f"score_history_chart{key_suffix}" if key_suffix else "score_history_chart"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
