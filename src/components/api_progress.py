"""
API 调用进度指示器
实时显示 API 调用进度，减少用户焦虑
"""

import streamlit as st
from typing import Dict, Any


def render_api_progress_indicator(state: Dict[str, Any]) -> None:
    """
    渲染 API 调用进度指示器
    
    Args:
        state: 拟合状态字典
    """
    current_cycle = state.get('current_cycle', 0)
    total_cycles = state.get('total_cycles', 0)
    total_api_calls = state.get('total_api_calls', 0)
    is_running = state.get('is_running', False)
    
    if not is_running and current_cycle == 0:
        return
    
    # 创建醒目的进度卡片
    st.markdown("""
        <style>
        .api-progress-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .api-progress-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .api-progress-details {
            font-size: 16px;
            opacity: 0.9;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    status_emoji = "🔄" if is_running else "✅"
    status_text = "正在运行..." if is_running else "已完成"
    
    st.markdown(f"""
        <div class="api-progress-card {'pulse' if is_running else ''}">
            <div class="api-progress-title">
                {status_emoji} API 调用进度 - {status_text}
            </div>
            <div class="api-progress-details">
                📊 当前循环: {current_cycle}/{total_cycles}<br>
                📡 API 调用次数: {total_api_calls}<br>
                {'⏳ 请稍候，模型正在思考中...' if is_running else '🎉 处理完成！'}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 添加进度条
    if total_cycles > 0:
        progress = current_cycle / total_cycles
        st.progress(progress, text=f"总体进度: {progress*100:.1f}%")


def render_cycle_status(state: Dict[str, Any]) -> None:
    """
    渲染当前循环状态
    
    Args:
        state: 拟合状态字典
    """
    current_cycle = state.get('current_cycle', 0)
    best_score = state.get('best_score', 0.0)
    best_expression = state.get('best_expression', '')
    
    if current_cycle == 0:
        return
    
    # 显示当前循环的最佳结果
    st.markdown("### 🎯 当前最佳结果")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label="最佳分数",
            value=f"{best_score:.4f}",
            delta="当前循环更新" if current_cycle > 0 else None
        )
    
    with col2:
        if best_expression:
            st.code(best_expression, language='python')
        else:
            st.info("等待拟合结果...")
