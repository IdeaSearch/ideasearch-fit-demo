"""
绘图画布组件
用于用户绘制目标曲线
"""

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional
import base64
from io import BytesIO
from PIL import Image


def extract_curve_from_canvas(canvas_result, width: int, height: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从画布结果中提取曲线数据
    
    Args:
        canvas_result: 画布结果对象
        width: 画布宽度
        height: 画布高度
    
    Returns:
        (x, y) 数组元组，如果没有数据则返回 None
    """
    if canvas_result.image_data is None:
        return None
    
    # 获取非白色像素点
    image_data = canvas_result.image_data
    non_white = np.where(image_data[:, :, 0] < 255)
    
    if len(non_white[0]) == 0:
        return None
    
    # 提取坐标点
    y_pixels = non_white[0]
    x_pixels = non_white[1]
    
    # 归一化到 [-10, 10] 范围
    x_normalized = (x_pixels / width) * 20 - 10
    y_normalized = 10 - (y_pixels / height) * 20
    
    # 按 x 值排序
    sorted_indices = np.argsort(x_normalized)
    x_sorted = x_normalized[sorted_indices]
    y_sorted = y_normalized[sorted_indices]
    
    # 去除重复的 x 值（取平均 y 值）
    unique_x, indices = np.unique(x_sorted, return_inverse=True)
    unique_y = np.array([y_sorted[indices == i].mean() for i in range(len(unique_x))])
    
    return unique_x, unique_y


def smooth_curve(x: np.ndarray, y: np.ndarray, sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    平滑曲线数据
    
    Args:
        x: x 坐标数组
        y: y 坐标数组
        sigma: 高斯滤波器标准差
    
    Returns:
        平滑后的 (x, y) 数组元组
    """
    if len(x) < 3:
        return x, y
    
    y_smooth = gaussian_filter1d(y, sigma=sigma)
    return x, y_smooth


def interpolate_curve(x: np.ndarray, y: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    插值曲线数据
    
    Args:
        x: x 坐标数组
        y: y 坐标数组
        num_points: 插值后的点数
    
    Returns:
        插值后的 (x, y) 数组元组
    """
    if len(x) < 3:
        return x, y
    
    # 使用三次样条插值
    try:
        f = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
        x_interp = np.linspace(x.min(), x.max(), num_points)
        y_interp = f(x_interp)
        return x_interp, y_interp
    except Exception:
        # 如果插值失败，使用线性插值
        f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        x_interp = np.linspace(x.min(), x.max(), num_points)
        y_interp = f(x_interp)
        return x_interp, y_interp


def canvas_to_base64(canvas_result) -> Optional[str]:
    """
    将画布转换为 base64 编码的图片
    
    Args:
        canvas_result: 画布结果对象
    
    Returns:
        base64 编码的图片字符串，如果没有数据则返回 None
    """
    if canvas_result.image_data is None:
        return None
    
    try:
        # 转换为 PIL Image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        
        # 转换为 base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return img_base64
    except Exception as e:
        print(f"转换画布为 base64 失败: {e}")
        return None


def render_drawing_canvas(config: dict) -> dict:
    """
    渲染绘图画布
    
    Args:
        config: 画布配置字典
    
    Returns:
        画布结果
    """
    st.markdown("### 🎨 绘制目标曲线")
    st.markdown("在下方画布上绘制您想要拟合的曲线")
    
    # 绘制模式选择和图片传递开关
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        drawing_mode = st.selectbox(
            "绘制模式",
            ["freedraw", "line", "point"],
            format_func=lambda x: {
                "freedraw": "自由绘制",
                "line": "直线",
                "point": "点"
            }[x],
            key="drawing_mode"
        )
    
    with col2:
        stroke_width = st.slider("线条宽度", 1, 10, config.get('stroke_width', 3), key="stroke_width")
    
    with col3:
        use_canvas_image = st.checkbox(
            "📷 传递图片",
            value=False,
            help="将画布图片传递给支持视觉的模型（如 Gemini），帮助理解拟合目标。注意：不支持图片的模型会忽略此选项。",
            key="use_canvas_image"
        )
    
    # 清除按钮（放在画布前面，这样可以先处理清除逻辑）
    if st.button("🗑️ 清除画布", key="clear_canvas"):
        # 删除画布相关的 session state
        if 'canvas_key' in st.session_state:
            st.session_state.canvas_key += 1
        else:
            st.session_state.canvas_key = 1
        st.rerun()
    
    # 获取画布 key
    canvas_key = st.session_state.get('canvas_key', 0)
    
    # 创建画布
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=config.get('stroke_color', '#1f77b4'),
        background_color=config.get('background_color', '#ffffff'),
        height=config.get('height', 400),
        width=config.get('width', 600),
        drawing_mode=drawing_mode,
        key=f"canvas_{canvas_key}",
        update_streamlit=config.get('update_streamlit', True),
    )
    
    return canvas_result


def process_canvas_data(
    canvas_result,
    width: int,
    height: int,
    smooth: bool = False,
    interpolate: bool = True,
    num_points: int = 100,
    smooth_sigma: float = 2.0
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    处理画布数据，包括提取、平滑和插值
    
    Args:
        canvas_result: 画布结果对象
        width: 画布宽度
        height: 画布高度
        smooth: 是否平滑
        interpolate: 是否插值
        num_points: 插值点数
        smooth_sigma: 平滑参数
    
    Returns:
        处理后的 (x, y) 数组元组，如果没有数据则返回 None
    """
    # 提取曲线
    curve_data = extract_curve_from_canvas(canvas_result, width, height)
    if curve_data is None:
        return None
    
    x, y = curve_data
    
    # 平滑处理
    if smooth and len(x) >= 3:
        x, y = smooth_curve(x, y, sigma=smooth_sigma)
    
    # 插值处理
    if interpolate and len(x) >= 3:
        x, y = interpolate_curve(x, y, num_points=num_points)
    
    return x, y
