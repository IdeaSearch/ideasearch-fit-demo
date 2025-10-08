"""
配置侧边栏组件
提供所有可配置参数的 UI 控件
"""

import streamlit as st
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any


def load_default_config() -> Dict[str, Any]:
    """加载默认配置"""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_available_models(api_keys_path: str) -> List[str]:
    """从 API 密钥文件加载可用模型列表"""
    try:
        with open(api_keys_path, 'r', encoding='utf-8') as f:
            api_keys = json.load(f)
        return list(api_keys.keys())
    except Exception as e:
        st.error(f"加载 API 密钥文件失败: {e}")
        return []


def render_sidebar_config() -> Dict[str, Any]:
    """
    渲染侧边栏配置界面
    
    Returns:
        配置字典
    """
    default_config = load_default_config()
    
    st.sidebar.markdown("# ⚙️ 配置参数")
    
    # API 配置
    st.sidebar.markdown("## 🔑 API 配置")
    api_keys_path = st.sidebar.text_input(
        "API 密钥文件路径",
        value="api_keys.json",
        help="指定包含 API 密钥的 JSON 文件路径"
    )
    
    # 加载可用模型
    available_models = load_available_models(api_keys_path)
    
    if available_models:
        # 优先选择 Gemini_2.5_Flash，否则选择第一个
        default_model = 'Gemini_2.5_Flash' if 'Gemini_2.5_Flash' in available_models else (available_models[0] if available_models else None)
        selected_models = st.sidebar.multiselect(
            "选择模型",
            options=available_models,
            default=[default_model] if default_model else [],
            help="选择一个或多个 LLM 模型进行拟合"
        )
    else:
        selected_models = []
        st.sidebar.warning("未找到可用模型，请检查 API 密钥文件")
    
    # 函数配置
    st.sidebar.markdown("## 📐 函数配置")
    available_functions = default_config['fitter']['available_functions']
    
    selected_functions = st.sidebar.multiselect(
        "可用函数",
        options=available_functions,
        default=available_functions,
        help="选择拟合时可以使用的数学函数"
    )
    
    # 常量配置
    use_constants = st.sidebar.checkbox(
        "使用预定义常量",
        value=True,
        help="允许在表达式中使用预定义的数学常量（如 π）"
    )
    
    constant_whitelist = []
    constant_map = {}
    if use_constants:
        constant_whitelist = ["pi"]
        constant_map = {"pi": 3.141592653589793}
    
    # 拟合参数
    st.sidebar.markdown("## 🎯 拟合参数")
    
    island_num = st.sidebar.number_input(
        "岛屿数量",
        min_value=1,
        max_value=20,
        value=default_config['ideasearch']['island_num'],
        help="并行进化的种群数量，增加可提高多样性"
    )
    
    cycle_num = st.sidebar.number_input(
        "循环次数",
        min_value=1,
        max_value=100,
        value=default_config['ideasearch']['cycle_num'],
        help="进化算法的代数，更多循环通常得到更好结果"
    )
    
    unit_interaction_num = st.sidebar.number_input(
        "单元交互数",
        min_value=1,
        max_value=100,
        value=default_config['ideasearch']['unit_interaction_num'],
        help="每个循环中的 LLM 调用次数"
    )
    
    shutdown_score = st.sidebar.slider(
        "目标分数",
        min_value=0.0,
        max_value=100.0,
        value=default_config['ideasearch']['shutdown_score'],
        help="达到此分数时自动停止拟合"
    )
    
    # 高级参数
    with st.sidebar.expander("🔧 高级参数"):
        samplers_num = st.number_input(
            "采样器数量",
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['samplers_num'],
            help="并行采样的工作线程数"
        )
        
        evaluators_num = st.number_input(
            "评估器数量",
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['evaluators_num'],
            help="并行评估的工作线程数"
        )
        
        sample_temperature = st.slider(
            "采样温度",
            min_value=0.0,
            max_value=100.0,
            value=float(default_config['ideasearch']['sample_temperature']),
            help="控制生成的随机性，越高越随机"
        )
        
        examples_num = st.number_input(
            "示例数量",
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['examples_num'],
            help="每次提示中包含的示例数量"
        )
        
        generate_num = st.number_input(
            "每次生成数",
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['generate_num'],
            help="每次 LLM 调用生成的表达式数量"
        )
        
        hand_over_threshold = st.slider(
            "交接阈值",
            min_value=-1.0,
            max_value=1.0,
            value=float(default_config['ideasearch']['hand_over_threshold']),
            step=0.1,
            help="岛屿间交换想法的分数阈值"
        )
    
    # Fuzzy 模式配置
    with st.sidebar.expander("🧠 Fuzzy 模式"):
        generate_fuzzy = st.checkbox(
            "启用 Fuzzy 模式",
            value=default_config['fitter']['generate_fuzzy'],
            help="使用自然语言理论描述作为中间步骤"
        )
        
        fuzzy_translator = None
        if generate_fuzzy and available_models:
            # 优先使用 Gemini_2.5_Flash
            default_idx = available_models.index('Gemini_2.5_Flash') if 'Gemini_2.5_Flash' in available_models else 0
            fuzzy_translator = st.selectbox(
                "Fuzzy 翻译器模型",
                options=available_models,
                index=default_idx,
                help="用于将理论描述转换为表达式的模型"
            )
    
    # 优化参数配置
    with st.sidebar.expander("⚡ 优化参数"):
        enable_mutation = st.checkbox(
            "启用变异",
            value=default_config['fitter'].get('enable_mutation', False),
            help="启用表达式变异操作，增加搜索多样性"
        )
        
        enable_crossover = st.checkbox(
            "启用交叉",
            value=default_config['fitter'].get('enable_crossover', False),
            help="启用表达式交叉操作，结合不同表达式特征"
        )
        
        optimization_method = st.selectbox(
            "优化方法",
            options=["L-BFGS-B", "differential-evolution"],
            index=0 if default_config['fitter'].get('optimization_method', 'L-BFGS-B') == 'L-BFGS-B' else 1,
            help="参数优化算法选择"
        )
        
        optimization_trial_num = st.number_input(
            "优化试验次数",
            min_value=1,
            max_value=1000,
            value=default_config['fitter'].get('optimization_trial_num', 5),
            help="每个表达式的参数优化试验次数，减小可显著提高拟合速度（推荐：5）"
        )

    # 数据处理配置
    with st.sidebar.expander("📊 数据处理"):
        num_points = st.number_input(
            "数据点数",
            min_value=10,
            max_value=1000,
            value=default_config['data']['num_points'],
            help="插值后的数据点数量"
        )
        
        smooth_data = st.checkbox(
            "数据平滑",
            value=default_config['data']['smooth_data'],
            help="对绘制的曲线进行平滑处理"
        )
        
        interpolate_data = st.checkbox(
            "数据插值",
            value=default_config['data']['interpolate_data'],
            help="对数据点进行插值以获得更均匀的分布"
        )
    
    # 为每个模型生成温度配置
    # 默认所有模型使用温度 1.0
    model_temperatures = [1.0] * len(selected_models)
    
    # 返回配置字典
    return {
        'api_keys_path': api_keys_path,
        'models': selected_models,
        'model_temperatures': model_temperatures,  # 关键！
        'functions': selected_functions,
        'constant_whitelist': constant_whitelist,
        'constant_map': constant_map,
        'island_num': island_num,
        'cycle_num': cycle_num,
        'unit_interaction_num': unit_interaction_num,
        'shutdown_score': shutdown_score,
        'samplers_num': samplers_num,
        'evaluators_num': evaluators_num,
        'sample_temperature': sample_temperature,
        'examples_num': examples_num,
        'generate_num': generate_num,
        'hand_over_threshold': hand_over_threshold,
        'generate_fuzzy': generate_fuzzy,
        'fuzzy_translator': fuzzy_translator,
        'num_points': num_points,
        'smooth_data': smooth_data,
        'interpolate_data': interpolate_data,
        # 新增的优化参数
        'enable_mutation': enable_mutation,
        'enable_crossover': enable_crossover,
        'optimization_method': optimization_method,
        'optimization_trial_num': optimization_trial_num,
        # 模型评估参数（从 default.yaml 读取）
        'model_assess_average_order': default_config['ideasearch'].get('model_assess_average_order', 15.0),
        'model_assess_initial_score': default_config['ideasearch'].get('model_assess_initial_score', 20.0),
        'record_prompt_in_diary': default_config['ideasearch'].get('record_prompt_in_diary', True),
    }
