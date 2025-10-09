"""
配置侧边栏组件
提供所有可配置参数的 UI 控件
"""

import streamlit as st
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
from ..utils import t


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
    
    st.sidebar.markdown(f"# {t('config.sidebar_title')}")
    
    # API 配置
    st.sidebar.markdown(f"## {t('config.api_config')}")
    api_keys_path = st.sidebar.text_input(
        t('config.api_keys_path'),
        value="api_keys.json",
        help=t('config.api_keys_help')
    )
    
    # 加载可用模型
    available_models = load_available_models(api_keys_path)
    
    if available_models:
        # 优先选择 Gemini_2.5_Flash，否则选择第一个
        default_model = 'Gemini_2.5_Flash' if 'Gemini_2.5_Flash' in available_models else (available_models[0] if available_models else None)
        selected_models = st.sidebar.multiselect(
            t('config.select_models'),
            options=available_models,
            default=[default_model] if default_model else [],
            help=t('config.models_help')
        )
    else:
        selected_models = []
        st.sidebar.warning(t('config.no_models_warning'))
    
    # 函数配置
    st.sidebar.markdown(f"## {t('config.function_config')}")
    available_functions = default_config['fitter']['available_functions']
    
    selected_functions = st.sidebar.multiselect(
        t('config.available_functions'),
        options=available_functions,
        default=available_functions,
        help=t('config.functions_help')
    )
    
    # 常量配置
    use_constants = st.sidebar.checkbox(
        t('config.use_constants'),
        value=True,
        help=t('config.constants_help')
    )
    
    constant_whitelist = []
    constant_map = {}
    if use_constants:
        constant_whitelist = ["pi"]
        constant_map = {"pi": 3.141592653589793}
    
    # 拟合参数
    st.sidebar.markdown(f"## {t('config.fitting_params')}")
    
    island_num = st.sidebar.number_input(
        t('config.island_num'),
        min_value=1,
        max_value=20,
        value=default_config['ideasearch']['island_num'],
        help=t('config.island_help')
    )
    
    cycle_num = st.sidebar.number_input(
        t('config.cycle_num'),
        min_value=1,
        max_value=100,
        value=default_config['ideasearch']['cycle_num'],
        help=t('config.cycle_help')
    )
    
    unit_interaction_num = st.sidebar.number_input(
        t('config.unit_interaction_num'),
        min_value=1,
        max_value=100,
        value=default_config['ideasearch']['unit_interaction_num'],
        help=t('config.interaction_help')
    )
    
    shutdown_score = st.sidebar.slider(
        t('config.shutdown_score'),
        min_value=0.0,
        max_value=100.0,
        value=default_config['ideasearch']['shutdown_score'],
        help=t('config.score_help')
    )
    
    # 高级参数
    with st.sidebar.expander(t('config.advanced_params')):
        samplers_num = st.number_input(
            t('config.samplers_num'),
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['samplers_num'],
            help=t('config.samplers_help')
        )
        
        evaluators_num = st.number_input(
            t('config.evaluators_num'),
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['evaluators_num'],
            help=t('config.evaluators_help')
        )
        
        sample_temperature = st.slider(
            t('config.sample_temperature'),
            min_value=0.0,
            max_value=100.0,
            value=float(default_config['ideasearch']['sample_temperature']),
            help=t('config.temperature_help')
        )
        
        examples_num = st.number_input(
            t('config.examples_num'),
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['examples_num'],
            help=t('config.examples_help')
        )
        
        generate_num = st.number_input(
            t('config.generate_num'),
            min_value=1,
            max_value=10,
            value=default_config['ideasearch']['generate_num'],
            help=t('config.generate_help')
        )
        
        hand_over_threshold = st.slider(
            t('config.hand_over_threshold'),
            min_value=-1.0,
            max_value=1.0,
            value=float(default_config['ideasearch']['hand_over_threshold']),
            step=0.1,
            help=t('config.threshold_help')
        )
    
    # Fuzzy 模式配置
    with st.sidebar.expander(t('config.fuzzy_mode')):
        generate_fuzzy = st.checkbox(
            t('config.enable_fuzzy'),
            value=default_config['fitter']['generate_fuzzy'],
            help=t('config.fuzzy_help')
        )
        
        fuzzy_translator = None
        if generate_fuzzy and available_models:
            # 优先使用 Gemini_2.5_Flash
            default_idx = available_models.index('Gemini_2.5_Flash') if 'Gemini_2.5_Flash' in available_models else 0
            fuzzy_translator = st.selectbox(
                t('config.fuzzy_translator'),
                options=available_models,
                index=default_idx,
                help=t('config.translator_help')
            )
    
    # 优化参数配置
    with st.sidebar.expander(t('config.optimization_params')):
        enable_mutation = st.checkbox(
            t('config.enable_mutation'),
            value=default_config['fitter'].get('enable_mutation', False),
            help=t('config.mutation_help')
        )
        
        enable_crossover = st.checkbox(
            t('config.enable_crossover'),
            value=default_config['fitter'].get('enable_crossover', False),
            help=t('config.crossover_help')
        )
        
        optimization_method = st.selectbox(
            t('config.optimization_method'),
            options=["L-BFGS-B", "differential-evolution"],
            index=0 if default_config['fitter'].get('optimization_method', 'L-BFGS-B') == 'L-BFGS-B' else 1,
            help=t('config.method_help')
        )
        
        optimization_trial_num = st.number_input(
            t('config.optimization_trial_num'),
            min_value=1,
            max_value=1000,
            value=default_config['fitter'].get('optimization_trial_num', 5),
            help=t('config.trial_help')
        )

    # 数据处理配置
    with st.sidebar.expander(t('config.data_processing')):
        num_points = st.number_input(
            t('config.num_points'),
            min_value=10,
            max_value=1000,
            value=default_config['data']['num_points'],
            help=t('config.points_help')
        )
        
        smooth_data = st.checkbox(
            t('config.smooth_data'),
            value=default_config['data']['smooth_data'],
            help=t('config.smooth_help')
        )
        
        interpolate_data = st.checkbox(
            t('config.interpolate_data'),
            value=default_config['data']['interpolate_data'],
            help=t('config.interpolate_help')
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
