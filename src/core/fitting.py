"""
拟合核心逻辑
封装 IdeaSearch 和 IdeaSearchFitter 的调用
"""

import os
import sys
import re
import time
import base64
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter


def print_flush(*args, **kwargs):
    """打印并立即刷新输出到控制台"""
    print(*args, **kwargs)
    sys.stdout.flush()


class FittingEngine:
    """拟合引擎，管理 IdeaSearch 拟合流程"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化拟合引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 确保 model_temperatures存在且长度与models一致
        if 'model_temperatures' not in config or not config['model_temperatures']:
            config['model_temperatures'] = [1.0] * len(config.get('models', []))
        elif len(config['model_temperatures']) < len(config.get('models', [])):
            # 如果温度数量不足，用 1.0 填充
            config['model_temperatures'].extend([1.0] * (len(config['models']) - len(config['model_temperatures'])))
        
        self.fitter: Optional[IdeaSearchFitter] = None
        self.ideasearcher: Optional[IdeaSearcher] = None
        self.result_path: Optional[str] = None
        self.database_path: Optional[str] = None
        
        # 状态跟踪
        self.is_running = False
        self.should_stop = False
        self.current_cycle = 0
        self.total_cycles = 0
        self.best_score = 0.0
        self.best_expression = ""
        self.total_api_calls = 0
        self.start_time = 0.0
        self.api_calls_log: List[Dict[str, Any]] = []
        self.score_history: List[float] = []
        # 版本与进度帧（用于与前端同步每次 epoch 打印）
        self.state_version: int = 0
        self.progress_frames: List[Dict[str, Any]] = []
        self.final_best: Optional[Dict[str, Any]] = None
        
        # 回调函数
        self.on_progress_update: Optional[Callable] = None
        self.on_api_call: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
    
    def prepare_data(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        准备数据格式
        
        Args:
            x: x 数据数组
            y: y 数据数组
        
        Returns:
            数据字典
        """
        # 确保 x 是 2D 数组
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        return {"x": x, "y": y}
    
    def initialize_fitter(self, x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray] = None) -> None:
        """
        初始化 IdeaSearchFitter
        
        Args:
            x: x 数据数组
            y: y 数据数组
            yerr: 误差数据数组（可选）
        """
        # 修改为固定目录：代码文件夹下的 logs 文件夹
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # 使用时间戳创建唯一的子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_path = str(logs_dir / f"fit_{timestamp}")
        os.makedirs(self.result_path, exist_ok=True)
        
        print_flush(f"\n📁 日志目录: {self.result_path}")
        
        # 准备数据
        data = self.prepare_data(x, y)
        if yerr is not None:
            data['error'] = yerr
        
        # 创建 IdeaSearchFitter
        fuzzy_translator = self.config.get('fuzzy_translator')
        
        # 获取fitter参数（如果有的话）
        fitter_params = self.config.get('fitter_params', {})
        
        # 打印 Fuzzy 模式配置信息
        print_flush(f"\n{'='*60}")
        print_flush("IdeaSearchFitter 配置:")
        print_flush(f"  - generate_fuzzy: {self.config['generate_fuzzy']}")
        print_flush(f"  - fuzzy_translator: {fuzzy_translator}")
        print_flush(f"  - auto_polish: True")
        print_flush(f"  - auto_polisher: {fuzzy_translator}")
        if fitter_params:
            print_flush(f"  - perform_unit_validation: {fitter_params.get('perform_unit_validation', False)}")
            print_flush(f"  - variable_names: {fitter_params.get('variable_names', ['x1'])}")
        print_flush(f"{'='*60}\n")
        
        # 构建IdeaSearchFitter参数
        fitter_kwargs = {
            'result_path': self.result_path,
            'data': data,
            'functions': self.config['functions'],
            'constant_whitelist': self.config['constant_whitelist'],
            'constant_map': self.config['constant_map'],
            'perform_unit_validation': fitter_params.get('perform_unit_validation', False),
            'auto_polish': True,
            'auto_polisher': fuzzy_translator,
            'generate_fuzzy': self.config['generate_fuzzy'],
            'fuzzy_translator': fuzzy_translator,
            'metric_mapping': "logarithm",
            'baseline_metric_value': 1.0e6,
            'good_metric_value': 1.0,
            # 新增的优化参数
            'enable_mutation': self.config.get('enable_mutation', False),
            'enable_crossover': self.config.get('enable_crossover', False),
            'optimization_method': self.config.get('optimization_method', 'L-BFGS-B'),
            'optimization_trial_num': self.config.get('optimization_trial_num', 5),
        }
        
        # 如果有自定义fitter参数，则使用它们
        if fitter_params:
            if 'variable_names' in fitter_params:
                fitter_kwargs['variable_names'] = fitter_params['variable_names']
            if 'variable_units' in fitter_params:
                fitter_kwargs['variable_units'] = fitter_params['variable_units']
            if 'output_name' in fitter_params:
                fitter_kwargs['output_name'] = fitter_params['output_name']
            if 'output_unit' in fitter_params:
                fitter_kwargs['output_unit'] = fitter_params['output_unit']
            if 'variable_descriptions' in fitter_params:
                fitter_kwargs['variable_descriptions'] = fitter_params['variable_descriptions']
            if 'output_description' in fitter_params:
                fitter_kwargs['output_description'] = fitter_params['output_description']
            if 'input_description' in fitter_params:
                fitter_kwargs['input_description'] = fitter_params['input_description']
        else:
            # 默认参数（用于画布拟合）
            fitter_kwargs.update({
                'input_description': "用户绘制的曲线，表示 x 和 y 之间的关系",
                'variable_descriptions': {"x1": "自变量 x，表示输入值"},
                'variable_names': ["x1"],
                'output_description': "因变量 y，与 x 存在某种数学关系",
                'output_name': "y",
            })
        
        self.fitter = IdeaSearchFitter(**fitter_kwargs)
    
    def initialize_searcher(self, canvas_image: Optional[str] = None) -> None:
        """
        初始化 IdeaSearcher
        
        Args:
            canvas_image: base64 编码的画布图片（可选）
        """
        if self.fitter is None:
            raise RuntimeError("必须先初始化 IdeaSearchFitter")
        
        # 修改为固定目录：代码文件夹下的 logs 文件夹
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # 使用时间戳创建唯一的子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.database_path = str(logs_dir / f"db_{timestamp}")
        os.makedirs(self.database_path, exist_ok=True)
        
        print_flush(f"📁 数据库目录: {self.database_path}\n")
        
        # 创建 IdeaSearcher
        self.ideasearcher = IdeaSearcher()
        
        # 设置数据库路径
        self.ideasearcher.set_database_path(self.database_path)
        
        # 绑定 fitter
        self.ideasearcher.bind_helper(self.fitter)
        
        # 如果提供了画布图片，则设置图片和修改提示词
        if canvas_image is not None:
            print_flush(f"📷 画布图片已设置（长度: {len(canvas_image)} 字符）\n")
            # base64 decoded image string
            canvas_image_decoded = base64.b64decode(canvas_image)
            self.ideasearcher.set_images([canvas_image_decoded])
            original_epilogue_section = self.ideasearcher.get_epilogue_section()
            epilogue_section_with_image = "Here is the image showing the function curve you are going to fit: <image>\n" + original_epilogue_section
            self.ideasearcher.set_epilogue_section(epilogue_section_with_image)
        
        # 设置参数
        self.ideasearcher.set_program_name("IdeaSearch Streamlit App")
        self.ideasearcher.set_samplers_num(self.config['samplers_num'])
        self.ideasearcher.set_sample_temperature(self.config['sample_temperature'])
        self.ideasearcher.set_hand_over_threshold(self.config['hand_over_threshold'])
        self.ideasearcher.set_evaluators_num(self.config['evaluators_num'])
        self.ideasearcher.set_examples_num(self.config['examples_num'])
        self.ideasearcher.set_generate_num(self.config['generate_num'])
        self.ideasearcher.set_shutdown_score(self.config['shutdown_score'])
        self.ideasearcher.set_record_prompt_in_diary(self.config.get('record_prompt_in_diary', True))
        
        # 设置 API 和模型配置
        self.ideasearcher.set_api_keys_path(self.config['api_keys_path'])
        self.ideasearcher.set_model_assess_average_order(self.config.get('model_assess_average_order', 15.0))
        self.ideasearcher.set_model_assess_initial_score(self.config.get('model_assess_initial_score', 20.0))
        self.ideasearcher.set_models(self.config['models'])
        self.ideasearcher.set_model_temperatures(self.config['model_temperatures'])  # 关键！
        
        # 打印配置信息
        print_flush(f"\n{'='*60}")
        print_flush("IdeaSearcher 配置:")
        print_flush(f"  - API 密钥路径: {self.config['api_keys_path']}")
        print_flush(f"  - 模型: {self.config['models']}")
        print_flush(f"  - 模型温度: {self.config['model_temperatures']}")
        print_flush(f"  - 岛屿数: {self.config['island_num']}")
        print_flush(f"  - 采样器数: {self.config['samplers_num']}")
        print_flush(f"  - 评估器数: {self.config['evaluators_num']}")
        print_flush(f"{'='*60}\n")
        
        # 添加岛屿
        for _ in range(self.config['island_num']):
            self.ideasearcher.add_island()
    
    def run_fitting(self, x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray] = None, canvas_image: Optional[str] = None) -> None:
        """
        执行拟合
        
        Args:
            x: x 数据数组
            y: y 数据数组
            yerr: 误差数据数组（可选）
            canvas_image: base64 编码的画布图片（可选）
        """
        try:
            # 初始化
            self.is_running = True
            self.should_stop = False
            self.current_cycle = 0
            self.total_cycles = self.config['cycle_num']
            self.start_time = time.time()
            self.api_calls_log = []
            self.score_history = []
            
            # 初始化 fitter 和 searcher
            self.initialize_fitter(x, y, yerr)
            self.initialize_searcher(canvas_image)
            
            print(f"\n🚀 开始拟合 - 目标循环数: {self.total_cycles}\n")
            sys.stdout.flush()
            
            # 运行循环
            for cycle in range(self.config['cycle_num']):
                if self.should_stop:
                    print_flush("\n⏹️  拟合已手动停止\n")
                    break
                
                self.current_cycle = cycle + 1
                
                # 动态设置 filter_func（参考演示代码）
                # 每3个循环切换一次过滤策略
                if cycle % 3 == 0:
                    self.ideasearcher.set_filter_func(lambda idea: idea)  # 保留想法
                else:
                    self.ideasearcher.set_filter_func(lambda idea: "")  # 清空想法（强制生成新的）
                
                # 重新填充岛屿（第一次循环除外）
                if cycle != 0:
                    self.ideasearcher.repopulate_islands()
                
                print(f"⏳ 循环 {self.current_cycle}/{self.total_cycles} 进行中...")
                sys.stdout.flush()
                
                # 分批运行 epoch 以实现实时更新
                # 每批运行较少的epoch，运行后立即更新状态，这样界面可以实时刷新
                epochs_per_batch = 1  # 每批运行1个epoch，确保频繁更新
                total_epochs = self.config['unit_interaction_num']
                
                for batch in range(total_epochs):
                    if self.should_stop:
                        print_flush("\n⏹️  拟合已手动停止\n")
                        break
                    
                    try:
                        # 运行一批epoch
                        print_flush(f"  🔄 执行 Epoch {batch + 1}/{total_epochs}...")
                        self.ideasearcher.run(epochs_per_batch)
                        
                        # 立即更新最佳结果
                        log_message = ""
                        try:
                            self.best_expression = self.fitter.get_best_fit()
                            self.best_score = self.ideasearcher.get_best_score()
                            
                            # 简洁的终端输出
                            progress = (batch + 1) / total_epochs * 100
                            log_message = f"  ✅ Epoch {batch + 1}/{total_epochs} ({progress:.0f}%) | 分数: {self.best_score:.4f} | 表达式: {self.best_expression[:50]}{'...' if len(self.best_expression) > 50 else ''}"
                            print_flush(log_message)
                        except Exception as e:
                            log_message = f"  ⚠️ Epoch {batch + 1}/{total_epochs} | 获取结果出错: {e}"
                            print_flush(log_message)
                        
                        # 更新 API 调用日志
                        self._update_api_calls()
                        
                        # 记录进度帧并增加版本号（用于前端精准同步）
                        try:
                            frame = {
                                'version': self.state_version + 1,
                                'cycle': self.current_cycle,
                                'epoch': batch + 1,
                                'total_epochs': total_epochs,
                                'score': float(self.best_score) if self.best_score is not None else None,
                                'expression': self.best_expression,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'log_message': log_message  # 记录日志消息
                            }
                            self.progress_frames.append(frame)
                            # 限制内存占用：仅保留最近的 2000 帧
                            if len(self.progress_frames) > 2000:
                                self.progress_frames = self.progress_frames[-2000:]
                            self.state_version += 1
                        except Exception:
                            pass

                        # 触发进度更新回调（关键！实时更新界面）
                        if self.on_progress_update:
                            self.on_progress_update(self.get_state())
                        
                        # 检查是否达到目标分数
                        if self.best_score >= self.config['shutdown_score']:
                            print_flush(f"\n🎯 达到目标分数 {self.config['shutdown_score']}，提前结束\n")
                            break
                            
                    except KeyboardInterrupt:
                        print_flush("\n⚠️ 用户中断，停止拟合\n")
                        self.should_stop = True
                        break
                    except Exception as e:
                        # 捕获异常但不中断，继续下一个epoch
                        error_msg = f"  ❌ Epoch {batch + 1} 执行出错: {e}"
                        print_flush(error_msg)
                        print_flush("  ⏭️  跳过此epoch，继续执行...")
                        # 继续下一个epoch
                        continue
                
                # 记录当前循环的分数历史
                if self.best_score > 0:
                    self.score_history.append(self.best_score)
                
                # 循环完成后的总结输出
                print(f"✅ 循环 {self.current_cycle} 完成 | 最终分数: {self.best_score:.4f}\n")
                sys.stdout.flush()
                
                # 如果已达到目标分数，退出循环
                if self.best_score >= self.config['shutdown_score']:
                    break
            
            print(f"\n✨ 拟合完成！最终分数: {self.best_score:.4f}\n")
            sys.stdout.flush()
            
            # 触发完成回调
            if self.on_complete:
                self.on_complete(self.get_state())
        
        except Exception as e:
            print(f"\n❌ 拟合过程出错: {e}\n")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        
        finally:
            self.is_running = False
            # 记录最终最优结果
            self.final_best = {
                'score': float(self.best_score) if self.best_score is not None else None,
                'expression': self.best_expression,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_cycles': self.total_cycles,
                'total_api_calls': self.total_api_calls,
            }
    
    def _update_api_calls(self) -> None:
        """更新 API 调用日志（增强版,提供更详细的调用信息）"""
        try:
            # 从 diary 文件中读取实际的 API 调用记录
            diary_path = self.ideasearcher.get_diary_path()
            if diary_path and os.path.exists(diary_path):
                with open(diary_path, 'r', encoding='utf-8') as f:
                    diary_content = f.read()
                    
                    # 统计实际的 API 调用次数（查找 get_answer 调用）
                    # 更精确的匹配模式
                    api_call_patterns = [
                        r'get_answer.*?调用成功',
                        r'API调用成功',
                        r'模型响应成功',
                    ]
                    
                    total_calls = 0
                    for pattern in api_call_patterns:
                        calls = re.findall(pattern, diary_content, re.DOTALL | re.IGNORECASE)
                        total_calls += len(calls)
                    
                    self.total_api_calls = total_calls
                    
                    # 提取最近的API调用详情
                    recent_calls = re.findall(
                        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?模型[：:]?\s*(\S+).*?(?:调用成功|响应成功)',
                        diary_content,
                        re.DOTALL | re.IGNORECASE
                    )
                    
                    # 打印API调用统计
                    if total_calls > 0 and total_calls % 10 == 0:  # 每10次调用打印一次
                        print_flush(f"  📊 累计API调用: {total_calls} 次")
                    
        except Exception as e:
            # 如果读取失败，使用估算值
            print_flush(f"  ⚠️ 读取API日志失败: {e}, 使用估算值")
            self.total_api_calls += self.config['unit_interaction_num'] * self.config['island_num']
        
        # 无论是否有 best_expression，都记录当前循环的状态
        # 这样即使没有找到有效表达式，也会有调用记录显示
        expression_display = self.best_expression if hasattr(self, 'best_expression') and self.best_expression else "尚未生成有效表达式"
        score_display = self.best_score if hasattr(self, 'best_score') and self.best_score is not None else 0.0
        
        call_record = {
            'cycle': self.current_cycle,
            'model': self.config['models'][0] if self.config['models'] else 'Unknown',
            'expression': expression_display,
            'score': score_display,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_api_calls': self.total_api_calls,
            'status': 'success' if (hasattr(self, 'best_expression') and self.best_expression) else 'no_expression'
        }
        self.api_calls_log.append(call_record)
        
        # 限制日志大小
        if len(self.api_calls_log) > 1000:
            self.api_calls_log = self.api_calls_log[-1000:]
    
    def stop_fitting(self) -> None:
        """停止拟合"""
        self.should_stop = True
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            状态字典
        """
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            'is_running': self.is_running,
            'current_cycle': self.current_cycle,
            'total_cycles': self.total_cycles,
            'best_score': self.best_score,
            'best_expression': self.best_expression,
            'total_api_calls': self.total_api_calls,
            'elapsed_time': elapsed_time,
            'api_calls_log': self.api_calls_log,
            'score_history': self.score_history,
            'state_version': self.state_version,
            'last_frame': self.progress_frames[-1] if self.progress_frames else None,
        }
    
    def get_pareto_frontier(self) -> Dict[int, Dict]:
        """
        获取 Pareto 前沿数据
        
        Returns:
            Pareto 前沿字典
        """
        if self.fitter is None:
            return {}
        
        try:
            return self.fitter.get_pareto_frontier()
        except Exception:
            return {}

    def get_state_version(self) -> int:
        """
        获取当前状态版本号（每个 epoch 增加 1）
        """
        return self.state_version

    def get_progress_frames(self, since_version: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取进度帧列表

        Args:
            since_version: 若提供，只返回 version 大于该值的帧

        Returns:
            进度帧列表
        """
        if since_version is None:
            return list(self.progress_frames)
        return [f for f in self.progress_frames if f.get('version', 0) > since_version]

    def get_final_best(self) -> Optional[Dict[str, Any]]:
        """
        拟合结束后的最终最优结果
        """
        return self.final_best
    
    def evaluate_expression(self, expression: str, x: np.ndarray) -> np.ndarray:
        """
        评估表达式在给定 x 值上的输出
        
        Args:
            expression: 数学表达式
            x: x 值数组
        
        Returns:
            y 值数组
        """
        try:
            import numexpr
            
            # 准备变量字典
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            
            local_dict = {}
            # 添加所有变量
            for i in range(x.shape[1]):
                local_dict[f'x{i+1}'] = x[:, i]
            
            # 添加常量
            for const_name, const_value in self.config['constant_map'].items():
                local_dict[const_name] = const_value
            
            # 评估表达式
            y = numexpr.evaluate(expression, local_dict=local_dict)
            return y
        
        except Exception as e:
            print(f"评估表达式出错: {e}")
            return np.zeros_like(x[:, 0] if x.ndim == 2 else x)
    
    # 移除了 cleanup() 方法和 __del__() 方法
    # 日志文件将永久保存在 logs 目录中，不会自动清理
    # 用户可以手动删除旧的日志文件
