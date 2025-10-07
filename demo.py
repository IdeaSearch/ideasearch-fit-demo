import sys
import numpy as np
from os.path import sep as seperator
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter
import time
import os
import json

try:
    from load_data import load_data
except ImportError as e:
    print(f"Failed to import necessary modules. Please check the paths in sys.path. Error: {e}")
    sys.exit(1)

with open('/data/sonny/ideas/ideasearch-fitter/data/path.py', 'r') as f:
    path_file = f.read()
    path_space = {}
    exec(path_file, path_space)
    f.close()
        
idesearch_result_path = path_space.get('idesearch_result_path')[-1]
os.makedirs(idesearch_result_path,exist_ok=True)
api_path = path_space.get('api_path')
models = ['gemini-pro', 'gpt-5-mini', 'gpt-5', 'qwen3', 'qwen-plus', 'gemini-2.5-flash', 'deepseek-v3', 'grok-4', 'doubao', 'gemini-2.5-pro']
translator = "gemini-2.5-flash"
ok_score = 79.9

def run_ideasearch(data_name, noise = 0, output_dir=idesearch_result_path, models=models):
    
    """运行IdeaSearch拟合指定数据集，并将结果保存到指定目录"""
    use_fuzzy = True
    output_dir = output_dir + f'/ideasearch_unit/noise_{noise}/' + data_name
    os.makedirs(output_dir,exist_ok=True)

    try:
        x, y, yerr, info = load_data(data_name, noise_level = noise)
    except Exception as e:
        print(e)
        return (data_name, f"失败", f'加载数据出错：{e}')
    variable_names = info['feature_names']
    variable_units = info['feature_units']
    output_name = info['target_name']
    output_unit = info['target_unit']
    # print(info.get('variable_descriptions', None))
    # assert info.get('variable_descriptions', None) is not None

    # 创建 IdeaSearcher 和 IdeaSearchFitter 的实例
    ideasearcher = IdeaSearcher()
    if yerr is not None:
        data = {"x": x, "y": y, "error": yerr,}
    else:
        data = {"x": x, "y": y, }
    if output_unit is not None and variable_units is not None:
        ideasearch_fitter = IdeaSearchFitter(data = data,
                          metric_mapping='logarithm',
                          baseline_metric_value=1e6, # 20 points
                          good_metric_value=1, # 80 points
                          perform_unit_validation = True, 
                          variable_names = variable_names, # e.g., ['m','a']
                          variable_units = variable_units, # ['m/s^2','m/s^2']
                          output_name = output_name,
                          # 'F'
                          output_unit = output_unit,
                          # 'kg m/s^2'
                          # auto_describe = True,
                          variable_descreption = info.get('variable_descriptions', None),
                          # {'m': '质量', 'a': '加速度'}
                          output_descreption = info.get('target_description', None),
                          # '力'
                          input_description = info.get('input_description', None),
                          # 'use物体的质量和加速度to derive the作用在物体上的力'
                          generate_fuzzy = use_fuzzy,
                          translator_name = translator,
                          constant_whitelist=['1','2','pi'],
                          constant_map={'1':1, '2':2, 'pi':np.pi},
                          )
    else:
        ideasearch_fitter = IdeaSearchFitter(data = data,
                          perform_unit_validation = False)

    file_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # 首先设置数据库路径，ideasearcher 对文件系统的一切操作都会在数据库中进行
    ideasearcher.set_database_path(f"{output_dir}{seperator}{file_name}")
    # 绑定 ideasearch_fitter
    # prompt、评估函数和 initial_ideas 都会外包给 ideasearch_fitter
    ideasearcher.bind_helper(ideasearch_fitter)
    
    # 在 ideasearcher 中设置其它必要和可选参数
    ideasearcher.set_program_name(f"IdeaSearch-fit {data_name}")
    
    # 可注释
    # --------------------------------------
    ideasearcher.set_samplers_num(3)
    ideasearcher.set_sample_temperature(20.0)
    ideasearcher.set_hand_over_threshold(-0.1)
    ideasearcher.set_evaluators_num(3)
    ideasearcher.set_examples_num(2)
    ideasearcher.set_generate_num(2)
    ideasearcher.set_record_prompt_in_diary(True)
    ideasearcher.set_shutdown_score(ok_score)
    # --------------------------------------

    ideasearcher.set_api_keys_path(api_path)
    ideasearcher.set_model_assess_average_order(15.0)
    ideasearcher.set_model_assess_initial_score(20.0)
    ideasearcher.set_models(models)
    #ideasearcher.set_model_temperatures(temperatures)

    # 开始 IdeaSearch
    island_num = 8
    cycle_num = 5
    unit_interaction_num = 6
    
    for _ in range(island_num):
        ideasearcher.add_island()
    
    flag = False
    for cycle in range(cycle_num):
        # ideasearcher.set_models(models[:cycle*len(models)//cycle_num])
        # ideasearcher.set_models(temperatures[:cycle*len(models)//cycle_num])
        if cycle % 3 == 0:
            ideasearcher.set_filter_func(lambda idea: idea)
        else:
            ideasearcher.set_filter_func(lambda idea: "")

        if cycle != 0:
            ideasearcher.repopulate_islands()
    
        ideasearcher.run(unit_interaction_num)
        
        # 通过 get_bset_fit 动作实时查看最优拟合函数
        best_fun = ideasearch_fitter.get_best_fit()
        print(best_fun)

        best_score = ideasearcher.get_best_score()
        if best_score >= ok_score:
            print(f"达到预设分数 {best_score}，提前结束")
            flag = True
            break
        
    if flag:
        return (data_name, f"成功", f'最佳函数：{best_fun}')
    else:
        return (data_name, f"失败", f'最佳函数：{best_fun}')
    
if __name__ == "__main__":
    data_name = "I.12.2"
    run_ideasearch(data_name,noise=0.01)