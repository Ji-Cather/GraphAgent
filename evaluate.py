import os
import shutil
import json
import yaml
import time
import openai

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list



def evaluate_tasks(configs,
              task_name,
              log_dir,
              xmin:int = 3,
              evaluate_type = "article"
              ):
    
    if task_name == "llm_agent":
        task_names = [f"llm_agent_{i}" for i in range(1,6)][1:3]
    else:
        task_names = [task_name]
    success_configs = []
    failed_configs = []
    
    command_template = "python evaluate/{evaluate_type}/main.py --task {task} --config {config} --xmin {xmin}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_task_dir = os.path.join(log_dir,"log")
    if not os.path.exists(log_task_dir):
        os.makedirs(log_task_dir)

    complete_path = os.path.join(log_dir,"complete.json")            
    
    """这边可以改成并行proc"""
    """tmux的launcher也可以改成并行proc"""
    for task_name in task_names:
        for idx, config in enumerate(configs):
        
            command = command_template.format(config = config,
                                            task = task_name,
                                            evaluate_type = evaluate_type,
                                            xmin = xmin
                                            )
            
            try:
                return_val = os.system(command)
                success_configs.append(config.replace("\(","(").replace("\)",")"))
            except Exception as e:
                print(e)
                failed_configs.append(config.replace("\(","(").replace("\)",")"))
    
            with open(complete_path,'w',encoding = 'utf-8') as f:
                uncomplete_configs = configs[idx+1:] if (idx+1)< len(configs) else []
                json.dump({"success":success_configs,
                        "failed":failed_configs,
                        "uncomplete":uncomplete_configs}, 
                        f,
                        indent=4,
                        separators=(',', ':'),ensure_ascii=False)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xmin', 
                    type=int, 
                    default=3, 
                    help='power law fit xmin')  # 添加参数
args=parser.parse_args()  # 解析参数
if __name__ == "__main__":
    
    task_dir ="LLMGraph/tasks"
    xmin = int(args.xmin)
    """article"""
    evaluate_type = "article"
    task_name = "llm_agent"
    # task_name = "llm_agent_retry"
    # configs = [
        # "filter_1",
        # "filter_3",
        # "no_big_name",
        # "bigname_t",
        # "bigname_wt",
        # "t_wt",
        # "filter_1_min15",
        # "bg_r0",
        # "bg_r0.1", 
        # # "bg_r0.2",
        # ]

    llms =[
        "gpt3.5",
        "gpt4-mini",
        "qwen2",
        "vllm",
           ]
    config_templates =[
            # "search_shuffle_base_{llm}_reason",
            # "search_shuffle_base_gpt3.5_ver1_longtime"
            "search_shuffle_base_{llm}",
            # # "search_shuffle_nocite_{llm}",
            # # "search_shuffle_no_content_{llm}",
            # "search_shuffle_no_country_{llm}",
            # # "search_shuffle_noauthor_{llm}",
            # # "search_shuffle_nopapertime_{llm}",
            # # 'search_shuffle_base_noauthorcite_{llm}', 
            # # 'search_shuffle_notopic_{llm}', 
            # # 'search_shuffle_noauthortopic_{llm}', 
            "search_shuffle_anonymous_{llm}",
            # # "nosearch_shuffle_base_{llm}",
            "search_shuffle_base_nosocial_{llm}",
            # "search_shuffle_no_author_country_{llm}"
    ]
    configs = []
    for llm in llms:
        configs.extend([config_template.format(llm = llm) 
                        for config_template in config_templates])
    

    # task_name = "llm_agent_1"
    # configs =[
    #     "gt"
    # ]
    task_name = "citeseer"
    configs =[
        # "fast_gpt3.5",
        # "fast_vllm",
        # "fast_qwen2",
        # "fast_gpt4-mini",
        # "fast_gpt4-mini_2",
        # "fast_qwen2_2",
        #  "fast_gpt3.5_2",
        # "fast_vllm_2",
        "gt"
    ]

    task_name = "cora"
    configs =[
        # "fast_gpt3.5",
        # "fast_vllm",
        # # "fast_qwen2_2",
        # "fast_qwen2",
        # # "fast_gpt3.5_2",
        # # "fast_vllm_2",
        # "fast_gpt4-mini",
        # "fast_gpt4-mini_2",
        "gt"
    ]

    # """movie"""
    # task_name = "movielens"
    # configs = [
    #     "llama_test_rpc_agent40_rpc48",
    #     "llama_test_rpc_agent40_rpc4"
    # ]

    # """movie"""
    # evaluate_type="movie"
    # task_name = "movielens"
    # configs = [
    #     "filter_1",
    #     "filter_2",
    #     "filter_all",
    #     "filter_all_r",
    #     "filter_all_k3",
    #     "filter_all_k5",
    #     "filter_all_k10",
    # ]

    # """social"""
    # evaluate_type ="social"
    # task_name = "tweets"
    # configs = [
    # # "filter_0",
    # # "filter_1",
    # # "filter_2",
    # # "filter_3",
    # # "filter_4",
    # # # "filter_all",
    # # "filter_all_k3",
    # # "filter_all_k5",
    # # "filter_all_k10",
    # # # "filter_all_r0",
    # # # "filter_all_r0.1",
    # # "filter_3"
    # "llama_test_7000_p0.0025",
    # # "llama_test_7000_p0.0025_hubFalse",
    # # "llama_test_1e6"
    # ]
    log_dir = f"LLMGraph/tasks/{task_name}/evaluate_cache"
    
    evaluate_tasks(configs,
                task_name,
                log_dir,
                xmin = xmin,
                evaluate_type=evaluate_type)