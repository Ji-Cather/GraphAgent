import os
import shutil
import json
import yaml
import time
import openai
import multiprocessing
from LLMGraph.utils.io import readinfo,writeinfo


def start_launchers(launcher_num:int =8,
                    launcher_save_paths = [
                  "LLMGraph/llms/launcher_info.json"
              ]):
    command_template = "python start_launchers.py --launcher_num {launcher_num} --launcher_save_path {launcher_save_path}"
    
    # 创建多个进程，每个进程执行函数一次，并传入不同的参数
    processes = []
    for launcher_save_path in launcher_save_paths:
        command = command_template.format(launcher_num = launcher_num,
                                            launcher_save_path = launcher_save_path)

        p = multiprocessing.Process(target=os.system, 
                                    args=(command,))
        processes.append(p)
        p.start()
    
    # 等待所有进程执行结束
    for p in processes:
        p.join()

def run_tasks(configs,
              task_name,
              log_dir,
              run_ex_times = 1,
              launcher_save_paths = [
                  "LLMGraph/llms/launcher_info.json"
              ]):

    assert len(configs)==len(launcher_save_paths), "len not equal for launcher_save_paths"
    
    
    command_template = "python main.py --task {task} --config {config} --build --launcher_save_path {launcher_save_path}"
    # command_template = "python main.py --task {task} --config {config} --save --launcher_save_path {launcher_save_path}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_task_dir = os.path.join(log_dir,"log")
    if not os.path.exists(log_task_dir):
        os.makedirs(log_task_dir)

    complete_path = os.path.join(log_dir,"complete.json")            
    
    # 创建多个进程，每个进程执行函数一次，并传入不同的参数
    processes = []
    for idx, command_info in enumerate(zip(configs,launcher_save_paths)):
        config,launcher_save_path = command_info
        # launcher_save_path = "LLMGraph/llms/launcher_info_none.json"

        # test_article_num >= 500
        path = f"LLMGraph/tasks/{task_name}/configs/{config}/data/log_info.json"
        if os.path.exists(path):
            log_info = readinfo(path)
            # if log_info["generated_articles"] >= 500:
            #     print("finished",config)
            #     continue

        command = command_template.format(config = config,
                                        task = task_name,
                                        launcher_save_path = launcher_save_path
                                        )
        p = multiprocessing.Process(target=os.system, 
                                    args=(command,))
        processes.append(p)
        p.start()

    # 等待所有进程执行结束
    success_configs = []
    failed_configs = []
    for config,p in zip(configs,processes):
        p.join()
        if p.exitcode == 0:
            success_configs.append(config)
        else:
            failed_configs.append(config)
    with open(complete_path,'w',encoding = 'utf-8') as f:
        json.dump({"success":success_configs,
                "failed":failed_configs}, 
                f,
                indent=4,
                separators=(',', ':'),ensure_ascii=False)
        
def run_tasks_logs(data = "PHA_51tenant_5community_28house",
                   configs:list = None
                   ):
    config_root = f"LLMGraph/tasks/{data}/configs"
    
    configs = os.listdir(config_root) if configs is None else configs
    
    
    command_template = "python main.py --task {task} --data {data} --log {log}"
    
    count = {}
    for config in configs:
        
        
        result_path = os.path.join(config_root,config,"result")
        config = config.replace("(","\(").replace(")","\)")
        
        if os.path.exists(result_path):
            # paths.append(os.path.join(result_path,os.listdir(result_path)[-1]))
            result_files = os.listdir(result_path)
            paths = []
            for result_file in result_files:
                # if os.path.exists(os.path.join(result_path,result_file,"tenental_system.json")) \ 
                # and os.path.exists(os.path.join(result_path,result_file,"all")):
                if os.path.exists(os.path.join(result_path,result_file,"tenental_system.json")):
                # result_file_path = os.path.join(result_path,result_file,"all")
                # if os.path.exists(result_file_path):
                    paths.append(os.path.join(result_path,result_file))
            for path in paths:
                path = path.replace("(","\(").replace(")","\)")
                command = command_template.format(task = config,
                                                  data = data,
                                                  log = path)
                try:
                    return_val = os.system(command)
                except Exception as e:
                    print(e)
        count[config]=paths
                
                
    # print(count)
    
    # print(len(count))
                    
            
def test_task_logs(data ="PHA_51tenant_5community_28house",
                   ):
    
    config_root = f"LLMGraph/tasks/{data}/configs"
    
    configs = os.listdir(config_root)
  
    not_available_results =[]
    
    for config in configs:
        
        result_path = os.path.join(config_root,config,"result")
        
        if os.path.exists(result_path):
            # paths.append(os.path.join(result_path,os.listdir(result_path)[-1]))
            result_files = os.listdir(result_path)
            paths = []
            ok = False
            for result_file in result_files:
                if os.path.exists(os.path.join(result_path,result_file,"tenental_system.json")):
                    tenental_info = readinfo(os.path.join(result_path,result_file,"tenental_system.json"))
                    last_round = list(tenental_info.keys())[-1]
                    try:
                        if (int(last_round)>=9):
                            ok = True
                    except:
                        pass
            if (not ok):not_available_results.append([config,list(tenental_info.keys())[-1]])
                        
    with open("LLMGraph/tasks/PHA_51tenant_5community_28house/cache/not_available_tasks.json",
              'w',encoding = 'utf-8') as f:
        json.dump(not_available_results, f, indent=4,separators=(',', ':'),ensure_ascii=False)
    
    

            
            
def set_data_configs(data):
    task_dir ="LLMGraph/tasks"
    
    config_root = os.path.join(task_dir,data,"configs")
    task_names = os.listdir(config_root)


    # task_names = list(filter(lambda x: 
    #     not os.path.exists(os.path.join(config_root,x,"result")),
    #                          task_names))
    
    dirs = {
        "house":"",
        "tenant":"",
        "forum":"",
        "community":""
    }
    
    distribution_batch_dir={
        "tenant":"",
        "community":""
    }
    
    data_files = os.listdir(os.path.join(task_dir,data,"data"))
    
    data_files = list(filter(lambda x:x!="visualize",data_files))
    
    for data_type  in dirs.keys():
        for data_file in data_files:
            if (data_type in data_file):
                dirs[data_type] = os.path.join("data",data_file)
                break
            
    
    
    for task_name in task_names:
        config_path = os.path.join(config_root,task_name,"config.yaml")
        task_config = yaml.safe_load(open(config_path))
        
        """default k"""
        if task_config["environment"]["rule"]["order"]["type"] == "kwaitlist":
            if "k" not in task_config["environment"]["rule"]["order"].keys():
                task_config["environment"]["rule"]["order"]["k"] = 2
            if "waitlist_ratio" not in task_config["environment"]["rule"]["order"].keys():
                task_config["environment"]["rule"]["order"]["waitlist_ratio"] = 1.2
                
        """communication_num"""
        task_config["environment"]["communication_num"] = 10
        
       
        
        distribution_data_paths = os.listdir(os.path.join(config_root,task_name,"data"))
        for data_path in distribution_data_paths:
            if "tenant" in data_path:
                distribution_batch_dir["tenant"] =  os.path.join("data",data_path)
            else: distribution_batch_dir["community"] = os.path.join("data",data_path)
        
        for data_type,data_dir in dirs.items():
            task_config["managers"][data_type]["data_dir"] = data_dir
        
        for distribution_key,distribution_path in distribution_batch_dir.items():
            task_config["managers"][distribution_key]["distribution_batch_dir"] = distribution_path

            
        task_config["name"] = task_name
        with open(config_path, 'w') as outfile:
            yaml.dump(task_config, outfile, default_flow_style=False)
    


def clear_experiment_cache(
                    configs,
                    task_name,):
    task_root = "LLMGraph/tasks"
    task_root = os.path.join(task_root,task_name)
    file_names =[
        "article_meta_info.pt",
        "author.pt"
    ]
    # configs_all = os.listdir(os.path.join(task_root,"configs"))
    # configs_no = list(filter(lambda x: x not in configs,configs_all))
    # for config in configs_no:
    #     config_path = os.path.join(task_root,"configs",config)
    #     shutil.rmtree(config_path)


    for config in configs:
        config_path = os.path.join(task_root,"configs",config)
        if not os.path.exists(config_path):
            print(config_path)
            continue
        data_dst = os.path.join(config_path,"data")
        data_src = os.path.join(task_root,"data")
        config_file = yaml.safe_load(open(os.path.join(config_path,"config.yaml")))
        config_file["environment"]["article_write_configs"]["use_graph_deg"] = True
        with open(os.path.join(config_path,"config.yaml"), 'w') as outfile:
            yaml.dump(config_file, outfile, default_flow_style=False)

        if os.path.exists(data_dst):
            shutil.rmtree(data_dst)
        if os.path.exists(os.path.join(config_path,"evaluate")):
            shutil.rmtree(os.path.join(config_path,"evaluate"))
        os.makedirs(data_dst)
        for file_name in file_names:
            shutil.copyfile(os.path.join(data_src,file_name),
                            os.path.join(data_dst,file_name))
    
def modify_config_name_info(task_name,
                            configs):
    task_root = "LLMGraph/tasks"
    task_root = os.path.join(task_root,task_name)
    for config in configs:
        config_path = os.path.join(task_root,"configs",config)
        if not os.path.exists(config_path):
            print(config_path)
            continue
        data_dst = os.path.join(config_path,"data")
        article_meta_info = readinfo(os.path.join(data_dst,"article_meta_info.pt"))
        for article in article_meta_info.values():
            if "/llm_agent/" in article["path"]:
                article["path"] = article["path"].replace("/llm_agent/",f"/{task_name}/")
            # if "/fast_gpt4o/" in article["path"]:
            #     article["path"] = article["path"].replace("/fast_gpt4o/",f"/{config}/")

        writeinfo(os.path.join(data_dst,"article_meta_info.pt"),article_meta_info)


def copy_modify_config(dst_llm, 
                       src_llm,
                       task_name,
                       config_templates):
    llm_map ={
        "gpt3.5":"gpt-3.5-turbo-0125",
        "vllm":"vllm",
        "gpt4-mini":"gpt-4o-mini",
        "qwen2":"qwen2",
        "gemini-1.5-flash":"gemini-1.5-flash",
        "mixtral":"mixtral"
    }

    task_root = "LLMGraph/tasks"
    task_root = os.path.join(task_root,task_name)
    file_names =[
        "article_meta_info.pt",
        "author.pt"
    ]

    for config_template in config_templates:
        src_config_path = os.path.join(task_root,
                                       "configs",
                                   config_template.format(llm = src_llm)
                                   )
        dst_config_path = os.path.join(task_root,
                                       "configs",
                                       config_template.format(llm = dst_llm))
        if os.path.exists(dst_config_path):
            continue
        shutil.copytree(src_config_path,dst_config_path)
        data_dst = os.path.join(dst_config_path,"data")
        data_src = os.path.join(task_root,"data")
        config_file = yaml.safe_load(open(os.path.join(dst_config_path,"config.yaml")))
        config_file["environment"]["article_write_configs"]["use_graph_deg"] = True
        config_file["environment"]["managers"]["article"]["model_config_name"] = \
            llm_map.get(dst_llm)
        config_file["environment"]["agent"]["llm"]["config_name"] = \
            llm_map.get(dst_llm)

        with open(os.path.join(dst_config_path,"config.yaml"), 'w') as outfile:
            yaml.dump(config_file, outfile, default_flow_style=False)

        if os.path.exists(data_dst):
            shutil.rmtree(data_dst)
        if os.path.exists(os.path.join(dst_config_path,"evaluate")):
            shutil.rmtree(os.path.join(dst_config_path,"evaluate"))
        os.makedirs(data_dst)
        for file_name in file_names:
            shutil.copyfile(os.path.join(data_src,file_name),
                            os.path.join(data_dst,file_name))


    
if __name__ == "__main__":
    
    task_dir ="LLMGraph/tasks"

    
    """article"""

    task_names = [f"llm_agent_{i}" for i in range(1,6)]
    # task_name = "llm_agent_retry"
    # task_name = "citeseer"

    llms =[
        # "gpt3.5", 
        # "gpt4-mini",
        "vllm",
        # "qwen2",
        # "gemini-1.5-flash",
        # "mixtral"
           ]
    config_templates =[
            # "search_shuffle_base_{llm}",
            # # "search_shuffle_nocite_{llm}",
            # # "search_shuffle_no_content_{llm}",
            # "search_shuffle_no_country_{llm}",
            # "search_shuffle_noauthor_{llm}",
            # "search_shuffle_nopapertime_{llm}",
            # 'search_shuffle_base_noauthorcite_{llm}', 
            # 'search_shuffle_notopic_{llm}', 
            # 'search_shuffle_noauthortopic_{llm}', 
            # "search_shuffle_anonymous_{llm}",
            # "search_shuffle_base_{llm}_2engine",
            # "nosearch_shuffle_base_{llm}",
            # "search_shuffle_base_nosocial_{llm}",
            # "search_no_country_info_{llm}",
            # "search_shuffle_no_author_country_{llm}"
    ]

    # task_names =["citeseer"]
    # config_templates =[
    #     # "fast_{llm}",
    #     "fast_{llm}_2",
    # ]
    # task_names =["cora"]
    # config_templates =[
    #     "fast_{llm}",
    #     "fast_{llm}_2",
    # ]

    configs = []
    
    for llm in llms:
        configs.extend([config_template.format(llm = llm) 
                        for config_template in config_templates])
    
    

   

    # configs =[
    #     "rpc4",
    #     "rpc24",
    #     "rpc48"
    # ]
    
    """movie"""
    # task_name = "movielens"
    # configs = [
    #     # "filter_1",
    #     # "filter_all",
    #     # "filter_all_r",
    #     # "filter_all_k3",
    #     # "filter_all_k5",
    #     "filter_0",
    #     # "filter_all_k10",
    # ]

    """social"""
    # task_name = "tweets"
    # configs = [
    # # "filter_1",
    # # "filter_2",
    # # "filter_3",
    # # "filter_3item",
    # # "filter_2item",
    # # "filter_4",
    # # "filter_all",
    # # "filter_all_k3",
    # # "filter_all_k5",
    # # "filter_all_k10",
    # # "filter_all_r0",
    # # "filter_all_r0.1",
    # # "filter_0",   
    # "llama_test_1e6",
    # ]
    

    # task_name = "llm_agent"
    # configs = [
    # ]

    task_names = ["citeseer"]
    configs = ["fast_gpt3.5_0.1k"]
    start_round = 0
    end_round = 1
    prefix = 0
    launcher_save_paths = []
    for task_name in task_names[start_round:end_round]:
        
        server_num = len(configs)
        launcher_save_paths.extend([f"LLMGraph/llms/launcher_filter_{i}.json" for i in range(prefix+1,prefix+server_num+1)])
        prefix += server_num

    # start_launchers(10,launcher_save_paths)

    for idx, task_name in enumerate(task_names[start_round:end_round]):        
        server_num = len(configs)   
        launcher_save_paths_group = launcher_save_paths[idx*server_num:(idx+1)*server_num]
        log_dir = f"LLMGraph/tasks/{task_name}/cache"
        run_tasks(configs,
                  task_name,
                  log_dir,
                  run_ex_times = 1,
                  launcher_save_paths=launcher_save_paths_group)
        """ 请谨慎执行，确保备份 """
        # clear_experiment_cache(configs,task_name)
        # copy_modify_config("gpt3.5","gpt4-mini",task_name,config_templates)
        # copy_modify_config("gpt4-mini","gpt3.5",task_name,config_templates)
        # copy_modify_config("qwen2","gpt3.5",task_name,config_templates)
        # copy_modify_config("vllm","gpt3.5",task_name,config_templates)
        # copy_modify_config("gemini-1.5-flash","gpt3.5",task_name,config_templates)
        # copy_modify_config("mixtral","gpt3.5",task_name,config_templates)
        # copy_modify_config("gpt3.5","vllm",task_name,config_templates)
        
        # modify_config_name_info(task_name,configs)
        