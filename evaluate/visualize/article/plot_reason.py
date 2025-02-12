from LLMGraph.utils.io import readinfo,writeinfo
import os
import networkx as nx

def normalize_dict_values(d):
    # 获取字典中的值
    values = list(d.values())
    
    # 计算最大值和最小值
    max_val = max(values)
    min_val = min(values)

    # 进行归一化
    # normalized_values = {
    #     k: (v - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0
    #     for k, v in d.items()
    # }
    normalized_values = {
        k: v/ sum(values) if max_val - min_val != 0 else 0
        for k, v in d.items()
    }
    
    return normalized_values

def in_common(a, b):
    return len(set(a) & set(b))
import copy
def calculate_reason():
    
    reason_map =["Paper Content",
                 "Paper Citation",
                 "Paper Timeliness", 
                 "Author Citation",
                 "Author Country",
                 "Paper Topic",
                 'Author Name'
                 ]
    reason_map = {str(idx): reson for idx, reson in enumerate(reason_map)}
    country_map = readinfo("evaluate/article/country.json")
    # llms = ["gpt3.5","gpt4", "qwen2", "vllm"]
    llm_config_map = {
       "gpt3.5":"LLMGraph/tasks/llm_agent_6/configs/search_shuffle_base_gpt3.5",
        "gpt4":"LLMGraph/tasks/llm_agent_6/configs/search_shuffle_base_gpt4-mini",
        "vllm":"LLMGraph/tasks/llm_agent_6/configs/search_shuffle_base_vllm",
        # "qwen2":"LLMGraph/tasks/llm_agent_4/configs/search_shuffle_base_qwen2"
    }
    llm_reasons = {}
    for llm, config_path in llm_config_map.items():
        # reasons = readinfo(os.path.join(config_path,"evaluate/reason/reason_info_cited.json"))
        reasons = readinfo(os.path.join(config_path,"evaluate/reason/reason_info.json"))

        article_meta_info = readinfo(os.path.join(config_path,"data/article_meta_info.pt"))
        threshold = 1000
        article_meta_info = dict(list(article_meta_info.items())[:threshold])
        author_info = readinfo(os.path.join(config_path,"data/author.pt"))
        core_author_ids = list(filter(lambda x:author_info[x]["country"] in country_map["core"], author_info.keys()))

        core_articles = list(filter(lambda x: in_common(article_meta_info[x]["author_ids"][:1],core_author_ids), article_meta_info.keys()))
        
        reasons_core = {}
        reasons_non_core = {}
        for title,reason in reasons.items():
            try:
                motive_reasons = reason["part_reason"]
            except:
                motive_reasons = reason["reason"]
            for k, v in motive_reasons.items():
                if title in core_articles:
                    if str(k) not in reasons_core.keys() and str(k) in reason_map.keys():
                        reasons_core[str(k)] = 0
                    if str(k) in reason_map.keys():
                        reasons_core[str(k)] += v
                else:
                    if str(k) not in reasons_non_core.keys() and str(k) in reason_map.keys():
                        reasons_non_core[str(k)] = 0
                    if str(k) in reason_map.keys():
                        reasons_non_core[str(k)] += v
        reasons_all = copy.deepcopy(reasons_core)
        reasons_core = normalize_dict_values(reasons_core)
        reasons_non_core = normalize_dict_values(reasons_non_core)
        for r,v in reasons_non_core.items():
            if r not in reasons_all.keys():
                reasons_all[r] = 0
            reasons_all[r] += v        
        reasons_all = normalize_dict_values(reasons_all)
        reasons = {
            "core":{reason_map[idx]:reasons_core[idx] for idx in reasons_core.keys()},
            "non_core":{reason_map[idx]:reasons_non_core[idx] for idx in reasons_non_core.keys()},
            "all":{reason_map[idx]:reasons_all[idx] for idx in reasons_all.keys()}
        }
        llm_reasons[llm] = reasons

    writeinfo("evaluate/article/reason_count_6.json",llm_reasons)

calculate_reason()