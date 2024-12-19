import pandas as pd
import os

def concat_ex_dfs(task,
                  configs:list = [],
                  result_df_names:list = [],
                  root_dir = "evaluate/experiment",
                  path_template = "LLMGraph/tasks/{task}/configs/{config}/evaluate/{result_df_name}.csv",
                  prefix_df_paths:dict = {}):
    
    os.makedirs(root_dir,exist_ok=True)
    # llms =["gpt3.5","vllm","gpt4-mini","qwen2"]
    if task == "llm_agent":
        if "vllm" in configs[0]:
            tasks = [f"llm_agent_{i}" for i in range(1,3)]
        elif "gpt3.5" in configs[0]:
            tasks = [f"llm_agent_{i}" for i in range(1,3)]
        else:
            tasks = [f"llm_agent_{i}" for i in range(1,3)]
    else: tasks =[task]
    for result_df_name in result_df_names:
        reses = []
        for task in tasks:
            dfs = []
            index_exs = []
            if prefix_df_paths.get(result_df_name) is not None:
                prefix_df_path = prefix_df_paths.get(result_df_name)
                dfs.append(pd.read_csv(prefix_df_path,index_col=0))
                index_exs.append("gt")

            for config in configs:
                path = path_template.format(task=task,config=config,result_df_name=result_df_name)
                if not os.path.exists(path):
                    print(path,"missing")
                    continue
                df = pd.read_csv(path,index_col=0)
                df["ex_name"] = config
                index_exs.append(config)
                dfs.append(df)
            
            if len(dfs) == 0:
                continue
            res = pd.concat(dfs)
            reses.append(res)
            
        res = pd.concat(reses)
        res.to_csv(os.path.join(root_dir,f"{result_df_name}.csv"))


def test_power_law_base_index():
    llms =["gpt3.5","vllm","gpt4-mini","qwen2"]
    file_name = "article_citation_all_power_law.csv"
    for llm in llms:
        path = os.path.join("evaluate/experiment","llm_agent",llm,file_name)
        df = pd.read_csv(path,index_col=0)
        df = df.loc["power_law",:]
        df = df.sort_values(by="KS")
        df["index"] = [i for i in range(len(df))]
        KS_index = df[df["ex_name"]==f"search_shuffle_base_{llm}"]["index"]
        try:
            bigger = df[df["ex_name"]==f"search_shuffle_base_{llm}"].loc["power_law","index"] < \
            df[df["ex_name"]==f"search_shuffle_nocite_{llm}"].loc["power_law","index"]
        except:
            bigger = 0
        KS = df["KS"].mean()
        KS_power_kaw = df[df["ex_name"]==f"search_shuffle_nocite_{llm}"].loc["power_law","KS"]
        try:
            KS_gap_cite = df[df["ex_name"]==f"search_shuffle_nocite_{llm}"].loc["power_law","KS"]-\
                df[df["ex_name"]==f"search_shuffle_base_{llm}"].loc["power_law","KS"] 
        except:
            KS_gap_cite = 0
        try:
            KS_gap = df["KS"][-1]-\
                df[df["ex_name"]==f"search_shuffle_base_{llm}"].loc["power_law","KS"]
        except:
            KS_gap = 0
        # df = df.sort_values(by="ll")
        # df["index"] = [i for i in range(len(df))]
        # ll_index = df[df["ex_name"]==f"search_shuffle_base_{llm}"]["index"]
        print(llm, KS_index,bigger, KS, KS_power_kaw, KS_gap_cite, KS_gap)


if __name__ == "__main__":
    """article"""
    task = "llm_agent"
    # task = "llm_agent_retry"
    llms =["gpt3.5","vllm","gpt4-mini","qwen2"]
    # llms =["qwen2"]
    config_templates =[
            "search_shuffle_base_{llm}",
            "search_shuffle_nocite_{llm}",
            "search_shuffle_no_content_{llm}",
            "search_shuffle_no_country_{llm}",
            "search_shuffle_noauthor_{llm}",
            "search_shuffle_nopapertime_{llm}",
            'search_shuffle_base_noauthorcite_{llm}', 
            'search_shuffle_notopic_{llm}', 
            'search_shuffle_noauthortopic_{llm}', 
            "search_shuffle_anonymous_{llm}",
            "nosearch_shuffle_base_{llm}",
            "search_shuffle_base_nosocial_{llm}"
    ]

    
    # for llm in llms:
    #     configs = []
    #     configs.extend([config_template.format(llm = llm) 
    #                     for config_template in config_templates])
    #     result_df_names = [
    #         # "article_citation_matrix",
    #         # "author_citation_matrix",
    #         # "co_authorship_matrix",
    #         "article_citation_all_power_law",
    #         # "article_citation_in_power_law",
    #         # "article_citation_out_power_law",
    #         # "author_citation_all_power_law",
    #     ]
    #     prefix_df_paths = {
    #         # "article_citation_matrix":\
    #         #                              "evaluate/experiment/llm_agent/gt_matrix.csv"
    #         }
    #     path_template = "LLMGraph/tasks/{task}/configs/{config}/evaluate/{result_df_name}.csv"
    #     root_dir = f"evaluate/experiment/{task}/{llm}"
    #     concat_ex_dfs(task,
    #               configs,
    #               result_df_names,
    #               root_dir = root_dir,
    #               path_template = path_template,
    #               prefix_df_paths = prefix_df_paths)
    # test_power_law_base_index()
    
    # task = "citeseer"
    # configs = [
    # ]
    # config_templates =[
    #     "fast_{llm}",
    #     "fast_{llm}_2",
    # ]
    task = "cora"
    configs = [
    ]
    config_templates =[
        "fast_{llm}",
        "fast_{llm}_2",
    ]
    
    configs =[]
    for llm in llms:
        configs.extend([config_template.format(llm = llm) 
                        for config_template in config_templates])
    configs.append("gt")
    prefix_df_paths = {}
    path_template = "LLMGraph/tasks/{task}/configs/{config}/evaluate/{result_df_name}.csv"
    result_df_names = [
            "article_citation_matrix",
            # "author_citation_matrix",
            # "co_authorship_matrix",
            "article_citation_all_power_law",
            # "article_citation_in_power_law",
            # "article_citation_out_power_law",
            # "author_citation_all_power_law",
        ]

    # """movie"""
    # task = "movielens"
    # configs = [
    #     "filter_1",
    #     "filter_2",
    #     "filter_all",
    #     "filter_all_r",
    #     "filter_all_k3",
    #     "filter_all_k5",
    #     "filter_all_k10",
    # ]
    # result_df_names = [
    #    "movielens_matrix",
    #    "user_projection_matrix"
    # ]
    # prefix_df_paths = {
    # }
    # path_template = "LLMGraph/tasks/{task}/configs/{config}/evaluate/{result_df_name}.csv"

    # """social"""
    # task = "tweets"
    # configs = [
    #     "filter_0",
    #     "filter_1",
    #     "filter_2",
    #     "filter_3",
    #     "filter_4",
    #     "filter_all",
    #     "filter_all_k3",
    #     "filter_all_k5",
    #     "filter_all_k10",
    #     "filter_all_r0",
    #     "filter_all_r0.1",
    # ]
    
    # result_df_names = [
    #     "action_matrix",
    #     "follow_matrix",
    #     "friend_matrix"
    # ]
    # prefix_df_paths = {
    # }
    # path_template = "LLMGraph/tasks/{task}/configs/{config}/evaluate/20240418/{result_df_name}.csv"

    root_dir = f"evaluate/experiment"
    root_dir = os.path.join(root_dir,task)
    concat_ex_dfs(task,
                  configs,
                  result_df_names,
                  root_dir = root_dir,
                  path_template = path_template,
                  prefix_df_paths = prefix_df_paths)