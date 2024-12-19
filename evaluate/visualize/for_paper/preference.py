import numpy as np
import os
from LLMGraph.utils.io import readinfo, writeinfo
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt


llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o",
        "llama8b":"LLAMA8B",
        "vllm": "LLAMA70B",
}



def concat_preference(preference_type,config_templates_map):
    
    countrys = readinfo("evaluate/article/country.json")
    countrys = [countrys[country] for country in countrys.keys()]
    c =[]
    for c_ in countrys:
        for _ in c_:
            c.append(_.lower())
    countrys = c

    ps_map = {
        "CPS": {},
        "RPS": {}
    }
    ps_map_std = {
        "CPS": {},
        "RPS": {}
    }

    for command, llm_name in zip(config_templates_map["LLM-Agent"],llm_name_map.values()):
        task_name, config = command
        path = f"LLMGraph/tasks/{task_name}/configs/{config}/evaluate/preferences.json"
        preferences = readinfo(path)
        for ps_id in ["CPS","RPS"]:
            ps_mean = preferences[ps_id]
            
            ps_map[ps_id][llm_name] = pd.DataFrame([ps_mean["mean"]])
            # ps_map_std[ps_id][llm_name] = pd.DataFrame([ps_mean["std"]])
    for k,df in ps_map.items():
        df_concat = pd.concat(df.values(),axis=0)
        df_concat.index = list(df.keys())
        # df_std_concat = pd.concat(ps_map_std[k].values(),axis=0)
        # df_std_concat.index = list(df.keys())
        root_dir = f"evaluate/article/preference/{preference_type}/{task_name}"
        os.makedirs(root_dir,exist_ok=True)
        df_concat.to_csv(os.path.join(root_dir,f"preference_{k}.csv"))
        # df_std_concat.to_csv(os.path.join(root_dir,f"preference_{k}_std.csv"))

def normalize_preference(preference_type,task_names,config_name):
    
    index = list(llm_name_map.values())
   
    ps_keys =["CPS","RPS"]
    ps_1 ={
        k: {"core":[], # 大于1的数量
        "other":[]
        } for k in ps_keys
    }
    # for ps_key in ps_keys:
    #     dfs = []
    #     for task_name in task_names:
           
    #         root_dir = f"evaluate/article/preference/{preference_type}/{task_name}"
    #         df = pd.read_csv(os.path.join(root_dir,f"preference_{ps_key}.csv"),index_col=0)
    #         df = df.iloc[:,:country_list]
    #         ps_1[ps_key]["other"].append((df.loc[id][10:] > 1).sum().to_dict())
    #         ps_1[ps_key]["core"].append((df.loc[id][:10] > 1).sum().to_dict())
    #         # print(ps_key)
    #         dfs.append(df)

    #     if len(dfs) > 1:
    #         df = pd.concat(dfs)
    #         df_std = []
    #         df_mean = []
    #         for llm in index:
    #             df_std.append(df.loc[llm].std(axis=0))
    #             df_mean.append(df.loc[llm].mean(axis=0))
    #         df_mean = pd.concat(df_mean,axis=1).T
    #         df_std = pd.concat(df_std,axis=1).T
    #         df_mean.index = index
    #         df_std.index = index
    #     else:
    #         df_mean = df
    #         df_std = pd.DataFrame(0, index=df.index, columns=df.columns)
    #     for id in df_mean.index:
    #         gini_pre = gini_coefficient(df_mean.loc[id].to_list())
    #         df_1 = (df_mean.loc[id] > 1).sum()
    #         # df_mean.loc[id] = df_mean.loc[id] / df_mean.loc[id].sum()
    #         # gini_after = gini_coefficient(df_mean.loc[id].to_list())
    #         # print(id,gini_pre, df_1)
    #         # print(id,(df_mean.loc[id][:10] > 1).sum(), (df_mean.loc[id][10:] > 1).sum())
    #     df_mean.to_csv(os.path.join(f"evaluate/article/preference/{preference_type}",f"preference_{ps_key}.csv"))
    #     df_std.to_csv(os.path.join(f"evaluate/article/preference/{preference_type}",f"preference_{ps_key}_std.csv"))

    p_llm_data_path = f"evaluate/article/preference/{preference_type}/all_ps.json"
    core_country_num = 10
    p_llm_data = {}
    for ps_key in ps_keys:
        for p_llm, llm in llm_name_map.items():
            if llm not in p_llm_data.keys():
                p_llm_data[llm] = {}
            
            # df_mean = pd.read_csv(os.path.join(f"evaluate/article/preference/{preference_type}/llm_agent_1",f"preference_{ps_key}.csv"),index_col=0)

            try:
                values = []
                for task_name in task_names:
                    rps_df = pd.read_csv(f"evaluate/article/preference/{config_name}/{task_name}/preference_RPS.csv",
                    index_col=0)
                    values_all = rps_df.loc[llm].values
                    core_rps = values_all[:core_country_num]
                    ph_rps = values_all[core_country_num:]
                    core_rps = list(filter(lambda x: x!=1, core_rps))
                    ph_rps = list(filter(lambda x: x!=1, ph_rps))
                    all_rps = ph_rps+core_rps
                    values.append((np.average(core_rps),
                                              np.average(ph_rps),
                                              np.mean(all_rps)))
                
                values = np.array(values)
                p_llm_data[llm][ps_key] = (np.average(values[:,0]),
                                           np.average(values[:,1]),
                                           np.average(values[:,2]),
                                           np.std(values[:,0]),
                                           np.std(values[:,1]),
                                           np.std(values[:,2])
                                           )

                # p_llm_data[llm][ps_key] = (sum(values_all.iloc[0,:10])/10,
                #                         sum(values_all.iloc[0,10:])/(values_all.shape[1]-10),
                #                         sum(values_all.iloc[0,:])/values_all.shape[1])
                
                # p_llm_data[llm][ps_key] =((values_all.iloc[0,:10] > 1).sum()/10,
                #                         (values_all.iloc[0,10:] > 1).sum()/10,
                #                         (values_all.iloc[0,:] > 1).sum()/20)
                
            except Exception as e:
                continue
    # p_llm_data.update(ps_1)
    writeinfo(p_llm_data_path,p_llm_data)
    


def gini_coefficient(incomes):
    """
    Calculate the Gini coefficient of a list of incomes.

    Parameters:
    incomes (list of float): A list of incomes

    Returns:
    float: The Gini coefficient
    """
    if not incomes:  # If the list is empty, return 0
        return 0

    # Sort the incomes in ascending order
    incomes = sorted(incomes)
    
    # Number of incomes
    n = len(incomes)
    
    # Summation of incomes
    sum_incomes = sum(incomes)
    
    if sum_incomes == 0:  # Prevent division by zero
        return 0
    
    # Calculating the Gini coefficient
    index = 0
    cumulate_income = 0
    for i in range(n):
        cumulate_income += incomes[i]
        index += (i + 1) * incomes[i]
    
    gini = (2 * index) / (n * sum_incomes) - (n + 1) / n
    
    return gini

def plot_preference():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    
    keys = ["RPS","CPS"]    
    countrys = [
        "China",
        "USA",
        "UK",
        "Canada",
        "Germany",
        "Australia",
        "India",
        "Japan",
        # "Austria",
        # "Spain"
        ]

    pre_map = {}
    for key in keys:
        root_dir = f"evaluate/article/preference/{preference_type}"
        df = pd.read_csv(os.path.join(root_dir,f"preference_{key}.csv"),index_col=0)
        df_std = pd.read_csv(os.path.join(root_dir,f"preference_{key}_std.csv"),index_col=0)
        pre_map[key] = (df,df_std)


    llms = list(llm_name_map.values())
    gridspec_kw = {'width_ratios': [8, 2], 'height_ratios': [5, 5]}
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharey=False,gridspec_kw=gridspec_kw)
    labels = []
    for id, key in enumerate(keys):
        bar_width = 0.2
        idx = id
        ax = axs[idx][0]
        # 定义每组柱子的x轴
        index = np.arange(len(countrys))
        import matplotlib.cm as cm
        # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
        # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        color_map_a = cm.get_cmap('plasma', len(llms))  # 获取color map
        colors_a = color_map_a(np.linspace(0, 1,len(llms)) )
        sub_labels =[]
        # 设置正方向的误差（下边的误差为0）
        ax_2 = axs[idx][1]
        llm_sums = []
        std_err = []
        for idx, llm in enumerate(llms):
            try:
                values = pre_map[key][0].loc[llm].to_list()[:len(countrys)]
            except:continue
            values_all = pre_map[key][0].loc[llm].to_list()
            stds = pre_map[key][1].loc[llm].to_list()[:len(countrys)]

            # values = values/values_sum
            # stds = stds/values_sum
            yerr_lower = np.zeros_like(stds)
            yerr_upper = stds
            
            bar = ax.bar(index+idx*bar_width, values, bar_width, label=f"{llm}",
                color= colors_a[idx], 
                yerr=[yerr_lower, yerr_upper])
            std_err.append(sum(stds))
            sub_labels.append(bar.get_label())
            # llm_sums.append(sum(values[:5])/values_sum)
            llm_sums.append(sum(pre_map[key][0].loc[llm].to_list()[:10])/10)

        # ax_2.bar(llms, llm_sums, 0.7, color = colors_a,yerr = std_err)
        
        

        ax_2.bar(["Citeseer","Cora"], llm_sums, 0.7, color = colors_a)
        
        ax_2.set_ylabel(f"$\\operatorname{{{key}}}(C_{{c}})$",fontsize = 16)

        ax.set_ylabel(f"{key}",fontsize = 16)
        # ax.set_ylabel('Article Number by Country')
        # ax.set_title('Bar graph of four dictionaries')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if id == len(keys)-1:
            ax.set_xticks(index + 1.5 * bar_width)
            ax.set_xticklabels(countrys,fontsize = 14,rotation=30, ha='right')
            ax_2.set_xticklabels(llms,fontsize = 12,rotation=30, ha='right')
            labels.extend(sub_labels)
            label = ax.axhline(y=1, color='#696969', linestyle='--', label=f'{key} = 1', alpha=0.7)
            labels.append(label.get_label())
        else:
            ax.set_xticks(index + 0.5 * bar_width)
            ax.get_xaxis().set_visible(False)
            ax_2.get_xaxis().set_visible(False)
            label = ax.axhline(y=1, color='black', linestyle='--', label=f'{key} = 1', alpha=0.7)
            labels.append(label.get_label())
    
    # plt.xticks(rotation=30, ha='right')  # 将国家名称标签旋转45度以减少重叠
    plt.subplots_adjust(top=0.8, bottom=0.17,hspace=0.1)
    # labels =[*labels[1:],labels[0]]
    handles, labels = axs.flat[0].get_legend_handles_labels() 
    handles_2, labels_2 = axs.flat[2].get_legend_handles_labels()
    handles =[handles_2[0],*handles]
    fig.legend(handles=handles, labels=[labels_2[0],*labels],loc='lower center',ncol=6, fontsize=16)
    # 显示图形
    plt.savefig(f"evaluate/visualize/for_paper/preference_{preference_type}.pdf")
    # 右边缺两张小的子图plot sum

def plot_preference_avg():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    
    keys = ["RPS","CPS","PS"]    
    countrys = [
        "China",
        "USA",
        "UK",
        "Canada",
        "Germany",
        "Australia",
        "India",
        "Japan",
        "Austria",
        "Spain"]

    pre_map = {}
    for key in keys:
        root_dir = f"evaluate/article/preference/{preference_type}"
        if key == "PS":
            continue
        df = pd.read_csv(os.path.join(root_dir,f"preference_{key}.csv"),index_col=0)
        df_std = pd.read_csv(os.path.join(root_dir,f"preference_{key}_std.csv"),index_col=0)
        pre_map[key] = (df,df_std)


    llms = list(llm_name_map.values())
    gridspec_kw = {'width_ratios': [8, 2],}
    fig, axs = plt.subplots(3, 2, figsize=(16, 8), sharey=False,gridspec_kw=gridspec_kw)
    labels = []
    
    for id, key in enumerate(keys):
        bar_width = 0.2
        idx = id
        ax = axs[idx][0]
        # 定义每组柱子的x轴
        index = np.arange(len(countrys))
        import matplotlib.cm as cm
        # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
        # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        color_map_a = cm.get_cmap('plasma', len(llms))  # 获取color map
        colors_a = color_map_a(np.linspace(0, 1,len(llms)) )
        sub_labels =[]
        # 设置正方向的误差（下边的误差为0）
        ax_2 = axs[idx][1]
        llm_sums = []
        std_err = []
        for idx, llm in enumerate(llms):
            if key == "PS":
                values = np.array(pre_map["RPS"][0].loc[llm].to_list()[:len(countrys)])+ np.array(pre_map["CPS"][0].loc[llm].to_list()[:len(countrys)])
                # values = np.array(values)/2
                stds = np.array(pre_map["RPS"][1].loc[llm].to_list()[:len(countrys)]) + np.array(pre_map["CPS"][1].loc[llm].to_list()[:len(countrys)])
                # stds = np.array(stds)/2

            else:
                values = pre_map[key][0].loc[llm].to_list()[:len(countrys)]
                values_sum = np.sum(pre_map[key][0].loc[llm].to_list())
                stds = pre_map[key][1].loc[llm].to_list()[:len(countrys)]

            # values = values/values_sum
            # stds = stds/values_sum
            yerr_lower = np.zeros_like(stds)
            yerr_upper = stds
            
            bar = ax.bar(index+idx*bar_width, values, bar_width, label=f"{llm}",
                color= colors_a[idx], 
                yerr=[yerr_lower, yerr_upper])
            std_err.append(sum(stds))
            sub_labels.append(bar.get_label())
            llm_sums.append(sum(values))

        
        ax_2.bar(llms, llm_sums, 0.7, color = colors_a,yerr = std_err)
        ax_2.set_ylabel(f"$\\sum{{}}{{}}\\operatorname{{{key}}}(C_{{c}})$",fontsize = 16)

        ax.set_ylabel(f"{key}",fontsize = 16)
        # ax.set_ylabel('Article Number by Country')
        # ax.set_title('Bar graph of four dictionaries')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if id == len(keys)-1:
            ax.set_xticks(index + 1.5 * bar_width)
            ax.set_xticklabels(countrys,fontsize = 14,rotation=30, ha='right')
            ax_2.set_xticklabels(llms,fontsize = 12,rotation=30, ha='right')
            labels.extend(sub_labels)
            label = ax.axhline(y=1, color='#696969', linestyle='--', label=f'{key} = 1', alpha=0.7)
            labels.append(label.get_label())
        else:
            ax.set_xticks(index + 0.5 * bar_width)
            ax.get_xaxis().set_visible(False)
            ax_2.get_xaxis().set_visible(False)
            label = ax.axhline(y=1, color='black', linestyle='--', label=f'{key} = 1', alpha=0.7)
            labels.append(label.get_label())
    
    # plt.xticks(rotation=30, ha='right')  # 将国家名称标签旋转45度以减少重叠
    plt.subplots_adjust(top=0.8, bottom=0.17,hspace=0.1)
    # labels =[*labels[1:],labels[0]]
    handles, labels = axs.flat[0].get_legend_handles_labels() 
    handles_2, labels_2 = axs.flat[2].get_legend_handles_labels()
    handles =[handles_2[0],*handles]
    fig.legend(handles=handles, labels=[labels_2[0],*labels],loc='lower center',ncol=6, fontsize=16)
    # 显示图形
    plt.savefig(f"evaluate/visualize/for_paper/preference_all.pdf")
    # 右边缺两张小的子图plot sum


def plot_gt_preference_scr():
    
    sc_path ="evaluate/article/preference/gt/self_citation.json"
    sc= readinfo(sc_path)
    countrys = {"united states":"US",
                "united kingdom":"UK",
                "canada":"CA",
                "australia":"AU",
                "germany":"GB",
                "netherlands":"NL",
                "india":"IN",
                "france":"FR",
                "china":"CN",
                }
    rps_df = pd.read_csv("evaluate/article/preference/gt/preference_RPS.csv",index_col=0)

    gridspec_kw = {'width_ratios': [8, 2],}
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharey=False)
    labels = []
    
    
    bar_width = 0.3
    idx = id
    
    # 定义每组柱子的x轴
    index = np.arange(len(countrys))
    import matplotlib.cm as cm
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    import seaborn as sns
    colors_map_a = sns.color_palette("rocket", as_cmap=True)
    colors_map_a = sns.color_palette('viridis', as_cmap=True)
    colors_map_a = sns.color_palette("Paired",as_cmap=True)
    # colors_map_a = sns.color_palette("hls", 8, as_cmap=True)
    colors_a = colors_map_a(np.linspace(0, 1, 2) )
    # colors_ = colors_map_a(np.linspace(0, 1, 2) )
   
    # 设置正方向的误差（下边的误差为0）
    ax_1 = axs[0]

    ax_1.bar(index, [sc["citeseer"][c] for c in countrys.keys()], bar_width, label="Citeseer",
            color= colors_a[0])
    ax_1.bar(index+bar_width, [sc["cora"][c] for c in countrys.keys()], bar_width, label="Cora",
            color= colors_a[1])
    ax_1.set_ylabel("Self-Citation Rate",fontsize = 16)

    ax_2 = axs[1]
    bar1= ax_2.bar(index, rps_df.loc["GPT-3.5",countrys.keys()], bar_width, label="Citeseer",
            color= colors_a[0])
    bar2 = ax_2.bar(index+bar_width, rps_df.loc["LLAMA-3-70B",countrys.keys()], bar_width,       label="Cora",
            color= colors_a[1])
    ax_2.axhline(y=1, color='black', linestyle='--', label=f'RPS = 1', alpha=0.7)
    ax_2.set_ylabel(f"RPS",fontsize = 16)
    ax_2.tick_params(axis='y', labelsize=14)
    ax_2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_2.set_xticks(index + 0.5 * bar_width)
    ax_2.set_xticklabels(list(countrys.values()),fontsize = 16,ha='right')
    ax_1.set_xticks(index + 0.5 * bar_width)
    ax_1.get_xaxis().set_visible(False)
    ax_1.tick_params(axis='y', labelsize=14)
    # plt.xticks(rotation=30, ha='right')  # 将国家名称标签旋转45度以减少重叠
    plt.subplots_adjust(top=0.8, bottom=0.13,hspace=0.1)
    # labels =[*labels[1:],labels[0]]
    handles, labels = axs.flat[0].get_legend_handles_labels() 
    handles_2, labels_2 = axs.flat[1].get_legend_handles_labels()
    handles =[handles_2[0],*handles]
    fig.legend(handles=handles, labels=[labels_2[0],*labels],loc='lower center',ncol=6, fontsize=16)
    # 显示图形
    plt.savefig(f"evaluate/visualize/for_paper/preference_all.pdf")

def plot_rps_compare():

    plt.figure(figsize=(10,8))
    rps ={
        "Cora":1.02,
        "Citeseer":1.08,
        "Pub.":1.08,
        "PB.":1.02,
        "FB.":0.86
    }
    names = list(rps.keys())
    values = list(rps.values())
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    import seaborn as sns
    colors_map_a = sns.color_palette("rocket", as_cmap=True)
    # colors_map_a = sns.color_palette('viridis', as_cmap=True)
    colors_map_a = sns.color_palette("Paired",as_cmap=True)
    colors_map_a = sns.color_palette("muted", as_cmap=True)
    plt.axhline(y=1, color='black', linestyle='--', label=f'RPS = 1', alpha=0.7)
    colors_a = colors_map_a
    patterns = ['/', '\\', '|', 'x','+']  # 不同的花纹
    plt.ylim(0.8, 1.2)
    bars = plt.bar(names, values, color=colors_a)
    # 设置不同的花纹
    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)

    # colors_a = colors_map_a(np.linspace(0, 1, len(rps)) )
    
    # Improve chart aesthetics
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('RPS', fontsize=25)
    # 设置标题和标签
    # plt.title('RPS Values')
    # plt.xlabel('Experiment')
    # plt.subplots_adjust(top=0.8, bottom=0.13,hspace=0.1)

    plt.savefig("evaluate/visualize/for_paper/rps_compare.pdf")


p_task_name = "llm_agent"

preference_type = "base"
config_templates_map ={
    "LLM-Agent":[
        (p_task_name,"search_shuffle_base_gpt3.5"),
        (p_task_name,"search_shuffle_base_vllm"),
        (p_task_name,"search_shuffle_base_gpt4-mini"),
        (p_task_name,"search_shuffle_base_qwen2"),
    ]
}

# preference_type = "no_country"
# config_templates_map ={
#     "LLM-Agent":[
#         (p_task_name,"search_shuffle_no_country_gpt3.5"),
#         (p_task_name,"search_shuffle_no_country_vllm"),
#         (p_task_name,"search_shuffle_no_country_gpt4-mini"),
#         (p_task_name,"search_shuffle_no_country_qwen2"),
#     ]
# }

# preference_type = "annoymous"
# config_templates_map ={
#     "LLM-Agent":[
#         (p_task_name,"search_shuffle_anonymous_gpt3.5"),
#         (p_task_name,"search_shuffle_anonymous_vllm"),
#         (p_task_name,"search_shuffle_anonymous_gpt4-mini"),
#         (p_task_name,"search_shuffle_anonymous_qwen2"),
#     ]
# }

# preference_type = "no_author_country"
# config_templates_map = {
#     "LLM-Agent":[
#         (p_task_name,"search_shuffle_no_author_country_gpt3.5"),
#         (p_task_name,"search_shuffle_no_author_country_vllm"),
#         (p_task_name,"search_shuffle_no_author_country_gpt4-mini"),
#         (p_task_name,"search_shuffle_no_author_country_qwen2"),
#     ]
# }

preference_type = "gt"
p_task_name = "gt"
config_templates_map ={
    "LLM-Agent":[
        ("citeseer","gt"),
        ("cora","gt"),
        # ("llm_agent_1","gt")
    ]
}


if p_task_name == "llm_agent":
    # task_names = ["llm_agent_1","llm_agent_2"]
    task_names = ["llm_agent_1","llm_agent_2","llm_agent_3"]
    # task_names = ["llm_agent_1"]
    for task_ in task_names:
        config_templates_map = {
            "LLM-Agent":[
                (task_, config_templates_map["LLM-Agent"][idx][1])
                for idx in range(4)
            ]
        }
        concat_preference(preference_type, config_templates_map)
    normalize_preference(preference_type, task_names,"base")
else:
    concat_preference(preference_type,config_templates_map)
    normalize_preference(preference_type,["cora"],"gt")

# plot_preference()
# plot_preference_avg()
# plot_gt_preference_scr()
# plot_rps_compare()