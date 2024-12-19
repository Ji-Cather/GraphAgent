import matplotlib.pyplot as plt


# 设置默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 选择你想要的字体
import os
from typing import List
from evaluate.visualize.article import plot_gini
import networkx as nx
import matplotlib.dates as mdates
import pandas as pd
from LLMGraph.utils.io import readinfo
import numpy as np

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import copy
def generate_monthly_dates(start_date, months):
    # 转换为 datetime 对象
    start = datetime.strptime(start_date, '%Y-%m')  
    date_list = []

    for i in range(months):
        date_list.append(start + relativedelta(months=i))

    return date_list

import matplotlib.patches as patches
def add_arrowed_spines(ax,x = True):
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
        ax.spines[spine].set_path_effects([])
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    for direction in ['left', 'bottom']:
        xy = (1, 0) if direction == 'bottom' else (0, 1)  # arrow at the end
        style = patches.ArrowStyle.CurveFilledB(head_length=0.15, head_width=0.08)
        arrow = patches.FancyArrowPatch((0, 0), xy, transform=ax.transAxes,
                                        fc='k', lw=0, mutation_scale=15, arrowstyle=style)
        ax.add_patch(arrow)
    
    if x:
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(direction='out')
    else:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(direction='out')

def plt_distortion(types:list,
                   save_dir:str ="evaluate/article/distortion"):


    llm_path_map = {
        "GPT-3.5":
            [
            "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_gpt3.5/evaluate",
            "LLMGraph/tasks/llm_agent_3/configs/search_shuffle_base_gpt3.5/evaluate",
            "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_base_gpt3.5/evaluate"
         ],
        "GPT-4o-mini":["LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_gpt4-mini/evaluate",
                       "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_base_gpt4-mini/evaluate",
                       "LLMGraph/tasks/llm_agent_3/configs/search_shuffle_base_gpt4-mini/evaluate"],        
        # "LLAMA8B":["LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_llama8b/evaluate",
        #             "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_base_llama8b/evaluate",
        #             "LLMGraph/tasks/llm_agent_3/configs/search_shuffle_base_llama8b/evaluate"],
        "LLAMA-3-70B":["LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_vllm/evaluate",
                       "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_base_vllm/evaluate",
                       "LLMGraph/tasks/llm_agent_3/configs/search_shuffle_base_vllm/evaluate"],
        }
    
    # llm_path_map = {
    #     "GPT-3.5":
    #         [
    #         "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_gpt3.5/evaluate",
           
    #      ],
    #     "LLAMA-3-70B":["LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_vllm/evaluate",
    #                    ],
    #     "GPT-4o-mini":["LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_gpt4-mini/evaluate",
    #                    ],
    #     "QWEN2-70B":["LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_qwen2/evaluate",
    #                    ]
    #     }
    
    beta_types = {
        "Base":'Base'
    }
    betas_dict = {
        # "country_all":[],
        "country_core":[],
        "country_used":[]
    }
    llm_datas = {}
    
    for llm, beta_roots in llm_path_map.items():
        llm_datas[llm] = {}
        for beta_type in beta_types.keys():
            rps_datas = []
            for root in beta_roots:
                rps_path = os.path.join(root, "preferences_time.json")
                rps_data = readinfo(rps_path)
                rps_datas.append(rps_data["RPS"])
            
            rps_min_lens = []
            
            idx_b = 0
            for col_idx, country_type in enumerate(betas_dict.keys()):
                rps_datas_col = [[_[col_idx] for _ in rps_data] for rps_data in rps_datas]
                min_rps_len = min([len(rps_data_col) for rps_data_col in rps_datas_col])
                rps_min_lens.append(min_rps_len)
                rps_datas_col = [rps_data_col[:min_rps_len] for rps_data_col in rps_datas_col]
            
            beta_data = []
            error_data = []
            for i in range(min(rps_min_lens)):
                beta_data.append({"country_core":1,
                                "country_used":1})
                error_data.append({"country_core":0,
                                "country_used":0})
            
            for col_idx, country_type in enumerate(betas_dict.keys()):
                rps_datas_col = [[_[col_idx] for _ in rps_data] for rps_data in rps_datas]
                min_rps_len = min([len(rps_data_col) for rps_data_col in rps_datas_col])
                rps_min_lens.append(min_rps_len)
                rps_datas_col = [rps_data_col[:min_rps_len] for rps_data_col in rps_datas_col]
                rps_datas_col_avg = np.average(np.array(rps_datas_col),axis =0)
                rps_datas_col_std = np.std(np.array(rps_datas_col),axis =0)
                # 修改
                try:
                    for idx in range(min_rps_len):
                        beta_data[idx][country_type] = rps_datas_col_avg[idx]
                        error_data[idx][country_type] = rps_datas_col_std[idx]
                except:
                    continue

                base_start_date = datetime(2023, 5, 11) + timedelta(days=111)
                base_dates = [base_start_date + timedelta(days=i*10) for i in range(len(beta_data))]
                base_dates = pd.to_datetime(base_dates, format="%Y-%m")
                base_dates = base_dates.strftime("%Y-%m")
                time_data = base_dates
            
            if llm == "QWEN2-70B":
                llm_datas[llm][beta_type] = (copy.deepcopy(time_data)[18:],
                                         copy.deepcopy(beta_data)[18:],
                                         copy.deepcopy(error_data)[18:]
                                         )
            else:
                llm_datas[llm][beta_type] = (copy.deepcopy(time_data),
                                         copy.deepcopy(beta_data),
                                         copy.deepcopy(error_data)
                                         )
    
    plot_betas(llm_datas,save_dir=save_dir,
                types=types,
                group_name="all")
        

def plot_betas(llm_datas,
               types:list,
               group_name:str,
               save_dir:str = "evaluate/article/distortion"):
    
    beta_type_len = len(llm_datas["GPT-3.5"])
    legend_map ={
        "country_all":"Core + Periphery",
        "country_core":"Core",
        "country_used":"Core + Periphery"
    }
    # 设置日期格式
    # fig, axs = plt.subplots(beta_type_len, 4, figsize=(18, 12), sharey=False)
    fig, axs = plt.subplots(beta_type_len, len(llm_datas), figsize=(18, 6), sharey=False)

    idy = 0
    labels = []
    colors = [["#E6846D","#F9BEBB"],["#8DCDD5","#89C9C8"]]
    for llm, beta_type_betas in llm_datas.items():
        idx = 0
        for beta_type, beta_data in beta_type_betas.items():
            x,y, error = beta_data
            try:
                ax = axs[idx][idy]
            except:
                ax = axs[idy]
            idz=0
            for type in types:
                # 绘制折线图
                x_data = x
                y_data = [y_[type] for y_ in y]
                error_ = [error_[type] for error_ in error]

                grouped_data = {}
                for x_,y_,e_ in zip(x_data,y_data,error_):
                    if x_ not in grouped_data:
                        grouped_data[x_] = []
                    grouped_data[x_].append((y_,e_))

                for x_,items in grouped_data.items():
                    y_ = [item[0] for item in items]
                    e_ = [item[1] for item in items]
                    e_ = sum(e_) / len(e_)
                    grouped_data[x_] = (sum(y_) / len(y_),
                                        e_)
               
                used_keys = list(grouped_data.keys())[1:]
                used_keys_time = pd.to_datetime(used_keys)
                # y_ = [(abs(grouped_data[k][0])) for k in used_keys]
                y_ = np.array([(grouped_data[k][0]) for k in used_keys])
                error_ = np.array([(grouped_data[k][1]) for k in used_keys])
                # y_ = [f"{_}:.2f" for _ in y_]

                bar = ax.errorbar(used_keys_time, 
                            y_, 
                            fmt='-o', ecolor=colors[idz][0],
                            color=colors[idz][0], capsize=5,
                            label = legend_map[type],
                            yerr = error_
                            )
                ax.fill_between(used_keys_time, y_ - error_, y_ + error_, color=colors[idz][1], alpha=0.5)
                idz+=1
                labels.append(bar.get_label())
                if idy == 0:
                    # ax.set_ylabel("$\\beta$", fontsize=20)
                    # ax.set_yticks(np.arange(0, 0.225, 0.025))

                    # add_arrowed_spines(ax,x=False)
                    
                    ax.annotate('', xy=(0, 1), xytext=(0, 0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='black'))
                else:
                    # ax.set_yticks(np.arange(0, 0.225, 0.025))
                    ax.get_yaxis().set_visible(False)
                    pass
                    ax.spines['left'].set_visible(False)
                    # if idx == 3:
                    #     add_arrowed_spines(ax,x=True)
                ax.tick_params(axis='both', which='major', labelsize=18)  # 更改主要刻度标记的字体大小
                ax.tick_params(axis='both', which='minor', labelsize=18)  # 更改次要刻度标记的字体大小
                if idx ==0:
                    ax.set_title(llm, fontsize=26)
                # if idy ==3:
                #     ax.text(-0.1, 0.5, beta_type, 
                # ha='center', va='center', fontsize=18)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.spines['left'].set_visible(True)


            # 设置标题和标签
            # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
           
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每隔3个月
            fig.autofmt_xdate()
            
            idx+=1
        
        idy+=1

    # plt.ylabel("$\\beta$", fontsize=20)
    # fig.text(0.5, 0.13, r'$k$', ha='center', fontsize=18)
    fig.text(0.02, 0.5, "   $\text{RPS }_{\text{year}}$\n(standard deviations)", va='center', rotation='vertical', fontsize=22)
    # for height, label in zip([0.26,0.52,0.78],
    #                          ["PA-edge + Same-Content","CiteAI-edge + Same-Content", "CiteAI-edge + LLM-Content",]):
    # for height, label in zip([0.26,0.52,0.78],
    #                          ["PA (e)","LLM-cite (e)", "LLM-cite (d)"]):
    #     fig.text(0.9, height, label, va='center', rotation='vertical', fontsize=18)
    beta_types = list(llm_datas["GPT-3.5"].keys())
    # for height, label in zip([0.2,0.4,0.6,0.8],
    #                          reversed(beta_types)):
    #     fig.text(0.9, height, label, va='center', rotation='vertical', fontsize=18)
    # for height, label in zip([0.25,0.75],
    #                          reversed(beta_types)):
    #     fig.text(0.9, height, label, va='center', rotation='vertical', fontsize=22)
    for height, label in zip([0.25,0.5,0.75],
                             reversed(beta_types)):
        fig.text(0.9, height, label, va='center', rotation='vertical', fontsize=22)

    plt.subplots_adjust(top=0.9, bottom=0.24,left=0.1, hspace=0.3,wspace=0)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    fig.legend(labels=labels[:2],loc='lower center',ncol=4, fontsize=20)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    group_name = group_name.replace(" ","_").lower()
    path = os.path.join(save_dir,f"rps_{group_name}.pdf")
    # plt.tight_layout()
    plt.savefig(path)

def plot_betas_ori(save_dir:str = "evaluate/article/distortion"):
    
    # 设置日期格式
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=False)

    idx = 0
    labels = []
    data = {
        
        "Core": [0.293,0.281,0.259,0.279,0.263,0.287,0.27,0.281,0.279,0.29,0.288,0.319,0.29,
                 0.30,0.309,0.319,0.292,0.321,0.308,0.29],
        "Core + Periphery": [0.27,0.265,0.241,0.258,0.27,0.259,0.256,0.258,0.26,0.261,0.27,0.2613,0.255,
                    0.256,0.26,0.261,0.27,0.261,0.257,0.258,0.259],
    }
    
    begin = "1980-01"
    colors = [["#E6846D","#F9BEBB"],["#8DCDD5","#89C9C8"]]
    len_data = min([len(v) for v in data.values()])
    used_keys_time = generate_monthly_dates(begin,len_data)
    idz = 0
    for type,y_ in data.items():
        y_ = y_[:len_data]
        bar = ax.errorbar(used_keys_time, 
                    y_, 
                    fmt='-o', ecolor=colors[idz][0],
                            color=colors[idz][0], capsize=5,
                    label = type)
        labels.append(bar.get_label())
        ax.set_yticks(np.arange(0.23, 0.33, 0.02))
        ax.set_ylabel("$\\beta$", fontsize=18)
        # add_arrowed_spines(ax,x=False)
        ax.annotate('', xy=(0, 1), xytext=(0, 0),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle="->", color='black'))
        idz+=1

        ax.tick_params(axis='both', which='major', labelsize=14)  # 更改主要刻度标记的字体大小
        ax.tick_params(axis='both', which='minor', labelsize=14)  # 更改次要刻度标记的字体大小
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        # 设置标题和标签
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 
        plt.gcf().autofmt_xdate()
        idx+=1

    plt.ylim(0.23, 0.33)
    plt.yticks(np.arange(0.23, 0.33, 0.02))
    # fig.text(0.5, 0.13, r'$k$', ha='center', fontsize=18)
    # fig.text(0, 0.5, "|$\\beta$|", va='center', rotation='vertical', fontsize=18)
    plt.subplots_adjust(top=0.8, bottom=0.24,left=0.05, hspace=0.2,wspace=0)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # fig.legend(labels=labels[:2],loc='lower center',ncol=4, fontsize=16)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir,f"beta_ori.pdf")
    plt.tight_layout()
    plt.savefig(path)






plt_distortion(["country_core","country_used"],
               "evaluate/visualize/for_paper")
# plot_betas_ori("evaluate/visualize/for_paper")
