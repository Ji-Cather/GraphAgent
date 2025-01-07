import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import os
from typing import List
from evaluate.visualize.article import plot_gini
import networkx as nx
import matplotlib.dates as mdates
import pandas as pd
def plot_betas(x:list,
               y:List[dict],
               types:list,
               group_name:str,
               error:List[list] = [],
               save_dir:str = "evaluate/article/distortion"):
    legend_map ={
        "country_all":"Core + Periphery",
        "country_core":"Core",
        "country_used":"Core + Periphery"
    }
    # 设置日期格式
    plt.figure(figsize=(10, 6))
    # fig, ax = plt.subplots()
    # date_format = mdates.DateFormatter('%Y-%m-%d')
    # plt.xaxis.set_major_formatter(date_format)
    plt.figure(figsize=(10, 6))  # 可以调整大小以更好地适应国家数量
    for type in types:
        # 绘制折线图
        x_data = x
        y_data = [y_[type] for y_ in y]
        if error ==[]:
            error_ =[None for _ in y]
        else:
            error_ = [error_[type] for error_ in error]

        grouped_data = {}
        for x_,y_,e_ in zip(x_data,y_data,error_):
            if x_ not in grouped_data:
                grouped_data[x_] = []
            grouped_data[x_].append((y_,e_))

        for x_,items in grouped_data.items():
            y_ = [item[0] for item in items]
            if error == []:
                e_ = None
            else:
                e_ = [item[1] for item in items]
                e_ = sum(e_) / len(e_)
            grouped_data[x_] = (sum(y_) / len(y_),
                                e_)
        # used_keys = [
        #     "2024-01",
        #     "2024-02",
        #     "2024-03",
        #     "2024-04",
        #     "2024-05",
        #     "2024-06",
        # ]
        used_keys = list(grouped_data.keys())[1:]
        used_keys_time = pd.to_datetime(used_keys)
        if error == []:
            plt.errorbar(used_keys_time, 
                     [(abs(grouped_data[k][0])) for k in used_keys], 
                     fmt='-o', ecolor='gray', capsize=5,
                     label = legend_map[type])
        else:
            plt.errorbar(used_keys_time, 
                     [(grouped_data[k][0]) for k in used_keys], 
                     yerr=[abs(grouped_data[k][1]) for k in used_keys], 
                     fmt='-o', ecolor='gray', capsize=5,
                     label = legend_map[type])
        # plt.errorbar(x_data, y_data, yerr=error_, fmt='-o', ecolor='gray', capsize=5,
        #              label = type)
    # plt.xticks(rotation=)
    # 设置标题和标签
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    # plt.xlabel('Time',fontsize=16)
    plt.ylabel("|$\\beta$|",fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    group_name = group_name.replace(" ","_").lower()
    path = os.path.join(save_dir,f"beta_{group_name}.pdf")
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


