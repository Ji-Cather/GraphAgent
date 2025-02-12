import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import networkx as nx

import pandas as pd
import os
import numpy as np
from evaluate.matrix.base_info import calculate_effective_diameter
import json

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)


def plot_relative_size(matrix_dir:str,
                        save_dir:str,
                        graph_name:str):
    matrix_names = ["action",
                    # "follow",
                    # "friend"
                    ]
    lable_map ={
        "action":"Action.",
        "follow":"Follow.",
        "friend":"Friend."
    }
    for matrix_name in matrix_names:
        matrix_path = os.path.join(matrix_dir, f"{matrix_name}_matrix.csv")
        matrix = pd.read_csv(matrix_path,index_col = 0)
        dates = []
        for index, row in matrix.iterrows():
            date = index.split("_")[0]
            dates.append(date)
        relative_sizes = matrix["Relative Size"].to_list()
    
        # 绘制曲线图
        plt.plot(dates, relative_sizes, 
                 marker='o', linestyle='-', 
                 label=lable_map[matrix_name])
        plt_data ={
            "out_degree": dates,
            "relative_size":  relative_sizes
        }
        from LLMGraph.utils.io import writeinfo
        writeinfo(os.path.join(save_dir, f"{matrix_name}_proportion_cc.json"),info=plt_data)
    
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_proportion_cc.pdf")
    plt.yticks(fontsize = 24)
    # plt.title('Relative Size of Connected Component',fontsize = 20)
    # plt.xlabel('Time',fontsize = 20)
    # plt.ylabel('Relative Size',fontsize = 20)
    # plt.legend(fontsize = 20)
    plt.xlabel('Time',fontsize = 26)
    plt.ylabel('Relative Size',fontsize = 26)
    plt.legend(fontsize = 26)
    plt.grid(True)

    # # 选择要显示的日期值
    # date_values = [] 
    # thunk_size = len(dates)//5
    # thunk_size = 1 if thunk_size==0 else thunk_size
    # for idx in range(0,len(dates),thunk_size):
    #     date_values.append(dates[idx])
    # plt.xticks(date_values, rotation=45)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

def plot_gcc_proportion(G:nx.Graph,
                        save_dir:str,
                        graph_name:str):
    
    # 初始化记录每次移除节点后的SCC大小分布
    scc_size_categories = [1, (2, 7), (8, 63), (64, float('inf'))]
    category_labels = ['1', '2-7', '8-63', '>63']
    H = nx.Graph(G)
    sccs = nx.connected_components(H)
    scc_sizes = [len(scc) for scc in sccs]

    category_counts = {label: 0 for label in category_labels}
    for size in scc_sizes:
        for category, label in zip(scc_size_categories, category_labels):
            if isinstance(category, int):
                if size == category:
                    category_counts[label] += 1
            else:
                if category[0] <= size <= category[1]:
                    category_counts[label] += 1

    # 计算每个分类的百分比
    total_nodes = len(G.nodes())
    normalized_scc_sizes_by_category = {label: 0 for label in category_labels}
    for label in category_labels:
        count = category_counts[label]
        if total_nodes > 0:
            normalized_scc_sizes_by_category[label] = count / total_nodes
        else:
            normalized_scc_sizes_by_category[label] = 0

    if nx.is_directed(G):
        largest_cc = max(nx.strongly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)

    largest_cc = G.subgraph(largest_cc)
    plot_data = {
        **normalized_scc_sizes_by_category,
        "total_nodes": total_nodes,
        "diameter_cc": calculate_effective_diameter(largest_cc)
    }
    os.makedirs(save_dir, exist_ok=True)
    writeinfo(os.path.join(save_dir, f"{graph_name}_gcc_proprtion.json"),info=plot_data)

    # 绘制百分比堆叠柱状图
    # fig, ax = plt.subplots(figsize=(12, 6))


    # for label in category_labels:
    #     ax.bar(ind, normalized_scc_sizes_by_category[label], bottom=bottom, label=f'SCC Size {label}')
    #     bottom += normalized_scc_sizes_by_category[label]

    # # 添加标题和标签
    # plt.title('Breakdown of Network into SCCs by Removing High-Degree Nodes (Stacked by Custom Sizes)')
    # plt.xlabel('Number of Nodes Removed')
    # plt.ylabel('Percentage of Total Nodes in SCC')
    # plt.xticks(ind, labels=[str(i+1) for i in ind])
    # plt.legend(title='SCC Size Category')
    # plt.grid(True)

    # # 显示图表
    # plt.show()

    save_path = os.path.join(save_dir, f"{graph_name}_gcc_proprtion.pdf")
    # 显示图表
    plt.savefig(save_path)




