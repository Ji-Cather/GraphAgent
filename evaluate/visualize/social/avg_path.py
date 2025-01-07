import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]


import os
from evaluate.matrix.base_info import calculate_effective_diameter

# 提取子图
def get_fraction_of_graph(graph, fraction):
    assert 0 < fraction <= 1, "Fraction must be between 0 and 1."
    num_nodes_to_sample = int(len(graph.nodes()) * fraction)
    if num_nodes_to_sample ==0:
        raise Exception("Empty Graph")
    sampled_nodes = random.sample(graph.nodes(), num_nodes_to_sample)

    # 确保子图不是空图
    while len(sampled_nodes) == 0:
        sampled_nodes = random.sample(graph.nodes(), num_nodes_to_sample)

    subgraph = graph.subgraph(sampled_nodes).copy()
    return subgraph



def plot_avg_path(G:nx.DiGraph,
                      save_dir:str,
                        graph_name:str):
    from LLMGraph.utils.io import writeinfo, readinfo
    # 指定要分析的图部分（fraction）
    # fractions = np.linspace(0.01, 1.0, 10)
    fractions =[
        0.0001,
        0.001,
        0.01,
        *np.linspace(0.01, 0.1, 10).tolist(),
        *np.linspace(0.1, 0.2, 10).tolist()
    ]
    if os.path.exists(os.path.join(save_dir, f"{graph_name}_avg_path.json")):
        data =  readinfo(os.path.join(save_dir, f"{graph_name}_avg_path.json"))
        fractions = data["fractions"]
        avg_path_lengths = data["avg_path_length"]
    else: 
        
        fraction_map ={}
        avg_path_lengths =[]
        for fraction in fractions:
            values = []
            if fraction in fraction_map.keys():
                avg_path_lengths.append(fraction_map[fraction])
                continue
            for i in range(5):
                try:
                    subgraph = get_fraction_of_graph(G, fraction)
                    avg_path_length_value = calculate_effective_diameter(subgraph)
                    values.append(avg_path_length_value)
                except:continue
            avg_path_lengths.append(np.mean(values))
            print(f"Fraction: {fraction:.2f}, Avg Path Length: {avg_path_length_value:.2f}")

        plt_data ={
            "fractions": fractions,
            "avg_path_length":  avg_path_lengths
        }
        
        writeinfo(os.path.join(save_dir, f"{graph_name}_avg_path.json"),
                info=plt_data)
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(fractions, avg_path_lengths, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    # plt.title('Average Path Length vs Fraction of Nodes')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Fraction of Nodes',fontsize = 18)
    plt.ylabel('$D_{e}$',fontsize = 18)
    plt.grid(True)

    # 显示图表
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_avg_path.pdf")
    plt.savefig(save_path)
    plt.clf()

if __name__ =="__main__":
    # 创建一个有向图 (DiGraph)
    G = nx.DiGraph()

    # 添加一些节点和边作为示例
    edges = [(1, 2), (2, 3), (3, 4), (4, 2), (2, 5), (5, 1), (3, 5), (5, 6), (6, 7), (7, 5)]
    G.add_edges_from(edges)
    
    plot_avg_path(G,
                  "LLMGraph/tasks/tweets/configs/llama_test_1e6/evaluate/20240418",
                  "follow")
    