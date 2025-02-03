import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import networkx as nx
import numpy as np
import os

def plot_friend_degree(G:nx.DiGraph,
                       save_dir:str,
                       graph_name:str):
    
    # 计算每个节点邻居的平均度数
    if graph_name == "follow":
        avg_neighbor_degrees = nx.average_neighbor_degree(G,
                                source="out", target="out")
        degrees = dict(G.out_degree())
    elif graph_name == "friend":
        avg_neighbor_degrees = nx.average_neighbor_degree(G)
        degrees = dict(G.degree())
    elif graph_name == "action":
        degrees = dict(G.out_degree())
        avg_neighbor_degrees = nx.average_neighbor_degree(G,
                                source="out", target="out")

    # 准备绘图数据：节点度数和对应的邻居平均度数
    x = [degrees[node] for node in G.nodes()]
    y = [avg_neighbor_degrees[node] for node in G.nodes()]
    # 添加 y=x 虚线
    max_degree = max(x)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.plot([0, max_degree], [0, max_degree], 'k--', label='y=x')
    # 对数刻度可以更好地理解数据分布，但取决于您的具体数据
    # plt.xscale('log')
    # plt.yscale('log')

    # plt.title('Node Degree vs. Average Neighbor Degree')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Node Degree',fontsize = 18)
    plt.ylabel('Average Neighbor Degree',fontsize = 18)

    save_path = os.path.join(save_dir, f"{graph_name}_neighbour_degree.pdf")
    # 显示图表
    plt.savefig(save_path)
