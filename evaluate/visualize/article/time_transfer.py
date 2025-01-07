import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import networkx as nx
import os
import numpy as np
import random

import pandas as pd

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

def calculate_effective_diameter(G, percentile=0.9):
    """
    Calculate the effective diameter of the graph G.

    :param G: NetworkX graph
    :param percentage: percentage of node pairs to consider (default is 0.9 for 90th percentile)
    :return: effective diameter of the graph
    """
    # 计算所有节点对之间的最短路径长度
    if nx.is_directed(G):
        assert isinstance(G, nx.DiGraph)
        G = G.to_undirected()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # 将所有路径长度放入一个列表中
    all_lengths = []
    for source, targets in lengths.items():
        all_lengths.extend(targets.values())
        
    # 过滤掉长度为0的路径（这些是节点自己到自己的路径）
    all_lengths = [length for length in all_lengths if length > 0]
    
    # Total number of pairs
    total_pairs = len(all_lengths)
    
    # Find the smallest integer d such that g(d) >= percentile
    cumulative_distribution = np.cumsum(np.bincount(all_lengths, minlength=max(all_lengths) + 1)) / total_pairs
    d = np.searchsorted(cumulative_distribution, percentile, side='right')
    
    # Interpolate between d and d+1
    if d == 0:
        effective_diameter = 0
    else:
        g_d = cumulative_distribution[d - 1] if d > 0 else 0
        g_d_plus_1 = cumulative_distribution[d]
        if g_d_plus_1 == g_d:
            effective_diameter = d
        else:
            effective_diameter = d - 1 + (percentile - g_d) / (g_d_plus_1 - g_d)
    
    return effective_diameter

def plot_avg_path(G:nx.DiGraph,
                save_dir:str,
                graph_name:str):
    fractions = [
        0.0001,
        0.001,
        0.01,
        0.05,
        *np.linspace(0.1, 0.2, 10).tolist()
    ]
    avg_path_lengths = []

    for fraction in fractions:
        values = []
        for i in range(5):
            try:
                subgraph = get_fraction_of_graph(G, fraction)
                avg_path_length_value = calculate_effective_diameter(subgraph)
                values.append(avg_path_length_value)
            except:continue
        avg_path_lengths.append(np.mean(values))
        print(f"Fraction: {fraction:.2f}, Avg Path Length: {avg_path_length_value:.2f}")

    plt_data ={
        "out_degree": fractions,
        "avg_path_length":  avg_path_lengths
    }
    from LLMGraph.utils.io import writeinfo
    writeinfo(os.path.join(save_dir, f"{graph_name}_outdegree_avg_path.json"),
              info=plt_data)
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(fractions, avg_path_lengths, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Fraction of Nodes',fontsize=18)
    plt.ylabel('Average Path Length',fontsize=18)
    plt.grid(True)

    # 显示图表
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_avg_path.pdf")
    plt.savefig(save_path)
    plt.clf()

def plot_outdegree_cc(G:nx.DiGraph,
                      save_dir:str,
                        graph_name:str):
    # 获取每个节点的出度
    if isinstance(G, nx.DiGraph):
        out_degrees = dict(G.out_degree())
    elif isinstance(G, nx.Graph):
        out_degrees = dict(G.degree())
    # 计算每个节点的聚类系数
    clustering_coefficients = nx.clustering(nx.Graph(G))

    # 整理数据
    degree_to_clustering = {}
    
    for node, out_degree in out_degrees.items():
        if out_degree not in degree_to_clustering:
            degree_to_clustering[out_degree] = []
        degree_to_clustering[out_degree].append(clustering_coefficients[node])

    start = 10
    end = 60
    degree_to_clustering = dict(
        filter(lambda x: start <= x[0] <= end, degree_to_clustering.items())
    )
    # 计算不同出度节点的平均聚类系数
    outdegree = []
    avg_clustering = []

    for degree in sorted(degree_to_clustering):
        outdegree.append(degree)
        avg_clustering.append(np.mean(degree_to_clustering[degree]))

    plt_data ={
        "out_degree": outdegree,
        "avg_clustering": avg_clustering
    }
    from LLMGraph.utils.io import writeinfo
    writeinfo(os.path.join(save_dir, f"{graph_name}_outdegree_cc.json"),
              info=plt_data)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(outdegree, avg_clustering, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title('Average Clustering Coefficient vs Outdegree')
    plt.xlabel('Outdegree (Number of Friends)')
    plt.ylabel('Average Clustering Coefficient')
    plt.grid(True)

    # 显示图表
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_outdegree_cc.pdf")
    plt.savefig(save_path)
    plt.clf()


    