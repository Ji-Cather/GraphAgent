import networkx as nx
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress

def get_hashimoto_matrix(G:nx.DiGraph):
    if not nx.is_directed(G):
        G = G.to_directed()
    
    # 初始化Hashimoto matrix的大小为边数的平方
    size = len(G.edges())
    B = np.zeros((size, size))
    
    # 边的映射到索引
    edge_to_index = {edge: idx for idx, edge in enumerate(G.edges())}
    
    # 填充Hashimoto matrix
    for ei, (u, v) in enumerate(G.edges()):
        for vi, x in G.edges(v):
            if u != x:  # 不形成简单环的条件
                ej = edge_to_index[(v, x)]
                B[ei, ej] = 1
                
    return B

def calculate_second_avg_degree(G:nx.DiGraph):
    # 初始化二阶度总和变量
    second_order_degrees_total = 0

    # 对于图中的每一个节点，计算它的二阶度并累加
    for node in G.nodes:
        # 获取节点的邻居节点
        neighbors = nx.neighbors(G, node)
        # 计算该节点的二阶度：邻居的度数之和
        second_order_degree = sum([G.degree(neighbor) for neighbor in neighbors])
        # 累加到二阶度总和中
        second_order_degrees_total += second_order_degree

    # 计算所有节点的二阶度的平均值
    avg_second_order_degree = second_order_degrees_total / G.number_of_nodes()
    return avg_second_order_degree


def calculate_DG_base_indicators(G:nx.DiGraph,
                            graph_name = "G",
                            type="article"):

    # 计算节点的度
    degrees = G.degree()
    mean_degree = sum(dict(degrees).values()) / G.number_of_nodes()
    std_degree = nx.degree_assortativity_coefficient(G)

    # # 计算图的最大特征值
    # # 将有向图转换为无向图
    # # H = G.to_undirected()

    # # # 计算Laplacian矩阵
    # # L = nx.laplacian_matrix(H).toarray()

    # # 计算Hashimoto矩阵
    # hashimoto_matrix = get_hashimoto_matrix(G)
    # h_eigenvalues, _ = np.linalg.eig(hashimoto_matrix)
    # max_eigenvalue = max(h_eigenvalues)

    # # 计算节点的聚类系数
    # clustering_coeffs = nx.clustering(G)
    # average_clustering = sum(clustering_coeffs.values()) / G.number_of_nodes()

    if nx.is_directed(G):
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        relative_size = len(largest_cc) / G.number_of_nodes()
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        relative_size = len(largest_cc) / G.number_of_nodes()

    # 计算平均最短路径长度
    try:
        average_shortest_path_length = nx.average_shortest_path_length(G)
    except: average_shortest_path_length = np.nan

    df = pd.DataFrame()
    # 添加指标值到DataFrame
    df.loc[graph_name,'Nodes'] = len(G.nodes())
    df.loc[graph_name,'Edges'] = len(G.edges())
    df.loc[graph_name,'Mean Degree'] = mean_degree
    df.loc[graph_name,'Std Degree'] = std_degree
    df.loc[graph_name,'effective diameter'] = calculate_effective_diameter(G)
    
    # df.loc[graph_name,'Max Eigenvalue'] = np.real(max_eigenvalue)
    # df.loc["randomwalk_mixing_time"] = random_walk_mixing_time(G)
    
    # df.loc[graph_name,'Average Clustering Coefficient'] = average_clustering
    # df.loc[graph_name,'Pseudo_diameter'] = calculate_pseudo_diameter(G)
    df.loc[graph_name,'Relative Size'] = relative_size
    df.loc[graph_name,'Average Shortest Path Length'] = average_shortest_path_length

    degree_squared_sum = sum(degree**2 for node, degree in G.degree())
    # 计算度数平方的平均值
    average_degree_squared = degree_squared_sum / G.number_of_nodes()
    df.loc[graph_name,'avg_degree_squared'] = calculate_second_avg_degree(G)
    df.loc[graph_name,'2nd_avg_degree'] = calculate_second_avg_degree(G)

    # # 计算传递性
    transitivity = nx.transitivity(G)
    df.loc[graph_name,'Transitivity'] = transitivity
    global_clustering = nx.average_clustering(G)
    df.loc[graph_name,'Average_Clustering'] = global_clustering

    

    return df





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


def random_walk_mixing_time(G:nx.DiGraph):
    # 计算邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).todense()

    # 定义初始分布
    initial_distribution = np.array([1, 0, 0, 0, 0, 0])

    # 定义步数上限
    max_steps = 100

    # 模拟随机游走
    current_distribution = initial_distribution
    for step in range(max_steps):
        next_distribution = np.dot(current_distribution, adj_matrix)
        if np.allclose(current_distribution, next_distribution, atol=1e-8):
            break
        current_distribution = next_distribution

    return step
    

def calculate_pseudo_diameter(G:nx.DiGraph):# 计算伪直径
    pseudo_diameter = 0
    for node in G.nodes():
        # 计算从节点node到其他节点的最短路径长度
        path_lengths = nx.single_source_shortest_path_length(G, node)
        # 找到最长的最短路径长度
        max_path = max(path_lengths.values())
        if max_path > pseudo_diameter:
            pseudo_diameter = max_path
    return pseudo_diameter