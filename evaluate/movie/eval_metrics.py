import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import torch
import json
from scipy.stats import norm
import matplotlib.pyplot as plt


from functools import cmp_to_key
import pandas as pd

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1. / (2. * sigma ** 2)
    dist = torch.sum(x**2, 1).unsqueeze(1) + torch.sum(y**2, 1) - 2 * torch.matmul(x, y.t())
    return torch.exp(-beta * dist)

def compute_mmd(x, y, sigma=1.0):
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    res = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return res.tolist()

def compute_nll(true_samples, generated_samples):
    # 使用真实样本估计正态分布参数
    mu = np.mean(true_samples)
    sigma = np.std(true_samples)
    
    # 计算生成样本的NLL
    nll = -np.mean(norm.logpdf(generated_samples, loc=mu, scale=sigma))
    return nll



def calculate_matrix(graph_true, graph_generated):
    df = pd.DataFrame()

    index_generated = 'generated'
    index_true ='true'
    
    # 平均度分布相似度
    degrees_true = [degree for _, degree in graph_true.degree()]
    degrees_generated = [degree for _, degree in graph_generated.degree()]
    hist_true, _ = np.histogram(degrees_true, bins=max(degrees_true), range=(0, max(degrees_true)))
    hist_generated, _ = np.histogram(degrees_generated, bins=max(degrees_generated), range=(0, max(degrees_generated)))
    
    df.loc[index_generated,'degree Distribution JS Divergence']= jensenshannon(hist_true, hist_generated)

    # 聚类系数
   
    df.loc[index_generated,'mdd_degree'] = compute_mmd(hist_true,hist_generated)

    # # 平均最短路径长度
    # if nx.is_connected(graph_true):
    #     matrix['Average Shortest Path Length True'] = nx.average_shortest_path_length(graph_true)
    # if nx.is_connected(graph_generated):
    #     matrix['Average Shortest Path Length Generated'] = nx.average_shortest_path_length(graph_generated)

    # # 图直径
    # if nx.is_connected(graph_true):
    #     matrix['Diameter True'] = nx.diameter(graph_true)
    # if nx.is_connected(graph_generated):
    #     matrix['Diameter Generated'] = nx.diameter(graph_generated)

    # 图编辑距离（对于大图可能不适用，因为计算成本高）
    # matrix['Graph Edit Distance'] = graph_edit_distance(graph_true, graph_generated)

    return df


def calculate_directed_graph_matrix(graph_true:nx.Graph, 
                                     graph_generated:nx.Graph) -> pd.DataFrame:
    df = pd.DataFrame()

    index_generated = 'generated'
    index_true ='true'
    
    # 确定两个图的最大入度和出度
    max_in_degree = max(max(d for n, d in graph_true.in_degree()), max(d for n, d in graph_generated.in_degree()))
    max_out_degree = max(max(d for n, d in graph_true.out_degree()), max(d for n, d in graph_generated.out_degree()))

    # 使用相同的bins设置计算入度和出度的直方图
    bins_in = np.arange(0, max_in_degree + 2)  # 加2以确保最大度数包含在内
    bins_out = np.arange(0, max_out_degree + 2)

    in_degrees_true = [d for n, d in graph_true.in_degree()]
    out_degrees_true = [d for n, d in graph_true.out_degree()]
    in_degrees_generated = [d for n, d in graph_generated.in_degree()]
    out_degrees_generated = [d for n, d in graph_generated.out_degree()]

    in_hist_true, _ = np.histogram(in_degrees_true, bins=bins_in)
    in_hist_generated, _ = np.histogram(in_degrees_generated, bins=bins_in)
    out_hist_true, _ = np.histogram(out_degrees_true, bins=bins_out)
    out_hist_generated, _ = np.histogram(out_degrees_generated, bins=bins_out)
    
    
    df.loc[index_generated,'In-degree Distribution JS Divergence'] = jensenshannon(in_hist_true, in_hist_generated)
    df.loc[index_generated,'Out-degree Distribution JS Divergence'] = jensenshannon(out_hist_true, out_hist_generated)

    # 有向图的平均聚类系数
    df.loc[index_generated,'mdd_in_degree'] = compute_mmd(in_hist_true,in_hist_generated)
    df.loc[index_generated,'mdd_out_degree'] = compute_mmd(out_hist_true,out_hist_generated)
    if len(graph_generated.nodes()) == len(graph_true.nodes()):
        df.loc[index_generated,'mdd_clustering'] = compute_mmd(list(nx.clustering(graph_true).values()),
                                                               list(nx.clustering(graph_generated).values()))
   
    # # 平均最短路径长度（仅当图是强连通时）
    # if nx.is_strongly_connected(graph_true):
    #     matrix['Average Shortest Path Length True'] = nx.average_shortest_path_length(graph_true)
    # if nx.is_strongly_connected(graph_generated):
    #     matrix['Average Shortest Path Length Generated'] = nx.average_shortest_path_length(graph_generated)

    # # 图直径（仅当图是强连通时）
    # if nx.is_strongly_connected(graph_true):
    #     matrix['Diameter True'] = nx.diameter(graph_true)
    # if nx.is_strongly_connected(graph_generated):
    #     matrix['Diameter Generated'] = nx.diameter(graph_generated)

    return df

def plot_figures(G:nx.DiGraph,
                 save_dir:str,
                 graph_name:str):
    plot_pk_k(G,save_dir,graph_name)
    plot_k_t(G,save_dir,graph_name)
    plot_shrinking_diameter(G,n=100,save_dir=save_dir,graph_name=graph_name)
    
    
def plot_degree_figures(G:nx.Graph,
                 save_dir:str,
                 graph_name:str):
    plot_pk_k_movie(G,save_dir,graph_name)
    plot_k_t_movie(G,save_dir,graph_name)
    
def plot_pk_k_movie(G:nx.Graph,
            save_dir:str,
            graph_name:str):
    # 计算所有节点的出度
    in_degrees = [G.in_degree(n) for n in G.nodes()]

    # 计算度分布
    degree_counts = np.bincount(in_degrees)
    degrees = np.arange(len(degree_counts))

    # 移除度为0的点
    degrees = degrees[degree_counts > 0]
    degree_counts = degree_counts[degree_counts > 0]


    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, degree_counts, 'bo-', linewidth=2, markersize=8)
    plt.title('Log-Log Plot of Movie Ratings Degree Distribution')
    plt.xlabel('Degree (k)')
    plt.ylabel('Frequency (P(k))')
    plt.grid(True, which="both", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_degree.pdf"))
    plt.clf()
    

def plot_pk_k(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    # 计算所有节点的出度
    in_degrees = [G.in_degree(n) for n in G.nodes()]

    # 计算度分布
    degree_counts = np.bincount(in_degrees)
    degrees = np.arange(len(degree_counts))

    # 移除度为0的点
    degrees = degrees[degree_counts > 0]
    degree_counts = degree_counts[degree_counts > 0]


    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, degree_counts, 'bo-', linewidth=2, markersize=8)
    plt.title('Log-Log Plot of Paper Citation Network Degree Distribution')
    plt.xlabel('Degree (k)')
    plt.ylabel('Frequency (P(k))')
    plt.grid(True, which="both", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_degree.pdf"))
    plt.clf()

def print():
    import json
    with open("LLMGraph/tasks/llm_agent/data/article_meta_info.json",'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    cites = [ value["cited"] for value in data_list.values()]
    
    # 计算度分布
    degree_counts = np.bincount(cites)
    degrees = np.arange(len(degree_counts))

    # 移除度为0的点
    # degrees = degrees[degree_counts > 0]
    # degree_counts = degree_counts[degree_counts > 0]


    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, degree_counts, 'bo-', linewidth=2, markersize=8)
    plt.title('Log-Log Plot of Paper Citation Network Degree Distribution')
    plt.xlabel('Degree (k)')
    plt.ylabel('Frequency (P(k))')
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.savefig("degree.pdf")
    plt.clf()
    
    


# 定义比较函数
def compare_items(item1, item2):
    item1 = item1[1]
    item2 = item2[1]
    # 检查time属性，如果都存在，则直接比较time
    if item1.get("time")is not None and item2.get("time") is not None:
        return (item1.get("time")> item2.get("time")) - (item1.get("time")< item2.get("time"))
    # 如果item1的time不存在，而item2的time存在，认为item1小于item2
    elif item1.get("time")is None and item2.get("time")is not None:
        return 1
    # 如果item2的time不存在，而item1的time存在，认为item1大于item2
    elif item1.get("time")is not None and item2.get("time")is None:
        return -1
    # 如果两者的time都不存在，比较round_id
    else:
        return (item1.get("round_id") > item2.get("round_id")) - \
            (item1.get("round_id") < item2.get("round_id"))
            
def compare_items_movie(item1, item2):
    item1 = item1[1]
    item2 = item2[1]
    # 检查time属性，如果都存在，则直接比较time
    if item1.get("time")is not None and item2.get("time") is not None:
        return (item1.get("time")> item2.get("time")) - (item1.get("time")< item2.get("time"))
    # 如果item1的time不存在，而item2的time存在，认为item1小于item2
    elif item1.get("time")is None and item2.get("time")is not None:
        return 1
    # 如果item2的time不存在，而item1的time存在，认为item1大于item2
    elif item1.get("time")is not None and item2.get("time")is None:
        return -1
    # 如果两者的time都不存在，比较round_id
    else:
        return (item1.get("round_id") > item2.get("round_id")) - \
            (item1.get("round_id") < item2.get("round_id"))

def plot_k_t_movie(G:nx.Graph,
            save_dir:str,
            graph_name:str):
    nodes = G.nodes()
    
    # 按照node的time进行排序

    nodes = sorted(nodes._nodes.items(), key=cmp_to_key(compare_items_movie))
    
    node_indexs = [node[0] for node in nodes]
    node_indexs.reverse()
    # 计算度分布
    degrees = [G.in_degree(n) for n in node_indexs]
    
    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(node_indexs)), degrees, 'bo-', linewidth=2, markersize=8)
    plt.title('t - k')
    plt.xlabel('t')
    plt.ylabel('k')
    plt.grid(True, which="both", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_t_k.pdf"))
    plt.clf()

def plot_k_t(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    nodes = G.nodes()
    
    # 按照node的time进行排序

    nodes = sorted(nodes._nodes.items(), key=cmp_to_key(compare_items))
    
    node_indexs = [node[0] for node in nodes]
    node_indexs.reverse()
    # 计算度分布
    degrees = [G.in_degree(n) for n in node_indexs]
    
    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(node_indexs)), degrees, 'bo-', linewidth=2, markersize=8)
    plt.title('t - k')
    plt.xlabel('t')
    plt.ylabel('k')
    plt.grid(True, which="both", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_t_k.pdf"))
    plt.clf()
    
    
def plot_k_t_llm_agent():
    articles = readinfo("LLMGraph/tasks/llm_agent/data/article_meta_info.json")
    # 按照node的time进行排序

    nodes = sorted(articles.items(), key=cmp_to_key(compare_items))
    
    # node_indexs = [node[0] for node in nodes]
    # node_indexs.reverse()
    # 计算度分布
    degrees = [n[1]["cited"] for n in nodes]
    times =[n[1]["time"] for n in nodes]
    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.plot(times, degrees, 'bo-', linewidth=2, markersize=8)
    plt.title('t - k')
    plt.xlabel('t')
    plt.ylabel('k')
    
    plt.xticks(times,times,rotation = 45)
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.savefig(os.path.join("LLMGraph/tasks/llm_agent/data",f"llmagent_t_k.pdf"))
    plt.clf()
    
    
def divide_list(lst, k):
    # 计算每份的基本长度
    n = len(lst)
    base_size = n // k
    remainder = n % k
    
    # 创建结果列表
    result = []
    
    # 记录当前位置
    start = 0
    
    for i in range(k):
        # 如果还有余数，当前分组加1
        end = start + base_size + (1 if i < remainder else 0)
        # 将当前分组添加到结果列表
        result.append(lst[start:end])
        # 更新下一分组的起始位置
        start = end
    
    return result
    
def plot_shrinking_diameter(DG:nx.DiGraph,
                            n:int = 100,
                            save_dir:str= "",
                            graph_name:str = ""):
    nodes = DG.nodes()
    nodes = sorted(nodes._nodes.items(), key=cmp_to_key(compare_items)) # 早->晚
    
    nodes_chunk = divide_list(nodes,n)
    
    sub_graphs = []
    sub_nodes = []
    for node_chunk in nodes_chunk:
        sub_nodes.extend([node[0] for node in node_chunk])
        sub_graph = DG.subgraph(sub_nodes).copy()
        sub_graphs.append(sub_graph)
    
    diameters =[]
    for sub_graph in sub_graphs:
        diameter = calculate_diameter(sub_graph)
        diameters.append(diameter)
    
    # 假设直径d随时间t缩减，这里我们使用一个简单的例子：d = e^(-t)
    t = np.linspace(0, len(nodes), n)  # 时间从0到5，总共100个点

    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(t, diameters, label='Shrinking Diameter')  # 绘制直径随时间变化的图表
    plt.title('Diameter Shrinking Over Time',fontsize=20)  # 图表标题
    # plt.xlabel('Time')  # x轴标签
    plt.ylabel('Diameter',fontsize=20)  # y轴标签
    plt.legend(fontsize=20)  # 显示图例
    plt.xticks(fontsize=14)  # 设置x轴刻度的字体大小
    plt.yticks(fontsize=14)  # 设置x轴刻度的字体大小
    plt.grid(True)  # 显示网格
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_diameter.pdf"))
    plt.clf()

def calculate_diameter(DG:nx.DiGraph):
    lengths = dict(nx.all_pairs_shortest_path_length(DG))

    # 初始化直径为0
    diameter = 0

    # 遍历所有路径长度，找出最大值
    for source, targets in lengths.items():
        for target, length in targets.items():
            if length > diameter:
                diameter = length
    return diameter

if __name__ =="__main__":
    # 示例：评估有向图
    # G_true = nx.fast_gnp_random_graph(100, 0.05, directed=True)
    # G_generated = nx.fast_gnp_random_graph(100, 0.05, directed=True)

    # matrix = calculate_directed_graph_matrix(G_true, G_generated)
    # for metric, value in matrix.items():
    #     print(f"{metric}: {value}")
    
    # plot_figures()
    # print()
    plot_k_t_llm_agent()
