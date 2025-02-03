import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import random


def find_driver_nodes(G):
    # 构建二分图
    B = nx.DiGraph()
    for u, v in G.edges():
        # 对于有向图的每条边从u到v，添加两个边到二分图：
        # "u_out"到"v_in" 和 "v_in"到"v_out"
        B.add_edge(f"{u}_out", f"{v}_in")
        B.add_edge(f"{v}_in", f"{v}_out")

    # 获取二分图的节点集合
    top_nodes = {n for n, d in B.nodes(data=True) if n.endswith('_out')}
    bottom_nodes = set(B) - top_nodes

    # 利用二分匹配算法找到最大匹配
    mate = bipartite.maximum_matching(B, top_nodes)  # 返回一个匹配字典

    # 计算控制节点，即在"mate"中找不到匹配的入节点
    driver_nodes = {n for n in bottom_nodes if n.endswith('_in') and not n in mate}

    # 将控制节点转换回原始图的节点
    driver_nodes = {n.split('_')[0] for n in driver_nodes}

    return driver_nodes

def calculate_control_matrix(
                            DG:nx.DiGraph,
                             graph_name:str):

    df = pd.DataFrame()
    control_nodes = find_driver_nodes(DG)
    Nc = len(control_nodes)
    
    # 找到所有源节点（入度为0）
    source_nodes = [n for n, d in DG.in_degree() if d == 0]
    Ns = len(source_nodes) # source nodes number
    # 找到所有汇节点（出度为0）
    sink_nodes = [n for n, d in DG.out_degree() if d == 0]
    Nt = len(sink_nodes) # sink nodes number
    Ne = max(0, Nt-Ns)
    Nse = Ne + Ns

    shuffled_DG = degree_preserving_shuffle(DG)
    control_nodes = find_driver_nodes(shuffled_DG)
    Nc_deg = len(control_nodes)
    N = DG.number_of_nodes()
    
    profile = {
        "nc":Nc/N,
        "nc_se":Nse/N,
        "nc_deg":Nc_deg/N
    }
    for k,v in profile.items():
        df.loc[graph_name,k] = v
    
    return df


def degree_preserving_shuffle(G):
    # 确保G是复制的，避免修改原始图
    G = G.copy()
    edges = list(G.edges())
    nodes = list(G.nodes())
    
    attempts = 0
    max_attempts = len(edges) * 10
    
    while attempts < max_attempts:
        # 随机挑选两条边进行交换
        edge1, edge2 = random.sample(edges, 2)
        new_edge1 = (edge1[0], edge2[1])
        new_edge2 = (edge2[0], edge1[1])

        # 检查是否能够进行交换而不产生自环或重边
        if new_edge1[0] != new_edge1[1] and new_edge2[0] != new_edge2[1] and \
           not G.has_edge(*new_edge1) and not G.has_edge(*new_edge2):
            G.remove_edge(*edge1)
            G.remove_edge(*edge2)
            G.add_edge(*new_edge1)
            G.add_edge(*new_edge2)
            edges.remove(edge1)
            edges.remove(edge2)
            edges.append(new_edge1)
            edges.append(new_edge2)
        attempts += 1
    
    return G


"""test code"""

import networkx as nx
import random

def preferential_attachment_directed(n, k):
    """生成一个含有 n 个节点和边数为 m 的 Barabási-Albert 有向图"""
    
    # 初始化含有 m 个节点的有向图
    l = n*k
    q = int()
    G = nx.DiGraph()
    G.add_nodes_from(range(m))
    
    # 每个节点有一个列表，保存了其出边
    target_list = list(range(m))
    
    # 从节点 m 开始，依次添加新节点
    for source in range(m, n):
        targets = _select_targets(target_list, m)
        G.add_edges_from(zip([source] * m, targets))  # 从新节点指向目标节点
        target_list.extend(targets)  # 更新目标节点列表
        target_list.extend([source] * m)  # 新节点自己也可以作为未来的目标节点
        
    return G

def _select_targets(target_list, m):
    """根据优先连接机制选择目标节点"""
    targets = set()
    while len(targets) < m:
        target = random.choice(target_list)
        targets.add(target)  # 防止选择重复的目标节点
    return list(targets)

def convert_graph(G:nx.DiGraph):
    # 创建一个新的空有向图
    G_directed = nx.DiGraph()

    # 为每条边随机分配方向
    for (u, v) in G.edges():
        if random.choice([True, False]):
            G_directed.add_edge(u, v)
        else:
            G_directed.add_edge(v, u)
    return G_directed


if __name__ =="__main__":
    # 创建一个示例有向图
    import networkx as nx
    import matplotlib.pyplot as plt


    # BA模型参数
    n = 100  # 节点数
    m = 2 # 每次加入的节点数，这些新节点会尝试与现有节点建立m个边
    n = 1000
    m = 3

    graph_number = 10
    # 使用networkx生成BA模型图
    nc = []
    nse = []
    nc_deg = []
    graph_name='BA'
    for i in range(graph_number):
        ba_graph = nx.barabasi_albert_graph(n, m)
        DG = convert_graph(ba_graph)

        # # 计算控制节点数量
        df = calculate_control_matrix(DG, graph_name)
        nc.append(df.loc[graph_name,'nc'])
        nse.append(df.loc[graph_name,'nc_se'])
        nc_deg.append(df.loc[graph_name,'nc_deg'])

    from evaluate.visualize.article import plot_nc
    plot_nc(nc,
            nse,
            nc_deg,
            "article_graph",
            save_dir="/mnt2/jijiarui/LLM4Graph")
    
    # 参数：节点总数和每个新节点的出边数
    # n = 100
    # k = 10
    # DG = preferential_attachment_directed(n, k)
    # df = calculate_control_matrix(DG, 'BA_directed')
    # df
    