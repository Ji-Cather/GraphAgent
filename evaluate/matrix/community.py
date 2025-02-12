import networkx as nx
import pandas as pd
import os
import community
from community import community_louvain
from networkx.algorithms.community.quality import modularity
import numpy as np
def get_graph_matrix(DG:nx.DiGraph,
                       graph_name:str):
    df = pd.DataFrame()
    # 计算图的传递性
    # triadic_closure = nx.transitivity(DG)
    # # df.loc[graph_name,"triadic_closure"] = triadic_closure
    # # df.loc[graph_name,"assortativity"] = nx.attribute_assortativity_coefficient(DG, 
    # #                                         'action_type')


    # modularity, community_sizes_map = count_community(DG)
    # df.loc[graph_name,"modularity"] = modularity

    # # 计算网络的相关系数
    # assortativity = nx.degree_assortativity_coefficient(DG)
    # df.loc[graph_name,'assortativity'] = assortativity
    # num_nodes = 10000
    # DG = DG.subgraph(list(DG.nodes)[:num_nodes])
    df.loc[graph_name,'homophily'] = compute_edge_homophily_ratio(DG,graph_name)


    return df



def compute_edge_homophily_ratio(graph:nx.Graph,
                                 graph_name):
    """
    计算图的同类别边比例。

    :param graph: networkx 图对象
    :param labels: 字典，节点的标签映射，格式为 {node: label}
    :return: 同类别边比例
    """
    same_label_edge_count = 0
    total_edge_count = graph.number_of_edges()
    if graph_name not in ["author_citation","article_citation","co_authorship"]:
        return np.nan

    for u, v in graph.edges():
        found_same = False  
        if graph_name == "co_authorship":
            if graph.nodes(data =True)[u]["topics"] == \
                graph.nodes(data =True)[v]["topics"]:
                found_same = True
        if "author_citation" == graph_name:
            
            for t_u in graph.nodes(data =True)[u]["topics"]:
                for t_v in graph.nodes(data =True)[v]["topics"]:
                    if t_u == t_v:
                        found_same = True
                        break
        elif graph_name == "article_citation":
            if graph.nodes(data =True)[u]["topic"] == \
                graph.nodes(data =True)[v]["topic"]:
                found_same = True
                
        if found_same:
            same_label_edge_count += 1

    return same_label_edge_count / total_edge_count if total_edge_count else np.nan



def count_community(DG:nx.Graph):
    if isinstance(DG, nx.DiGraph):
        DG = DG.to_undirected()
    # 使用Louvain方法找到最佳社区划分
    partition = community_louvain.best_partition(DG)

    # 计算这种划分的modularity
    modularity = community_louvain.modularity(partition, DG)

    # 计算各个社区的大小
    community_sizes_map = {}
    for community in set(partition.values()):
        # 统计每个社区的节点数
        size = list(partition.values()).count(community)
        if size not in community_sizes_map.keys():
            community_sizes_map[size] = []
        community_sizes_map[size].append(community)
    return modularity, community_sizes_map
    
if __name__ == "__main__":
    # 构建示例图
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    labels = {1: 'A', 2: 'B', 3: 'A', 4: 'A', 5: 'B'}

    # 计算同类别边比例
    homophily_ratio = compute_edge_homophily_ratio(G, labels)
    modularity_ratio, community_sizes_map = count_community(G)
    print(f"同类别边比例: {homophily_ratio}")
    print(f"同类别边比例: {modularity_ratio}")