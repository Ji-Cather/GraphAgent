import networkx as nx
import os
from datetime import datetime

from evaluate.matrix.base_info import calculate_effective_diameter

def calculate_gcc_size(DG:nx.DiGraph,
                       graph_name:str,
                       save_dir:str):

    try:
        subgraphs = split_graph_by_time(DG, 10)
    except:
        return
    relative_sizes = []
    diameters = []
    for subgraph in subgraphs:
        if nx.is_directed(subgraph):
            largest_cc = max(nx.strongly_connected_components(subgraph), key=len)
            relative_size = len(largest_cc) / subgraph.number_of_nodes()
        else:
            largest_cc = max(nx.connected_components(subgraph), key=len)
            relative_size = len(largest_cc) / subgraph.number_of_nodes()
        relative_sizes.append(relative_size)
        diameter = calculate_effective_diameter(largest_cc)
        diameters.append(diameter)

    plt_data ={
        "relative_size":  relative_sizes,
        "diameter": diameters
    }
    from LLMGraph.utils.io import writeinfo
    writeinfo(os.path.join(save_dir, f"{graph_name}_proportion_cc.json"),info=plt_data)


# 示例函数：根据时间属性划分图
def split_graph_by_time(G, k):
    # 获取所有节点的时间属性
    times = [datetime.strptime(G.nodes[node]['time'], '%Y-%m') for node in G.nodes]
    # 确定时间的划分点
    min_time = min(times)
    max_time = max(times)
    interval = (max_time - min_time) / k

    # 划分节点到不同子图
    subgraphs = []
    for i in range(k):
        # 当前时间区间
        upper_bound = min_time + (i + 1) * interval

        # 过滤属于当前区间的节点
        nodes_in_interval = [
            node for node in G.nodes
            if  datetime.strptime(G.nodes[node]['time'], '%Y-%m') <= upper_bound
        ]


        # 创建子图并添加相关节点和边
        subg = G.subgraph(nodes_in_interval).copy()
        subgraphs.append(subg)

    return subgraphs



if __name__ == '__main__':
    # 示例图创建及节点时间属性设置
    G = nx.Graph()
    G.add_nodes_from([
    (1, {'time': '2021-01'}), (2, {'time': '2021-02'}), 
    (3, {'time': '2021-03'}), (4, {'time': '2021-04'}),
    (5, {'time': '2021-05'}), (6, {'time': '2021-06'}), 
    (7, {'time': '2021-07'}), (8, {'time': '2021-08'})
])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1)])

    # 按时间属性划分图
    k = 3  # 例如，划分成3个子图
    subgraphs = split_graph_by_time(G, k)

    # 输出子图信息
    for i, subg in enumerate(subgraphs):
        print(f"Subgraph {i}:")
        print("Nodes:", subg.nodes(data=True))
        print("Edges:", subg.edges(data=True))
        print()
