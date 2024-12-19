import networkx as nx
import os
from datetime import datetime
from evaluate.matrix.base_info import calculate_effective_diameter
from networkx.algorithms import bipartite
def get_projected_graph(G_generated:nx.DiGraph):
    user_nodes = set(n for n,d in G_generated.nodes(data=True) if "watcher" in n)
    # 计算用户节点的投影图
    user_projection = bipartite.projected_graph(G_generated, user_nodes)
    return user_projection

def calculate_gcc_size(DG:nx.DiGraph,
                       graph_name:str,
                       save_dir:str):
    try:
        subgraphs = split_graph_by_time(DG, 10)
    except Exception as e:
        print(e)
        return
    relative_sizes = []
    diameters = []
    user_relative_sizes = []
    user_diameters = []

    for subgraph in subgraphs:
        try:
            if nx.is_directed(subgraph):
                largest_cc = max(nx.strongly_connected_components(subgraph), key=len)
                relative_size = len(largest_cc) / subgraph.number_of_nodes()
            else:
                largest_cc = max(nx.connected_components(subgraph), key=len)
                relative_size = len(largest_cc) / subgraph.number_of_nodes()
            largest_cc = subgraph.subgraph(largest_cc)
            relative_sizes.append(relative_size)
            diameters.append(calculate_effective_diameter(largest_cc))
            user = get_projected_graph(subgraph)
            if nx.is_directed(user):
                largest_cc = max(nx.strongly_connected_components(user), key=len)
                relative_size = len(largest_cc) / user.number_of_nodes()
            else:
                largest_cc = max(nx.connected_components(user), key=len)
                relative_size = len(largest_cc) / user.number_of_nodes()
            largest_cc = subgraph.subgraph(largest_cc)
            user_relative_sizes.append(relative_size)
            user_diameters.append(calculate_effective_diameter(largest_cc))
        except:
            continue

    plt_data ={
        "relative_size":  relative_sizes,
        "diameter": diameters
    }
    from LLMGraph.utils.io import writeinfo
    writeinfo(os.path.join(save_dir, f"{graph_name}_proportion_cc.json"),info=plt_data)

    plt_data ={
        "relative_size":  user_relative_sizes,
        "diameter": user_diameters
    }
    from LLMGraph.utils.io import writeinfo
    writeinfo(os.path.join(save_dir, f"user_projection_proportion_cc.json"),info=plt_data)


# 示例函数：根据时间属性划分图
def split_graph_by_time(G, k):
    # 获取所有节点的时间属性
    try:
        for u, v, data in G.edges(data=True):
            data['timestamp'] = datetime.strptime(data['timestamp'], "%Y%m%d")
    except:
        pass
    # 获取所有边的时间并排序
    times = [data['timestamp'] for _, _, data in G.edges(data=True)]
    times.sort()


    # 创建子图字典
    subgraphs = {i: nx.Graph() for i in range(k)}

    # 计算每一份的时间间隔
    time_diff = (times[-1] - times[0]) / k
    # 将边划分到不同的子图
    for u, v, data in G.edges(data=True):
        time = data['timestamp']
        for i in range(k):
            if time < times[0] + (i + 1) * time_diff:
                subgraphs[i].add_edge(u, v, time=time)
                break

    # 考虑到最后一份可能包含边界问题，将属于最后一份的边加入
    for u, v, data in G.edges(data=True):
        time = data['timestamp']
        if time >= times[-1] - time_diff:
            subgraphs[k - 1].add_edge(u, v, time=time)

    return list(subgraphs.values())



if __name__ == '__main__':
    # 示例图创建及节点时间属性设置
    # 创建有向图
    G = nx.DiGraph()

    # 添加带有时间属性的边 (时间格式为 mm/dd/yy)
    edges = [
        (1, 2, "01/01/21"),
        (2, 3, "02/15/21"),
        (1, 3, "01/20/21"),
        (3, 4, "03/10/21"),
        (2, 4, "02/25/21"),
        (4, 5, "04/05/21"),
        (5, 6, "05/22/21"),
        (3, 6, "03/15/21"),
        (1, 5, "04/22/21")
    ]

    for u, v, time in edges:
        G.add_edge(u, v, time=time)


    # 按时间属性划分图
    k = 3  # 例如，划分成3个子图
    subgraphs = split_graph_by_time(G, k)

    # 输出子图信息
    for i, subg in enumerate(subgraphs):
        print(f"Subgraph {i}:")
        print("Nodes:", subg.nodes(data=True))
        print("Edges:", subg.edges(data=True))
        print()
