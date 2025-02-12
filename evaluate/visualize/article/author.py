
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import numpy as np

import os
def plot_degree_compare(graph_maps:dict,
                save_root:str):
    
    fig, ax = plt.subplots(1, len(graph_maps), figsize=(14, 6))
    data_combined = []
    for graph in graph_maps.values():
        data_combined += [d for n, d in graph.degree()]
    
    bins = np.linspace(min(data_combined), max(data_combined), 30)

    for idx, graph_name in enumerate(graph_maps.keys()):
        graph = graph_maps[graph_name]
        assert isinstance(graph, nx.DiGraph)
        # 获取节点的度
        degrees_G = list(dict(graph.in_degree(weight='count')).values())
        # 绘制度的概率密度分布函数
        ax[idx].hist(degrees_G, bins=bins, density=True, alpha=0.6, color='b')
        ax[idx].set_title(f'{graph_name} Graph Degree PDF')
        ax[idx].set_xlabel('In Degree')
        ax[idx].set_ylabel('Density')


from igraph import Graph, plot

# 将NetworkX图转换为igraph图
def convert_networkx_to_igraph(nx_graph):
    g = Graph(directed=nx_graph.is_directed())
    g.add_vertices(sorted(nx_graph.nodes()))
    g.add_edges(nx_graph.edges())
    weights =[]
    for e_idx,e_info in nx_graph.edges().items():
        weights.append(e_info["count"])
    g.es['weight'] = weights
    return g

def plot_igraph_compare(graph_maps:dict,
                save_root:str):
    
    root = save_root
    if not os.path.exists(root):os.makedirs(root)
    fig, ax = plt.subplots(1, len(graph_maps), figsize=(14, 6))
    data_combined = []
    for graph in graph_maps.values():
        data_combined += [d for n, d in graph.degree()]
    
    bins = np.linspace(min(data_combined), max(data_combined), 30)

    for idx, graph_name in enumerate(graph_maps.keys()):
        graph = graph_maps[graph_name]
        # 转换图并进行可视化
        igraph_graph = convert_networkx_to_igraph(graph)
        # 定义igraph的可视化风格
        visual_style = {}
        visual_style["vertex_size"] = 5  # 将大小设置小一些，因为节点可能会非常多
        visual_style["vertex_label"] = None  # 太多的节点，标签可能会导致无法阅读
        visual_style["edge_width"] = [ w/5 for w in igraph_graph.es['weight']]  # 你可以调整权重的影响
        # visual_style["edge_color"] = ['red' if bipartite == 1 else 'blue' for bipartite in igraph_graph.vs['bipartite']]
        # visual_style["layout"] = igraph_graph.layout('bipartite', types=igraph_graph.vs['bipartite'])
        # visual_style["layout"] = igraph_graph.layout("fr")
        root_id = list(graph.nodes().keys())[0]
        visual_style["layout"] = igraph_graph.layout("fr")
        plot(igraph_graph,ax[idx], **visual_style)
    
    plt.tight_layout()
    save_path = os.path.join(save_root,f"author_igraph.pdf")
    plt.savefig(save_path)
