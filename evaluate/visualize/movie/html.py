import networkx as nx
import os

# import pygraphviz as pgv
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import networkx as nx
from igraph import Graph, plot

# 将NetworkX图转换为igraph图
def convert_networkx_to_igraph(nx_graph,top_nodes):
    g = Graph(directed=nx_graph.is_directed())
    g.add_vertices(sorted(nx_graph.nodes()))
    g.add_edges(nx_graph.edges())
    g.vs['bipartite'] = [1 if node in top_nodes else 0 for node in g.vs['name']]
    weights =[]
    for e_idx,e_info in nx_graph.edges().items():
        rating = 0
        if isinstance(e_info["rating"],list):
            rating =  float(e_info["rating"][0])
        else:
            rating = float(e_info["rating"])
        weights.append(rating)
    g.es['weight'] = weights
    return g

def create_movie_visualize(G:nx.Graph,
                           save_path:str = "visualize/movie/bipartite_graph.pdf"):
    
    ### filter degree =0 in DG
    # 创建一个新的图H，包含G中度数大于0的节点
    H = nx.Graph()
    for node in G.nodes():
        if G.degree(node) > 0:
            # 将节点及其属性添加到新图
            H.add_node(node, **G.nodes[node])
            # 将与该节点相关的边添加到新图
            for neighbor in G.neighbors(node):
                H.add_edge(node, neighbor, **G.get_edge_data(node, neighbor))
    G =  H
    root = os.path.dirname(save_path)
    if not os.path.exists(root):os.makedirs(root)
    # 生成并显示图
    
    top_nodes = list(filter(lambda node: node[1]['bipartite'] == 0,
                       G.nodes().items()))
    top_nodes = [node[0] for node in top_nodes]

    # 转换图并进行可视化
    igraph_graph = convert_networkx_to_igraph(G,top_nodes)

   # 定义igraph的可视化风格
    visual_style = {}
    visual_style["vertex_size"] = 5  # 将大小设置小一些，因为节点可能会非常多
    visual_style["vertex_label"] = None  # 太多的节点，标签可能会导致无法阅读
    visual_style["edge_width"] = [ w/5 for w in igraph_graph.es['weight']]  # 你可以调整权重的影响
    visual_style["edge_color"] = ['red' if bipartite == 1 else 'blue' for bipartite in igraph_graph.vs['bipartite']]
    # visual_style["layout"] = igraph_graph.layout('bipartite', types=igraph_graph.vs['bipartite'])
    # visual_style["layout"] = igraph_graph.layout("fr")
    visual_style["layout"] = igraph_graph.layout("drl")

    # 绘制igraph图
    plot(igraph_graph,save_path, **visual_style)

if __name__ == "__main__":
    create_movie_visualize()