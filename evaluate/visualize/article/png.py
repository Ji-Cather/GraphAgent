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
from evaluate.utils import readinfo
# 将NetworkX图转换为igraph图
def convert_networkx_to_igraph(nx_graph):
    g = Graph(directed=nx_graph.is_directed())
    g.add_vertices(sorted(nx_graph.nodes()))
    g.add_edges(nx_graph.edges())
    
    # weights =[]
    # for e_idx,e_info in nx_graph.edges().items():
    #     rating = 0
    #     if isinstance(e_info["rating"],list):
    #         rating =  float(e_info["rating"][0])
    #     else:
    #         rating = float(e_info["rating"])
    #     weights.append(rating)
    # g.es['weight'] = weights
    return g

def create_article_visualize_nx(G_true, G_generated, root_dir):
    
    # 绘制图
    nx.draw(G_true, with_labels=True, node_color='lightblue', edge_color='gray')
    # 保存图形到文件
    plt.savefig(f"{root_dir}/graph_true.pdf")  # 可以更换为.jpg或其他支持的格式
    plt.clf()
    
    nx.draw(G_generated, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.savefig(f"{root_dir}/graph_generated.pdf")  # 可以更换为.jpg或其他支持的格式
    plt.clf()


def create_article_visualize(G:nx.Graph,
                           save_path:str = "LLMGraph/visualize/article/graph.pdf"):
    
    ### filter degree =0 in DG
    # 创建一个新的图H，包含G中度数大于0的节点
    # H = nx.Graph()
    # for node in G.nodes():
    #     if G.degree(node) > 0:
    #         # 将节点及其属性添加到新图
    #         H.add_node(node, **G.nodes[node])
    #         # 将与该节点相关的边添加到新图
    #         for neighbor in G.neighbors(node):
    #             H.add_edge(node, neighbor, **G.get_edge_data(node, neighbor))
    # G =  H
    root = os.path.dirname(save_path)
    if not os.path.exists(root):os.makedirs(root)
    # 生成并显示图
    
   

    # 转换图并进行可视化
    igraph_graph = convert_networkx_to_igraph(G)

   # 定义igraph的可视化风格
    visual_style = {}
    visual_style["vertex_size"] = 5  # 将大小设置小一些，因为节点可能会非常多
    visual_style["vertex_label"] = None  # 太多的节点，标签可能会导致无法阅读
    # visual_style["edge_width"] = [ w/5 for w in igraph_graph.es['weight']]  # 你可以调整权重的影响
    # visual_style["edge_color"] = ['red' if bipartite == 1 else 'blue' for bipartite in igraph_graph.vs['bipartite']]
    # visual_style["layout"] = igraph_graph.layout('bipartite', types=igraph_graph.vs['bipartite'])
    # visual_style["layout"] = igraph_graph.layout("fr")
    root_id = list(G.nodes().keys())[0]
    visual_style["layout"] = igraph_graph.layout("reingold_tilford")
    #visual_style["layout"] = igraph_graph.layout("fruchterman_reingold")

    # 绘制igraph图
    plot(igraph_graph,save_path, **visual_style)

from collections import defaultdict
import numpy as np

def segment_data(data):
    # 计算每篇文章的平均重要性
    average_importance = {paper: np.mean(attributes['importance'])
                        for paper, attributes in data.items()}

    # 按平均重要性，从高到低排序文章
    sorted_papers_by_importance = sorted(average_importance.items(), key=lambda item: -item[1])

    # 分割列表到5等分，每段大约包含total/5篇文章
    total_papers = len(sorted_papers_by_importance)
    papers_per_segment = total_papers // 5
    segments = [sorted_papers_by_importance[i * papers_per_segment: (i + 1) * papers_per_segment]
                for i in range(5)]

    # 如果不是5的整数倍，我们需要将剩余的文章分配到已有的分段
    remaining_papers = sorted_papers_by_importance[papers_per_segment * 5:]
    for index, paper in enumerate(remaining_papers):
        segments[index].append(paper)

    # 打印分好段的文章 (只打印文章标题）
    return segments

def plot_reason_visualize(reason_info_path,
                     save_dir:str):
    reason_info = readinfo(reason_info_path)
    segments = segment_data(reason_info)

    # for i, segment in enumerate(segments):
    #     print(f"\nSegment {i+1}:")
    #     segment_keys = [item[0] for item in segment]
    #     cite_plot_reason_visualize(segment_keys,
    #                           reason_info=reason_info,
    #                           save_dir=os.path.join(save_dir, f"segment_{i+1}"),
    #                           save = True
    #                           )
        
    # cite_plot_reason_visualize(list(reason_info.keys()),
    #                       reason_info=reason_info,
    #                       save_dir=os.path.join(save_dir, "all"))
    plot_reason(segments,reason_info,save_dir)
    plot_section(segments,reason_info,save_dir)
    cite_importance_visualize(list(reason_info.keys()),reason_info,save_dir)


def plot_reason(segments,
                reason_info,
                save_dir:str):
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        segment_keys = [item[0] for item in segment]
        cite_plot_reason_visualize(segment_keys,
                              reason_info=reason_info,
                              save_dir=os.path.join(save_dir, f"segment_{i+1}"),
                              save = False
                              )
    cite_plot_reason_visualize(list(reason_info.keys()),
                          reason_info=reason_info,
                          save_dir=os.path.join(save_dir, "all"),
                          save=True)
    
def plot_section(segments,
                reason_info,
                save_dir:str):
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        segment_keys = [item[0] for item in segment]
        cite_section_visualize(segment_keys,
                              reason_info=reason_info,
                              save_dir=os.path.join(save_dir, f"segment_{i+1}"),
                              save = False
                              )
    cite_section_visualize(list(reason_info.keys()),
                          reason_info=reason_info,
                          save_dir=os.path.join(save_dir, "all"),
                          save=True)

def cite_plot_reason_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    reason_map ={
        "1":"Background",
        "2":"Fundamental idea",
        "3":"Technical basis",
        "4":"Comparison"
    }
    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    reason_distribution = defaultdict(int)
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        for reason, count in attributes['reason'].items():
            reason_distribution[reason] += count
        for importance in attributes['importance']:
            importance_distribution[importance] += 1

    reason_distribution = dict(sorted(reason_distribution.items(), key=lambda item: item[0]))
    
    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    print("Reason Distribution:", dict(reason_distribution))
    
    total_reason_count = sum(reason_distribution.values())
    reason_distribution = {reason: count / total_reason_count for reason, count in reason_distribution.items()}
        # And plot the importance distribution

    
    reason_keys = [reason_map.get(reason) for reason in reason_distribution.keys()]
    label = 'Importance score {:.2f}'.format(
                    avg_importance)
    if save: label = "All: " + label
    plt.plot(reason_keys, 
             list(reason_distribution.values()), 
             marker='o',
             label=label)
    
    plt.title('Reason Distribution')
    plt.xlabel('Reason')
    plt.ylabel('Frequency')
    
    save_path = os.path.join(save_dir, "reason_distribution.pdf")
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
    else:
        return


def cite_section_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    section_map ={
        "1": "Introduction",
        "2": "Related work",
        "3": "Method",
        "4": "Experiment"
    }
    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    section_distribution = defaultdict(int)
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        for reason, count in attributes['section'].items():
            section_distribution[reason] += count
        for importance in attributes['importance']:
            importance_distribution[importance] += 1

    section_distribution = dict(sorted(section_distribution.items(), key=lambda item: item[0]))
    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    print("Section Distribution:", dict(section_distribution))
    # And plot the importance distribution
    total_section_count = sum(section_distribution.values())
    section_distribution = {reason: count / total_section_count for reason, count in section_distribution.items()}
    sections = [section_map.get(reason) for reason in section_distribution.keys()]
    label = 'Importance score {:.2f}'.format(
                    avg_importance)
    if save: label = "All: " + label
    plt.plot(sections, 
             list(section_distribution.values()), 
             marker='o',
             label=label)
    
    plt.title('Section Distribution')
    plt.xlabel('Section')
    plt.ylabel('Frequency')
    
    save_path = os.path.join(save_dir, "section_distribution.pdf")
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
    else:
        return

def cite_importance_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    
    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        for importance in attributes['importance']:
            importance_distribution[importance] += 1

    importance_distribution = dict(sorted(importance_distribution.items(), key=lambda item: item[0]))

    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    
    plt.axvline(x=avg_importance, color='r', linestyle='--', label=f'Avg Importance: {avg_importance:.2f}')
    # And plot the importance distribution
    
    total_importance_count = sum(importance_distribution.values())
    importance_distribution = {reason: count / total_importance_count for reason, count in 
                               importance_distribution.items()}
   
    
    plt.plot(list(importance_distribution.keys()), 
             list(importance_distribution.values()), 
             marker='o')
    
    plt.title('Importance Distribution')
    plt.xlabel('Importance score')
    plt.ylabel('Frequency')
    
    save_path = os.path.join(save_dir, "importance_distribution.pdf")
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.savefig(save_path)
    



if __name__ == "__main__":
    create_article_visualize()