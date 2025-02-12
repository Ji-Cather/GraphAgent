"""Follow Measurement and Analysis of Online Social Networks
    p39
"""
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


def plot_outdegree_cc(G:nx.DiGraph,
                      save_dir:str,
                        graph_name:str):
    # 获取每个节点的出度
    if isinstance(G, nx.DiGraph):
        out_degrees = dict(G.out_degree())
    else:
        out_degrees = dict(G.degree())
   
    from LLMGraph.utils.io import writeinfo, readinfo
    if os.path.exists(os.path.join(save_dir, f"{graph_name}_outdegree_cc.json")):
        plt_data =  readinfo(os.path.join(save_dir, f"{graph_name}_outdegree_cc.json"))
        outdegree = plt_data["out_degree"]
        avg_clustering = plt_data["avg_clustering"]
    else:
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
        
        writeinfo(os.path.join(save_dir, f"{graph_name}_outdegree_cc.json"),
                info=plt_data)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(outdegree, avg_clustering, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    # plt.title('Average Clustering Coefficient vs Outdegree')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Outdegree',fontsize = 18)
    plt.ylabel('$\\bar{cc}$',fontsize = 18)
    plt.grid(True)

    # 显示图表
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_outdegree_cc.pdf")
    plt.savefig(save_path)
    plt.clf()




plot_outdegree_cc(nx.DiGraph(),"LLMGraph/tasks/tweets/configs/llama_test_1e6/evaluate/20240418",
                  "follow"
                  )