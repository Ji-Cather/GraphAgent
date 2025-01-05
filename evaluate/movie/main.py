import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
from networkx.algorithms.similarity import graph_edit_distance
import argparse
import matplotlib.pyplot as plt



import os
import torch

import networkx as nx
from evaluate.matrix import calculate_directed_graph_matrix
from evaluate.movie.gcc_size import calculate_gcc_size
from evaluate.visualize.movie import (create_movie_visualize,
                             plot_degree_figures,
                             plot_nc,
                             plot_err)
from LLMGraph.utils.io import readinfo, writeinfo
import json


def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器
parser.add_argument('--config', 
                    type=str, 
                    default="test_config", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="movielens", 
                    help='The task setting for the LLMGraph')  # 添加参数






def print_graph_info(DG:nx.Graph):
    print('Number of nodes', len(DG.nodes))
    print('Number of edges', len(DG.edges))
    print('Average degree', sum(dict(DG.degree).values()) / len(DG.nodes))


def load_graph(model_dir, 
               val_dataset,
               nodes_len:int = None):
    generated_model_path = os.path.join(model_dir,"graph.graphml")
    # 示例：比较两个随机图
    assert os.path.exists(generated_model_path),"The generated graph path doesn't exist"
    # G_generated = nx.read_adjlist(generated_model_path,create_using=nx.DiGraph())
    G_generated = nx.read_graphml(generated_model_path)
    # if nodes_len is not None:
    #     sub_nodes = [str(i) for i in range(nodes_len)]
    #     G_generated = G_generated.subgraph(sub_nodes).copy()
    if nodes_len is not None:
        n = len(G_generated.nodes())
        nodes = [str(i) for i in range(n-nodes_len,n)]
        G_generated = G_generated.subgraph(nodes).copy()
    watcher_nodes = list(filter(lambda x: "watcher" in x, G_generated.nodes()))
    movie_nodes = list(filter(lambda x: "movie" in x, G_generated.nodes()))
    print("Generated Graph:", 
          G_generated,
          f"watcher {len(watcher_nodes)}",
          f"movie {len(movie_nodes)}",
          'Average degree', 
          "{degree:.3f}".format(degree =
            sum(dict(G_generated.degree).values()) / len(G_generated.nodes)))
    return G_generated


        
 
    


    
      
def calculate_movie_matrix(G_generated:nx.DiGraph,
                            save_dir:str,
                            type:str = "movielens"):
    graphs_map ={
        "movielens":G_generated,
        "user_projection":get_projected_graph(G_generated)
    }

    for graph_name, graph in graphs_map.items():
        matrix = calculate_directed_graph_matrix(
                                                graph,
                                                graph_name,
                                                type=type,
                                                calculate_matrix=[
                                                    #    "clustering",
                                                       "community",
                                                    #    "control",
                                                    "base_info"
                                                ])
        calculate_gcc_size(graph,graph_name,save_dir)
        save_path = os.path.join(save_dir,f"{graph_name}_matrix.csv")
        os.makedirs(save_dir, exist_ok=True)
        matrix.to_csv(save_path)

from networkx.algorithms import bipartite
def get_projected_graph(G_generated:nx.DiGraph, bipartite_label =0):
    user_nodes = set(n for n,d in G_generated.nodes(data=True) if d['bipartite']==bipartite_label)
    # 计算用户节点的投影图
    user_projection = bipartite.projected_graph(G_generated, user_nodes)
    return user_projection


if __name__ == "__main__":
    args = parser.parse_args()  # 解析参数
    save_root = "LLMGraph/tasks/{task}/configs/{config}/evaluate".format(
        task = args.task,
        config = args.config
    )
    model_root = "LLMGraph/tasks/{task}/configs/{config}/data/model".format(
        task = args.task,
        config = args.config
    )
    
   
    G_generated = load_graph(model_root,"movielens")
