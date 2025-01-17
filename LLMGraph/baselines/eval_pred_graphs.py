import pickle
import torch as th
import powerlaw

from pathlib import Path
import pickle
import numpy as np
import networkx as nx

import os
import pandas as pd
from community import community_louvain
from citation_graph import readinfo, build_citation_graph

from graph_gen.analysis.mmd import evaluate_sampled_graphs


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

from tqdm import tqdm
def generated_random_graphs(N_gen, graph_name,N_GT):    
    graph_path_map = {
        "llmcitationcora":"/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/citation/cora/raw/article_meta_info.pt",
        "llmcitationciteseer":"/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/citation/citeseer/raw/article_meta_info.pt"
    }
    if graph_name not in graph_path_map.keys():return {}
    ers = []
    bas = []
    small_world_graphs =[]
    graph_path = graph_path_map[graph_name]
    G = build_citation_graph(readinfo(graph_path))
    for i in tqdm(range(N_GT),"generating random graphs"):
        N = len(G.nodes())
        degree = len(G.edges())/N
        p = degree / (N - 1)

        er = nx.erdos_renyi_graph(N_gen, p)
        m = int(degree // 2) if int(degree // 2)>0 else 1
        ba = nx.barabasi_albert_graph(N_gen, m)
        k = int(degree*2) if int(degree*2)>2 else 2
        small_world_graph = nx.connected_watts_strogatz_graph(N_gen, k , 0.1)
        small_world_graphs.append(small_world_graph)
        ers.append(er)
        bas.append(ba)

    return {"er":ers,
            "ba":bas,
            "small_world_graph":small_world_graphs}

def eval_citation_graphs(data_name):
    import torch
    

    graph_size_gen = 1000
    graph_size_min = graph_size_gen

    graph_eval_len = 20
    
    df_save_root = os.path.join("graph_gen_df",data_name)
    
    model_dir_map = {
        "bwr_graphrnn": "baseline_checkpoints/bwr_graphrnn",
        "ppgn":"baseline_checkpoints/l_ppgn",
        "bigg":"baseline_checkpoints/bigg_gen",    
        "gran":"baseline_checkpoints/gran_gen",
        "graphmaker_sync":"baseline_checkpoints/graphmaker_sync",
      
    }
    
    pred_model_graphs = {}
    
    path = "baseline_checkpoints/gag_generated.pt"
    generated = torch.load(path)
    pred_model_graphs["gag"] = generated[data_name][:graph_eval_len]
    dataset_path = os.path.join("baseline_checkpoints",
                    f"{data_name}.pkl"
    )
    with open(dataset_path,"rb") as f:
        generated_graphs = pickle.load(f)
        pred_model_graphs["GT"] = generated_graphs["test"][:graph_eval_len]

    
    for model, dir in model_dir_map.items():
        path = os.path.join(dir,data_name, "pred_graphs.pt")
        try:
            eval_file = th.load(path)
        except Exception as e:
            continue
        pred_graphs = eval_file["pred_graphs"]
        pred_graphs = list(filter(lambda x: len(x.nodes())>graph_size_min,pred_graphs))
        # pred_graphs = list(filter(lambda x: len(x.edges())>0,pred_graphs))
        pred_graphs = [
            nx.Graph(pred_graph.subgraph(list(pred_graph.nodes())[:graph_size_gen])) for pred_graph in pred_graphs][:graph_eval_len]
        if len(pred_graphs)==0:
            continue
        pred_model_graphs[model] = pred_graphs # 
    
    df_path = os.path.join(df_save_root,f"generated_eval_{graph_size_gen}.csv")
    if os.path.exists(df_path):
        df = pd.read_csv(df_path,index_col =0)
    else:
        df = pd.DataFrame()


    """calculate random graphs"""
    
    pred_model_graphs.update(generated_random_graphs(graph_size_gen,data_name,graph_eval_len))
    # pred_model_graphs = {"gran":pred_model_graphs["gran"],
    #                      "GT":pred_model_graphs["GT"]}
    pred_graphs_processed = {}
    for model_name, graph_list in pred_model_graphs.items():
        pred_graphs_processed[model_name]= [nx.convert_node_labels_to_integers(nx.Graph(graph)) for graph in graph_list]

    for model_name, graph_list in tqdm(pred_model_graphs.items(),"evaluating graphs"):
        out = evaluate_sampled_graphs(graph_list,pred_model_graphs["GT"])
        for k,v in out.items():
            df.loc[model_name, f"{k}"] = v
    
    os.makedirs(df_save_root,exist_ok=True)
    df.to_csv(df_path)
    df = add_gem_col(df)
    df.to_csv(os.path.join(df_save_root,f"generated_eval_{graph_size_gen}_mean_new.csv"))

    
import copy
from sklearn.preprocessing import MinMaxScaler
def add_gem_col(df):
    negative_cols = ["degree_mmd",
                     "cluster_mmd",
                     "spectra_mmd",
                     "orbit_mmd"
                     ]
    use_cols = [*negative_cols,"valid"]

    # df = df[use_cols]
    # 将所有的 int 转换为 float 并保留到小数点后两位
    df = df.astype(float).round(2)
    
    # 自定义索引
    custom_index = [
        "er",
        "ba",
        "small_world_graph",
        "bigg",
        "gran",
        "bwr_graphrnn",
        "graphmaker_sync",
        "ppgn",
        "gag"
        ]       
    df_gt = df.loc["GT"]
    custom_index = list(filter(lambda x:x in df.index, custom_index))
    df = df.loc[custom_index]

    df_norm = copy.deepcopy(df)

    for idx in df.index:
        for col in negative_cols:
            df_norm.loc[idx,col] = 1- sigmoid(df_norm.loc[idx,col])
        df_norm.loc[idx,"valid"] = df_norm.loc[idx,"valid"]

    for idx in df.index:
        values = []
        for col in use_cols:
            values.append(df_norm.loc[idx,col])
        df.loc[idx,"gem"] = np.mean(values)

    df.loc["GT"] = df_gt
    df.loc["GT","gem"] = np.nan
    return df


# tobedone
# def eval_tweet_graphs(model, checkpoint_dir, key):
#     # model = "bwr_graphrnn"
#     # checkpoint_dir = "bwr_graphrnn"

#     # model = "ppgn_oneshot"
#     # checkpoint_dir ="outputs/2024-09-15/13-44-00/test"

#     df_save_root = "graph_gen_df"
#     model_dir_map = {
#         "bwr_graphrnn": "bwr_graphrnn/pred_llmcitation.pt",
#         "ppgn":"/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/outputs/2024-09-14/21-47-38/test/step_1000_700.pt",
        
#     }
    

#     pred_model_graphs = {}
#     for model, path in model_dir_map.items():
#         try:
#             eval_file = th.load(path)
#         except Exception as e:
#             continue
#         pred_model_graphs[model] = eval_file["pred_graphs"][0] # 
    
#     train_p = 0.4
#     val_p = 0.3
#     graph_path = "/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/citation/vllm/raw/article_meta_info.pt"

#     G = build_citation_graph(readinfo(graph_path))
#     nodes = list(G.nodes())
#     n = len(nodes)

#     df_path = os.path.join(df_save_root,f"generated_eval_{model}.csv")
#     if os.path.exists(df_path):
#         df = pd.read_csv(df_path,index_col =0)
#     else:
#         df = pd.DataFrame()
#     xmin = 0
#     graph_gt = build_citation_graph(readinfo("/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/citation/seed/raw/article_meta_info.pt"))
#     # graph_gt = graph_gt.subgraph(nodes[2000:2700])
#     graph_gt = nx.Graph(graph_gt)
#     df.loc["GT","k"] = len(graph_gt.nodes())/len(graph_gt.edges())
#     df.loc["GT","cc"] = nx.average_clustering(graph_gt)
#     partition = community_louvain.best_partition(graph_gt)
#     # 计算这种划分的modularity
#     modularity = community_louvain.modularity(partition, graph_gt)
#     df.loc["GT","modularity"] = modularity
#     graph_gt = nx.convert_node_labels_to_integers(graph_gt)
    
#     G2 = G.subgraph(nodes[int(n*(train_p+val_p)):int(n*(train_p+val_p))+graph_size])
#     pred_model_graphs["generated"] = G2
#     for model_name, graph in pred_model_graphs.items():
#             graph = nx.Graph(graph)
#             graph = nx.convert_node_labels_to_integers(graph)
#             degree_list = [graph.degree(n) for n in graph.nodes()]
#             results = powerlaw.Fit(list(degree_list), 
#                                     discrete=True,
#                                         # fit_method="KS",
#                                         xmin=xmin
#                                         )
#             alpha = results.power_law.alpha
#             xmin = results.power_law.xmin
#             sigma = results.power_law.sigma
#             D = results.power_law.D
#             partition = community_louvain.best_partition(graph)
#             # 计算这种划分的modularity
#             modularity = community_louvain.modularity(partition, graph)

#             df.loc[model_name,f"D"] = D
#             df.loc[model_name,f"alpha"] = alpha
#             df.loc[model_name,f"k"] = len(graph.edges())/len(graph.nodes())
#             df.loc[model_name,f"cc"] = nx.average_clustering(graph)
#             df.loc[model_name,f"modularity"] = modularity
#             out = evaluate_sampled_graphs([graph],[graph_gt])
#             for k,v in out.items():
#                 df.loc[model_name, f"{k}"] = v
#     os.makedirs(df_save_root,exist_ok=True)
#     df.to_csv(df_path)





data_names = [
    # "llmcitationcora",
    "llmcitationciteseer",
    ]

# appendix experiment on cora are to be released
for data_name in data_names:
    eval_citation_graphs(data_name)
    df_path = os.path.join("graph_gen_df",data_name,"generated_eval_1000.csv")
    df = pd.read_csv(df_path,index_col=0)
    df = add_gem_col(df)
    df.to_csv(os.path.join("graph_gen_df",data_name,"generated_eval_1000_mean_new.csv") )
    
