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
from evaluate.article.build_graph import build_citation_graph, readinfo

from baselines.analysis.mmd import evaluate_sampled_graphs

def generated_random_graphs(N_gen):    
    graph_path = "graph_generation/data/netcraft/citation/vllm/raw/article_meta_info.pt"
    ers = []
    bas = []

    for i in range(20):
        G = build_citation_graph(readinfo(graph_path))
        N = len(G.nodes())
        degree = len(G.edges())/N
        p = degree / (N - 1)

        er = nx.erdos_renyi_graph(N_gen, p)
        # er_c = nx.average_clustering(er)
        er_c = p
        m = int(degree // 2)
        ba = nx.barabasi_albert_graph(N_gen, m)
        ers.append(er)
        bas.append(ba)

    return {"er":ers,
            "ba":bas}

def eval_citation_graphs():
    
    graph_size_gen = 1000
    graph_size_min = graph_size_gen - 50

    graph_eval_len = 20
    
    df_save_root = "graph_gen_df"
    model_dir_map = {
        "bwr_graphrnn": "bwr_graphrnn/pred_llmcitation.pt",
        "ppgn":"graph_generation/outputs/2024-09-14/21-47-38/test/step_1000.pt",
        "bigg":"google-research-master/bigg/bigg_gen/pred_llmcitation.pt",    
        "gran":"GRAN-master/exp/GRAN/GRANMixtureBernoulli_llmcitation_2024-Sep-19-14-28-45_14408/gen_graphs.pt"
    }
   
    prefix_graph_sub = 445
    pred_model_graphs = {}
    for model, path in model_dir_map.items():
        try:
            eval_file = th.load(path)
        except Exception as e:
            continue
        pred_graphs = eval_file["pred_graphs"]
        pred_graphs = list(filter(lambda x: len(x.nodes())>graph_size_min,pred_graphs))
        pred_graphs = [pred_graph.subgraph(list(pred_graph.nodes())[20:20+graph_size_gen]) for pred_graph in pred_graphs][:graph_eval_len]
        if len(pred_graphs)==0:
            continue
        pred_model_graphs[model] = pred_graphs # 

    graph_path_2 = "LLMGraph\tasks\citeseer\configs\fast_gpt3.5_subgraph\data\article_meta_info.pt"
    G_2 = build_citation_graph(readinfo(graph_path_2))
    pred_model_graphs["generated_sub"] = []
    for idx in range(1,21):
        pred_model_graphs["generated_sub"].append(G_2.subgraph(list(G_2.nodes())[prefix_graph_sub+idx:prefix_graph_sub+graph_size_gen+idx]))

    df_path = os.path.join(df_save_root,f"generated_eval_{graph_size_gen}.csv")
    if os.path.exists(df_path):
        df = pd.read_csv(df_path,index_col =0)
    else:
        df = pd.DataFrame()
    
    graph_gt = build_citation_graph(readinfo("LLMGraph\tasks\citeseer\data\article_meta_info.pt"))
    graph_gts = []
    graph_gt = nx.convert_node_labels_to_integers(graph_gt)
    graph_gt = nx.Graph(graph_gt)
    
    for idx in range(1, 20):
        
        graph_gt_sub = graph_gt.subgraph(list(graph_gt.nodes())[-graph_size_gen-idx:-idx])
        graph_gts.append(graph_gt_sub)
    
    pred_model_graphs["GT"] = graph_gts
    pred_model_graphs.update(generated_random_graphs(graph_size_gen))

    for model_name, graph_list in pred_model_graphs.items():
        
        out = evaluate_sampled_graphs(graph_list,graph_gts)
        for k,v in out.items():
            df.loc[model_name, f"{k}"] = v
    os.makedirs(df_save_root,exist_ok=True)
    df.to_csv(df_path)


eval_citation_graphs()
