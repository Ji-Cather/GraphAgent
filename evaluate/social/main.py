import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
from networkx.algorithms.similarity import graph_edit_distance
import argparse
import matplotlib.pyplot as plt

import os
import torch


import pandas as pd
import json
from PIL import Image
from LLMGraph.utils.io import readinfo, writeinfo
from evaluate.visualize.social import (plot_degree_figures, 
                                       plot_indegree_outdegree,
                                       plot_shrinking_diameter,
                                       plot_relative_size,
                                       plot_mean_degree,
                                       plot_densification_power_law,
                                       plot_friend_degree,
                                       plot_outdegree_cc,
                                       plot_avg_path,
                                       plot_gcc_proportion)
import copy
from evaluate.social.gcc_diameter import calculate_lcc_proportion

from evaluate.matrix import calculate_directed_graph_matrix
from datetime import datetime, date
from tqdm import tqdm
parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器
parser.add_argument('--config', 
                    type=str, 
                    default="test_config", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="tweet", 
                    help='The task setting for the LLMGraph')  # 添加参数


def print_graph_info(DG:nx.Graph):
    print('Number of nodes', len(DG.nodes))
    print('Number of edges', len(DG.edges))
    print('Average degree', sum(dict(DG.degree).values()) / len(DG.nodes))



        
        
def save_graph_fig(G_true, G_generated, root_dir):
    
    # 绘制图
    nx.draw(G_true, with_labels=True, node_color='lightblue', edge_color='gray')
    # 保存图形到文件
    plt.savefig(f"{root_dir}/graph_true.pdf")  # 可以更换为.jpg或其他支持的格式
    plt.clf()
    
    nx.draw(G_generated, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.savefig(f"{root_dir}/graph_generated.pdf")  # 可以更换为.jpg或其他支持的格式
    plt.clf()
    

def build_nx_graph(social_member_data_path, 
                   action_logs,
                   pos, 
                   date,
                   date_map,
                   transitive_nodes:dict = {}):
    social_member_data = pd.read_csv(social_member_data_path)
    DG = nx.DiGraph()
    G = nx.Graph()

    delete_nodes = transitive_nodes["delete_ids"]
    for index,row in social_member_data.iterrows():
        user_index = row['user_index']
        if user_index not in date_map.keys():
            date_map[user_index] = date
        if user_index in delete_nodes:
            continue
        DG.add_node(user_index,date = date_map[user_index])
        G.add_node(user_index,date = date_map[user_index])

    for _, row in social_member_data.iterrows():
        user_index = row['user_index']
        if user_index not in pos:
            pos[user_index] = None
        follow_ids = json.loads(row['follow'])
        friend_ids = json.loads(row['friend'])
        for follow_id in follow_ids:
            if user_index in delete_nodes or \
                follow_id in delete_nodes:
                continue
            if user_index != follow_id:
                DG.add_edge(user_index, follow_id)
        for friend_id in friend_ids:
            if user_index in delete_nodes or \
                friend_id in delete_nodes:
                continue
            if user_index != friend_id:
                DG.add_edge(user_index, friend_id)
                DG.add_edge(friend_id, user_index)
                G.add_edge(user_index, friend_id)

    date = datetime.strptime(date, "%Y%m%d").date()
    nodes = DG.nodes
    action_graph = build_nx_action_graph(action_logs,
                                         date,
                                         nodes)
    return DG, G, action_graph, pos, date_map

def build_nx_action_graph(action_logs:list,
                           timestamp:date,
                           nodes):
    DG = nx.DiGraph()
    DG.add_nodes_from(nodes)

    action_logs_filtered = list(filter(
        lambda x: datetime.strptime(x[3], "%Y-%m-%d").date()<=timestamp, 
        action_logs)
    )

    for action_one in action_logs_filtered:
        act_id = action_one[0]
        own_id = action_one[1]
        DG.add_edge(act_id, own_id, action_type = action_one[2])
            
    return DG

def visualize_graph(G, save_path, pos, date_str, diameter):
    plt.figure(figsize=(16, 16))
    pos = nx.kamada_kawai_layout(G, scale=2)
    nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=150, edge_color='#FF5733', font_size=9, font_color='black')
    plt.title(f"Date: {date_str} - Diameter: {diameter}", fontsize=16, loc='center')
    plt.savefig(save_path)
    plt.close()


def safe_diameter(G):
    if G.is_directed():
        """这里对于shrinking diamter, 如果不是强连通的图, 应该diameter为节点数? 
            待修改
        """
        if nx.is_strongly_connected(G):
            return nx.diameter(G)
        else:
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            return nx.diameter(G.subgraph(largest_scc))
    else:
        if nx.is_connected(G):
            return nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            return nx.diameter(G.subgraph(largest_cc))

def pseudo_diameter(G):
    diameter = 0
    for node in G.nodes():
        # 从该节点出发能够到达的最远距离
        lengths_forward = nx.single_source_shortest_path_length(G, node)
        max_length_forward = max(lengths_forward.values(), default=0)

        # 能够到达该节点的最远距离
        G_reversed = G.reverse()
        lengths_backward = nx.single_source_shortest_path_length(G_reversed, node)
        max_length_backward = max(lengths_backward.values(), default=0)

        # 更新伪直径
        diameter = max(diameter, max_length_forward, max_length_backward)
    return diameter



def get_sn_graph_matrix(graph_lists,
                        sn_root):
    matrix = []
    matrix_root = os.path.join(sn_root, "matrix")
    os.makedirs(matrix_root, exist_ok=True)
    for DG, G, date_str, positions in graph_lists:
        follower_diameter = safe_diameter(DG)
        # friend_diameter = safe_diameter(G)
        ps_diameter = pseudo_diameter(DG)
        matrix.append({"date": date_str, 
                       "follower_diameter": follower_diameter,
                       "pseudo_diameter": ps_diameter})

    diameter_matrix = pd.DataFrame(matrix)
    diameter_matrix.to_csv(os.path.join(matrix_root, "diameter_matrix.csv"), index=False)


def visualize_sn_graphs(graph_generator,
                        sn_root):
    images = []
    images_friend = []
    follower_save_root = os.path.join(sn_root, "follower_graph")
    friend_save_root = os.path.join(sn_root, "friend_graph")
    os.makedirs(follower_save_root, exist_ok=True)
    os.makedirs(friend_save_root, exist_ok=True)

    for DG, G, date_str, positions in graph_generator:
        follower_diameter = safe_diameter(DG)
        friend_diameter = safe_diameter(G)
        
        follower_img_path = os.path.join(follower_save_root, f"{date_str}_DG.pdf")
        friend_img_path = os.path.join(friend_save_root, f"{date_str}_G.pdf")
        visualize_graph(DG, follower_img_path, positions, date_str, follower_diameter)
        visualize_graph(G, friend_img_path, positions, date_str, friend_diameter)
        images.append(follower_img_path)
        images_friend.append(friend_img_path)

    frames = [Image.open(image) for image in images]
    gif_path = os.path.join(sn_root, "follower_network_evolution.gif")
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)
    frames = [Image.open(image) for image in images_friend]
    gif_path = os.path.join(sn_root, "friend_network_evolution.gif")
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)

    
def build_sn_graphs(data_root):
    sn_root = os.path.join(data_root,"social_network")
    transitive_nodes_path = os.path.join(data_root, "transitive_agent_log.json")
    transitive_nodes_log = readinfo(transitive_nodes_path)
    action_logs_path = os.path.join(data_root, "action_logs.json")
    csv_files = [f for f in os.listdir(sn_root) if f.endswith('.csv')]
    
    positions = {}
    date_map = {}
    action_logs = readinfo(action_logs_path)
    transitive_nodes_all = {"delete_ids":[],
                            "add_ids":[]}
    for csv_file, transitive_nodes in zip(sorted(csv_files),
                                          transitive_nodes_log):

        date_str = csv_file.split("_")[3][:8]
        file_path = os.path.join(sn_root, csv_file)
        for k in transitive_nodes.keys():
            transitive_nodes_all[k].extend(transitive_nodes[k])
        DG, G, action_graph, positions, date_map = build_nx_graph(file_path, 
                                                    action_logs,
                                                    positions,
                                                    date_str, 
                                                    date_map,
                                                    transitive_nodes_all
                                                    )
        
        yield DG, G, action_graph, date_str, positions


def transfer_to_user_info(action_logs:list,
                        forum_map:dict,    
                        nodes,
                        sn_file:pd.DataFrame,
                        tweet_idx:int = 4):
    

    DIRECTED = True
    UNDIRECTED = False
    edges = []
    nodes_info = {}
    tweet_nodes = []
    user_nodes = []
    user_template = """
Name: {user_name},
Description: {user_description},
"""
    tweet_template = """
Content: {page_content},
Topic: {topic}
"""
    
    for action_one in action_logs:
        act_id = action_one[0]
        own_id = action_one[1]
        act_type = action_one[2]
        if act_type in ["tweet","reply"]:
            tweet_nodes.append(f"tweet_{tweet_idx}")     
            nodes_info[f"tweet_{tweet_idx}"]  = {
                "info":{"content":forum_map[tweet_idx]["page_content"],
                        "topic":forum_map[tweet_idx]["topic"]},
                "type":"tweet"
            }
            edges.append((f"user_{act_id}",f"tweet_{tweet_idx}",act_type,
                          DIRECTED))
            tweet_idx += 1
        else:
            edges.append((f"user_{act_id}",f"user_{own_id}",act_type,
                          DIRECTED))
            
        if f"user_{act_id}" not in nodes:
            user_nodes.append(f"user_{act_id}")
    
    for user_id, user_info in sn_file.iterrows():
        nodes_info[f"user_{user_id}"]  = {
            "info":{
            "user_name":user_info["user_name"],
            "user_description":user_info["user_description"],
            },
        "type":"user"
        }
    added_nodes = [*user_nodes,*tweet_nodes]



    return list(set(added_nodes)), edges, nodes_info, tweet_idx


def save_front_end_data(data_root):
    sn_root = os.path.join(data_root,"social_network")
 
    action_logs_path = os.path.join(data_root, "action_logs.json")
    csv_files = [f for f in os.listdir(sn_root) if f.endswith('.csv')]
    tweet_idx = 0
    
    time_author_logs = [] # {nodes:[], edges:[], nodes_info:}
    action_logs = readinfo(action_logs_path)
    forum_info = readinfo(os.path.join(data_root, "forum.json"))
    forum_map = {}
    for tweet in forum_info:
        forum_map[tweet["tweet_idx"]] = tweet

    last_timestamp = "20240412"
    nodes = []
    save_dir = os.path.join(data_root,"front_end")
    os.makedirs(save_dir, exist_ok=True)

    for csv_file in tqdm(sorted(csv_files)[:15]):
        
        timestamp = csv_file.split("_")[3][:8]
        file_path = os.path.join(sn_root, csv_file)
        sn_file = pd.read_csv(file_path,index_col=0)
        action_logs_filtered = list(filter(
        lambda x: datetime.strptime(x[3], "%Y-%m-%d").date()<datetime.strptime(timestamp,"%Y%m%d").date() and \
            datetime.strptime(x[3], "%Y-%m-%d").date()>=datetime.strptime(last_timestamp,"%Y%m%d").date(), 
        action_logs))

        chunk_size = 50
        chunks = [action_logs_filtered[i:i + chunk_size] for i in range(0, len(action_logs_filtered), chunk_size)]

        # 查看每个 chunk 的大小
        for i, chunk in enumerate(chunks):
            # print(f"Chunk {i + 1}: {len(chunk)} items")
            added_nodes, edges, nodes_info, tweet_idx = transfer_to_user_info(chunk,
                                                            forum_map,
                                                            nodes,
                                                            sn_file,
                                                            tweet_idx)
            nodes.extend(added_nodes)

            time_author_logs.append({"nodes":list(added_nodes), 
                                    "edges":list(edges), 
                                    "nodes_info":nodes_info
                                    })
            print("nodes", len(list(added_nodes)), "edges", len(list(edges)))

        last_timestamp = timestamp

    writeinfo(os.path.join(save_dir,"nodes_edges_15.json"),
              time_author_logs)

    

def plt_sn_attrs(sn_root):
    ## 这里添加plt shrinking diameter ；pk、k；prefrentianl attachment的图

    """shrinking diameter"""
    matrix_path = os.path.join(sn_root, "matrix","diameter_matrix.csv")
    matrix = pd.read_csv( matrix_path)
    plt_save_dir = os.path.join(sn_root, "plt")
    plot_shrinking_diameter(matrix, plt_save_dir, "follower")
    

def calculate_social_matrix(graph_lists:list,
                            save_dir:str):
    
    
    graph_types = [
        "follow",
        "friend",
        "action"
    ]
    graph_list_map = {
        graph_type:[]
        for graph_type in graph_types
    }
    # if os.path.exists(os.path.join(save_dir, "follow_matrix.csv")):
    #     follow_index = pd.read_csv(os.path.join(save_dir, "follow_matrix.csv"),
    #                                index_col=0).index.to_list()
    # else:
    #     follow_index = []

    for DG, G, action_graph, date_str, positions in tqdm(graph_lists):
        graph_name = f"{date_str}_social"
        graph_type_map ={
            "follow":DG,
            "friend":G,
            "action":action_graph
        }
        # if graph_name in follow_index:
        #     continue
        for graph_type in graph_types:
            # matrix = calculate_directed_graph_matrix(
            #                                 graph_type_map[graph_type],
            #                                 type="social",
            #                                 graph_name=graph_name,
            #                                 calculate_matrix=[
            #                                     # "mmd",
            #                                     #    "community",
            #                                     #    "control",
            #                                     "base_info"
            #                                    ])
            # matrix = calculate_lcc_proportion(graph_type_map[graph_type], 
            #                                   graph_name=graph_name)

            matrix = readinfo(os.path.join(save_dir,date_str,f"{graph_type}_gcc_proprtion.json"))
            df = pd.DataFrame()
            for k,v in matrix.items():
                df.loc[graph_name, k] = v   
            
            graph_list_map[graph_type].append(df)

    os.makedirs(save_dir, exist_ok=True)
    for graph_type in graph_types:
        df = pd.concat(graph_list_map[graph_type])
        if os.path.exists(os.path.join(save_dir, f"{graph_type}_matrix.csv")):
            origin_df = pd.read_csv(os.path.join(save_dir, f"{graph_type}_matrix.csv"), 
                             index_col=0)
            df = pd.concat([origin_df, df], axis=1)
        save_path = os.path.join(save_dir, f"{graph_type}_matrix.csv")
        df.to_csv(save_path)    


def visualize_social(
                    df_root:str,
                    save_root:str):
    """single graph"""
    # DG, G, action_graph, date_str, positions = graph_lists[-1]=
    

    os.makedirs(save_root, exist_ok=True)
    """graph evolution along time"""
    # 可视化shrinking diameter
    # plot_shrinking_diameter(df_root, save_root,"tweet")
    plot_relative_size(df_root,save_root, "tweet")
    # plot_mean_degree(df_root,save_root, "tweet")
    # plot_densification_power_law(df_root,save_root, "tweet")
    

def calculate_single_graph(data_root,
                           csv_file_index,
                           save_root):
    sn_root = os.path.join(data_root,"social_network")
    transitive_nodes_path = os.path.join(data_root, "transitive_agent_log.json")
    transitive_nodes_log = readinfo(transitive_nodes_path)
    action_logs_path = os.path.join(data_root, "action_logs.json")
    csv_files = [f for f in os.listdir(sn_root) if f.endswith('.csv')]
    
    positions = {}
    date_map = {}
    action_logs = readinfo(action_logs_path)
    csv_file = sorted(csv_files)[csv_file_index]
    transitive_nodes = transitive_nodes_log[csv_file_index]
    date_str = csv_file.split("_")[3][:8]
    file_path = os.path.join(sn_root, csv_file)
    DG, G, action_graph, positions, date_map = build_nx_graph(file_path, 
                                                action_logs,
                                                positions,
                                                date_str, 
                                                date_map,
                                                transitive_nodes)
    import pickle
    # debug, save seed graph
    graphs = {
        "follow":DG,
        "friend":G,
        "action":action_graph
    }
    save_path = os.path.join(save_root,"llmtweet_generated.pkl")
    os.makedirs(save_root,exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)

    graph_lists = [(DG, G, action_graph, date_str, positions)]
    save_root = os.path.join(save_root, date_str)
    os.makedirs(save_root, exist_ok=True)
    
    # calculate_social_matrix(graph_lists, save_root)


def save_degree_list(G:nx.DiGraph,
                    save_dir:str,
                    graph_name:str):
    save_degree_root = os.path.join(save_dir,"degree")
    os.makedirs(save_degree_root,exist_ok=True)
    degree_list = [G.in_degree(n) for n in G.nodes()]
    writeinfo(os.path.join(save_degree_root,f"{graph_name}.json"),degree_list)

def plot_single_graph(data_root,
                    csv_file_index,
                    save_root):
    sn_root = os.path.join(data_root,"social_network")
    transitive_nodes_path = os.path.join(data_root, "transitive_agent_log.json")
    csv_files = [f for f in os.listdir(sn_root) if f.endswith('.csv')]
    if os.path.exists(transitive_nodes_path):
        transitive_nodes_log = readinfo(transitive_nodes_path)
    else:
        transitive_nodes_log = [{"delete_ids":[]} for i in range(len(csv_files))]
    action_logs_path = os.path.join(data_root, "action_logs.json")

    
    positions = {}
    date_map = {}
    action_logs = readinfo(action_logs_path)
    csv_file = sorted(csv_files)[csv_file_index]
    transitive_nodes = transitive_nodes_log[csv_file_index]
    date_str = csv_file.split("_")[3][:8]
    file_path = os.path.join(sn_root, csv_file)
    DG, G, action_graph, positions, date_map = build_nx_graph(file_path, 
                                                action_logs,
                                                positions,
                                                date_str, 
                                                date_map,
                                                transitive_nodes)
    save_root = os.path.join(save_root, date_str)
    os.makedirs(save_root, exist_ok=True)
    graph_name_map = {
        "follow":DG,
        "friend":G,
        "action":action_graph
    }
    for graph in graph_name_map.values():
        print_graph_info(graph)
    for graph_name, graph  in graph_name_map.items():
        # 可视化prefrential attachment/ 度分布
        # plot_degree_figures(graph,save_dir=save_root,graph_name=graph_name)
        plot_friend_degree(graph, save_root, graph_name)
        # plot_indegree_outdegree(graph, save_root, graph_name)
        # plot_avg_path(graph, save_root, graph_name)
        # plot_outdegree_cc(graph, save_root, graph_name)
        # plot_gcc_proportion(graph, save_root, graph_name)

def print_graph_info(DG:nx.Graph): 
    print('Number of nodes', len(DG.nodes))
    print('Number of edges', len(DG.edges))
    print('Average degree', sum(dict(DG.degree).values()) / len(DG.nodes))





if __name__ == "__main__":
    args = parser.parse_args()  # 解析参数
    save_root = "LLMGraph/tasks/{task}/configs/{config}/evaluate".format(
        task = args.task,
        config = args.config
    )
    vis_save_root = "LLMGraph/tasks/{task}/configs/{config}/visualize".format(
        task = args.task,
        config = args.config
    )
  
    data_root = "LLMGraph/tasks/{task}/configs/{config}/data/generated/data".format(
        task = args.task,
        config = args.config
    ) # 这里的root之前不对我就暴力改了，按道理重定义系统路径比较文明 by: lrl
    graph_generator = build_sn_graphs(data_root)
    graph_lists = []
    from pathlib import Path
    import pickle
    for DG, G, action_graph, date_str, positions in graph_generator:
        
        graph_lists.append((DG, G, action_graph, date_str, positions))

    