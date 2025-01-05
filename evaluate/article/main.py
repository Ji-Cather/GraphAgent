import networkx as nx
import numpy as np

from networkx.algorithms.similarity import graph_edit_distance
import argparse
import matplotlib.pyplot as plt


import os
import torch
import pandas as pd
import random
from evaluate.article.libs.mrqap import MRQAP
from evaluate.article.libs.qap import QAP
from evaluate.article.libs.ols import OLS
from evaluate.article.gcc_size import calculate_gcc_size
import scipy.stats as stats
from typing import Dict
import time
from datetime import datetime,date
from evaluate.visualize.article import (
                                        plot_self_citation,
                                        plot_betas,
                                        plot_degree_figures,
                                        plot_nc,
                                        plot_err,
                                        plot_degree_compare,
                                        plot_igraph_compare,
                                        plot_gini,
                                        plot_reason_visualize,
                                        plt_topic_given
                                        )
from evaluate.article.calculate_reason import calculate_reason, calculate_reason_cited
from evaluate.article.build_graph import (build_author_citation_graph,
                                          update_citation_graph,
                                          build_country_citation_graph,
                                          build_relevance_array,
                                          build_group_relevance_array,
                                        #   build_group_number_array,
                                          assign_topic,
                                          build_citation_group_array,
                                          build_citation_group_array_from_citation,
                                            build_citation_graph,
                                            build_bibliographic_coupling_network,
                                            build_co_citation_graph,
                                            build_co_authorship_network)
from evaluate.matrix import calculate_directed_graph_matrix
from LLMGraph.utils.io import readinfo, writeinfo
from tqdm import tqdm



parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器
parser.add_argument('--config', 
                    type=str, 
                    default="test_config", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--configs', 
                    type=str, 
                    default="test_config,test_config_2", 
                    help='a list of configs for the config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="cora", 
                    help='The task setting for the LLMGraph')  # 添加参数






parser.add_argument('--xmin', 
                    type=int, 
                    default=3, 
                    help='power law fit xmin')  # 添加参数




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
    if nodes_len is not None:
        sub_nodes = [str(i) for i in range(nodes_len)]
        G_generated = G_generated.subgraph(sub_nodes).copy()
    print("Generated Graph:", 
          G_generated,
          'Average degree', 
          "{degree:.3f}".format(degree =
            sum(dict(G_generated.degree).values()) / len(G_generated.nodes)))
    G_true = load_test_dataset(val_dataset)
    print(f"True Graph {val_dataset}:", 
          G_true,
          'Average degree', 
          "{degree:.3f}".format(degree =
            sum(dict(G_true.degree).values()) / len(G_true.nodes)))
    return G_true, G_generated

def build_graphs(article_meta_data:dict,
                 author_info:dict,
                 article_num = None,
                 graph_types:list = [ 
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ]):
    
    graphs = {}
    if article_num is not None:
        article_meta_data = dict(list(article_meta_data.items())[:article_num])

    article_graph = build_citation_graph(article_meta_data)
    graphs["article_citation"] = article_graph
   
    # 节点是论文，边是论文间的引用
    
    if "bibliographic_coupling" in graph_types:
        bibliographic_coupling_network = build_bibliographic_coupling_network(
            article_graph
        )
        graphs["bibliographic_coupling"] = bibliographic_coupling_network
    
    if "co_citation" in graph_types:
        co_citation_graph = build_co_citation_graph(article_graph)
        graphs["co_citation"] = co_citation_graph
    
    if "author_citation" in graph_types:
        # 节点是作者， 边是他们的引用
        author_citation_graph = build_author_citation_graph(article_meta_data,
                                                        author_info)
        graphs["author_citation"] = author_citation_graph

    if "country_citation" in graph_types:
        article_graph = update_citation_graph(article_graph,article_meta_data,author_info)
        # article_graph = article_graph.subgraph(list(article_graph.nodes())[:500])
        country_citation_graph = build_country_citation_graph(article_meta_data,
                                                             author_info,
                                                             article_graph)
        # 节点是国家， 边是他们的引用
        graphs["country_citation"] = country_citation_graph

    if "co_authorship" in graph_types:
        co_authorship_network =  build_co_authorship_network(article_meta_data,
                                                             author_info)
        graphs["co_authorship"] = co_authorship_network
    
    for graph_type, graph in graphs.items():
        if isinstance(graph,nx.DiGraph):
            print(f"{graph_type:>20} Graph:", 
              graph,
              'Average degree', 
              "{degree:.3f}".format(degree =
                sum(dict(graph.degree).values()) / len(graph.nodes)),
                'indegree',
                 "{degree:.3f}".format(degree =
                sum(dict(graph.in_degree).values()) / len(graph.nodes)),
                'outdegree',
                 "{degree:.3f}".format(degree =
                sum(dict(graph.out_degree).values()) / len(graph.nodes)))
        else:
            try:
                print(f"{graph_type:>20} Graph:", 
                graph,
                'Average degree', 
                "{degree:.3f}".format(degree =
                    sum(dict(graph.degree).values()) / len(graph.nodes)))
            except:pass
    

    return graphs


def load_test_dataset(val_dataset):
    
    if val_dataset == "cora":
        DG = nx.DiGraph()
        pass
        return DG
    
    if val_dataset == "citeseer":
        DG = nx.DiGraph()
        path = "LLMGraph/tasks/citeseer/data/article_meta_info.pt"
        articles = readinfo(path)
        article_idx_title_map = {}
        for idx,title in enumerate(articles.keys()):
            article_idx_title_map[title] = idx
            DG.add_node(idx,title=title,time=articles[title]["time"])
            
        for title, article_info in articles.items():
            edges =[]
            cited_articles = article_info.get("cited_articles",[])
            title_idx = article_idx_title_map.get(title)
            for cite_title in cited_articles:
                cited_idx = article_idx_title_map.get(cite_title)
                if cited_idx is not None:
                    edges.append((cited_idx,title_idx))  
            DG.add_edges_from(edges)        
        return DG
    
    if val_dataset == "llm_agent":
        DG = nx.DiGraph()
        path = "LLMGraph/tasks/llm_agent/data/article_meta_info.pt"
        articles = readinfo(path)
        article_idx_title_map = {}
        for idx,title in enumerate(articles.keys()):
            article_idx_title_map[title] = idx
            DG.add_node(idx,title=title,time=articles[title]["time"])
            
        for title, article_info in articles.items():
            edges =[]
            cited_articles = article_info.get("cited_articles",[])
            title_idx = article_idx_title_map.get(title)
            for cite_title in cited_articles:
                cited_idx = article_idx_title_map.get(cite_title)
                if cited_idx is not None:
                    edges.append((cited_idx,title_idx))  
            DG.add_edges_from(edges)        
        DG = DG.subgraph(nodes=list(DG.nodes())[:100])
        return DG
    

    

def visualize_article(
                    generated_article_dir:str,
                    article_meta_data:dict,
                    author_info:dict,
                    save_dir:str,
                    task:str = "llm_agent"):
    graphs = build_graphs(article_meta_data,author_info,
                          graph_types=[#"author_citation",
                                    #    "article_citation",
                            # "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                                       ])
    article_graph = graphs["article_citation"]
    # author_graph = graphs["author_citation"]
    country_graph = graphs["country_citation"]


    # topic distribution
    # plt_topic_given(task_name=task,article_meta_data=article_meta_data,save_dir=save_dir)

    # # 可视化igraph
    # # create_article_visualize(article_graph,"article",save_dir=save_dir)
    # create_article_visualize(author_graph,"author",save_dir=save_dir)
    # create_article_visualize(country_graph,"country",save_dir=save_dir)

    # # 可视化gini系数
    # 计算每个节点的加权入度
    
    # weighted_in_degree = {}
    # for node in author_graph.nodes():
    #     in_edges = author_graph.in_edges(node, data=True)
    #     weight_sum = sum([attr['count'] for (_, _, attr) in in_edges])
    #     weighted_in_degree[node] = weight_sum
    # plot_gini(weighted_in_degree,save_dir=save_dir,data_type="author")
    # countrys = readinfo("evaluate/article/country.json")
    # countrys = countrys["core"] + countrys["periphery"][:10]
    # weighted_in_degree = {c:0 for c in countrys}
    # weighted_in_degree = {}
    # for node in list(country_graph.nodes()):
    #     in_edges = country_graph.in_edges(node, data=True)
    #     weight_sum = sum([attr['count'] for (_, _, attr) in in_edges])
    #     weighted_in_degree[node] = weight_sum
    # assert isinstance(country_graph,nx.DiGraph)
    # in_degrees = dict(country_graph.in_degree())
    # plot_gini(in_degrees,save_dir=save_dir,data_type="country")
    # plot_gini(weighted_in_degree,save_dir=save_dir,data_type="country")
    # plot_citation_fairness(country_graph,save_dir=save_dir,data_type="country")

    # # 可视化自引用
    # plot_self_citation(article_graph,save_dir=save_dir)

    # 可视化原因
    # reason_path = os.path.join(save_root,"reason","reason_info.json")
    # calculate_reason(article_meta_data,reason_path)
    # reason_path = os.path.join(save_root,"reason","reason_info_cited.json")
    # calculate_reason_cited(article_meta_data,reason_path)
    # save_dir = os.path.join(save_root,"reason","visualize")
    # plot_reason_visualize(reason_path,save_dir)

    # distortion分析（MRQAP）以及可视化
    country_types = [
                    # "country_all",
                    "country_core",
                     "country_used"
                     ]
    for country_type in country_types:
        distortion_count(article_graph, 
                        article_meta_data,
                        author_info, 
                        article_meta_info_path,
                        save_dir=save_dir,
                        type=country_type,
                        group=False)  
        distortion_count(article_graph, 
                        article_meta_data,
                        author_info, 
                        article_meta_info_path,
                        save_dir=save_dir,
                        type=country_type,
                        group=False,
                        experiment_ba=True)  
        distortion_count(article_graph, 
                        article_meta_data,
                        author_info, 
                        article_meta_info_path,
                        save_dir=save_dir,
                        type=country_type,
                        group=False,
                        experiment_er=True)  
        # distortion_count(article_graph, 
        #                 article_meta_data,
        #                 author_info, 
        #                 article_meta_info_path,
        #                 save_dir=save_dir,
        #                 type=country_type,
        #                 group=False,
        #                 experiment_base=True)  
     
    # plt_distortion(beta_save_root=save_root,
    #                article_meta_data=article_meta_info,
    #                types=country_types,
    #                save_dir=save_root)
    
    # 可视化prefrential attachment/ 度分布
    # plot_degree_figures(gt_graph,save_dir=save_root,graph_name=task)
    # for graph_type, graph in graphs.items():
    #     plot_degree_figures(graph,
    #                     save_dir=save_root,
    #                     graph_name=f"{graph_type}_nodes{len(article_graph.nodes())}")
    


def build_er_article_meta_graph(article_graph:nx.DiGraph):
    import random
    countrys = readinfo("evaluate/article/country.json")
    countrys_list = []
    for v in countrys.values():
        countrys_list.extend(v)
    countrys_list = countrys_list
    for node,node_info in article_graph.nodes(data=True): # 所有国家的article number same chance
        country = random.choice(countrys_list)
        article_graph.nodes[node]["country"] = [country]
    return article_graph





# 3. 选择key值
def weighted_random_choice(countrys_articles):
    # 1. 计算总和
    total = sum(countrys_articles.values())

    # 2. 归一化概率
    probabilities = {country: count / total for country, count in countrys_articles.items()}
    # 使用random.choices()可以根据权重选择国家
    countries = list(probabilities.keys())
    weights = list(probabilities.values())
    selected_country = random.choices(countries, weights=weights, k=1)[0]
    return selected_country

def build_ba_article_meta_graph(article_meta_data:dict,
                                G:nx.DiGraph,
                                type:str):
    # use all_countrys
    countrys = readinfo("evaluate/article/country.json")
    if type =="country_all":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
    elif type =="country_core":
        countrys_list = countrys["core"]
    elif type =="country_used":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
        countrys_list = countrys_list
    
    countrys_list = [country.lower() for country in countrys_list]
    citation_array = np.zeros((len(countrys_list),len(countrys_list)))

    countrys_articles = {}
    start_point = 90
    points_all = len(article_meta_data)
    start_G = G.subgraph(nodes=list(G.nodes())[:start_point])
    # 用子图 H 初始化一个新的有向图
    new_DG = nx.DiGraph()

    # 将子图中的节点和边添加到有向图中
    new_DG.add_nodes_from(start_G.nodes(data=True))  # 保留节点属性
    new_DG.add_edges_from(start_G.edges(data=True))  # 保留边属性
    start_G = new_DG
    country_node_map = {}
    for node, node_info in start_G.nodes().items():
        for country in node_info["country"]:
            country_node_map[country] = node

    for node in list(article_meta_data.keys())[:start_point]:
        countrys_articles[country] = article_meta_data[node]["cited"]

    titles_add = list(article_meta_data.keys())
    idx = start_point
    edges = 10
    for node in list(G.nodes())[start_point:]:
        countrys = G.nodes[node]["country"]
        for country in countrys:
            if country not in countrys_articles:
                countrys_articles[country] = 0
            countrys_articles[country]+=1

        # country_max = max(countrys_articles, key=countrys_articles.get)
        country_max = weighted_random_choice(countrys_articles)
        start_G.add_node(idx, country = country_max, title = titles_add[idx])
        idx_node = idx
        # 1. 计算总和
        total = sum(countrys_articles.values())
        # 2. 归一化概率
        probabilities = {country: count / total for country, count in countrys_articles.items()}
        for country, prob in probabilities.items():
            edge_num = prob * edges
            if country not in country_node_map.keys():
                start_G.add_node(idx, country = country, title = titles_add[idx])
                idx +=1
                if idx >= len(article_meta_data):
                    break
                else:
                    continue
            for _ in range(int(edge_num)):
                start_G.add_edge(country_node_map[country], idx_node)
                
        if idx >= len(article_meta_data):
            break
        start_G.add_node(idx, country = country_max, title = titles_add[idx])
        idx +=1
        if idx >= len(article_meta_data):
            break
    return start_G





def get_countrys_list(article_meta_data,
                      article_graph,
                      map_index:dict,
                      type:str
                      ):
    countrys = readinfo("evaluate/article/country.json")
    if type =="country_all":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
    elif type =="country_core":
        countrys_list = countrys["core"]
    elif type =="country_used":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
        countrys_list = countrys_list
    
    countrys_list = [country.lower() for country in countrys_list]

    group_articles = {}
    ## 计算分组的文章相似度

    for idx, article_info in enumerate(article_meta_data.values()):
        node = map_index[article_info["title"]]
        try:
            countrys = article_graph.nodes[node]["country"]
        except:
            continue
        for country in countrys:
            if country not in countrys_list:
                continue
            if country not in group_articles.keys():
                group_articles[country] = []
            group_articles[country].append(node)
    for country in group_articles.keys():
        group_articles[country] = list(set(group_articles[country]))
    
    # 使用存在article的country
    # countrys_list_all = list(group_articles.keys())
    # countrys_list = list(filter(lambda x: x in countrys_list,countrys_list_all))

    group_number_array = np.zeros((len(countrys_list),len(countrys_list)))
    for country in group_articles.keys():
        country_index = countrys_list.index(country.lower())
        group_number_array[country_index][country_index] = len(group_articles[country])

    return group_number_array,countrys_list

def distortion_count(article_graph:nx.DiGraph,
                     article_meta_data:dict,
                    author_info:dict,
                    article_meta_info_path:str,
                    save_dir:str = "evaluate/article/distortion",
                    group:bool = False,
                    type:str = "country_all",
                    experiment_base: bool = False,
                    experiment_ba:bool = False,
                    experiment_er: bool = False,
                    use_equal_similarity: bool = False,
                    # method = "ols" # ols / pearson
                    # method = "pearson" # ols / pearson
                    method = "ols"
                    # "country_all","country_core","country_used"
                    ):
    save_dir_ori = save_dir
    if use_equal_similarity:
        save_dir = os.path.join(save_dir,"equal_similarity")
    
    save_dir = os.path.join(save_dir,f"{method}")
    if experiment_base:
        save_dir = os.path.join(save_dir,"distortion_base")
    elif experiment_er:
        save_dir = os.path.join(save_dir,"distortion_er")
    elif experiment_ba:
        save_dir = os.path.join(save_dir,"distortion_ba")
    else:
        save_dir = os.path.join(save_dir,"distortion_llm")
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    beta_save_root = os.path.join(save_dir,f"beta_dict_{type}.json")
   
    # if os.path.exists(beta_save_root):
    #     return
    article_graph = update_citation_graph(article_graph,article_meta_data,author_info)
    # author_graph = build_author_citation_graph(article_meta_data,
    #                                            author_info)
    # country_graph = build_country_citation_graph(article_meta_data,
    #                                              author_info,
    #                                              article_graph)   
    if use_equal_similarity:
        relevance_array = np.ones((len(article_meta_data),len(article_meta_data)))
    else:
        relevance_array_all_path = os.path.join(save_dir_ori,"relevance_array_all.npy")
        if os.path.exists(relevance_array_all_path):
            relevance_array = np.load(relevance_array_all_path)
        else:
            relevance_array = build_relevance_array(article_meta_data)
            np.save(relevance_array_all_path,relevance_array)

   
    article_meta_data_grouped = {}
    group_key = "topic"
    for title,article in article_meta_data.items():
        if article[group_key] not in article_meta_data_grouped:
            article_meta_data_grouped[article[group_key]] = {}
        article_meta_data_grouped[article[group_key]].update({title:article})
    
    # if os.path.exists(beta_save_root):
    #     beta_dict = readinfo(beta_save_root)
    # else: 
    #     beta_dict = {}
    beta_dict = {}
    map_index = {
        title:str(idx) for idx,title in enumerate(article_meta_data.keys())} # keep index map
    
    if beta_dict == {}:
        if "citeseer" in save_dir:
            start_time = datetime.strptime("2004-01", "%Y-%m").date()
            end_time = datetime.strptime("2011-01", "%Y-%m").date()
        else:
            start_time = datetime.strptime("2021-04", "%Y-%m").date()
            end_time = datetime.strptime("2024-08", "%Y-%m").date()
        # start_time = datetime.strptime("2004-01", "%Y-%m").date()
        # end_time = datetime.strptime("2011-01", "%Y-%m").date()
        article_meta_data = dict(filter(
            lambda x: datetime.strptime(x[1]["time"],"%Y-%m").date() <= end_time, 
                article_meta_data.items()
        ))
        index_start = len(article_meta_data)
        for idx,title in enumerate(article_meta_data.keys()):
            if datetime.strptime(article_meta_data[title]["time"],"%Y-%m").date() >= start_time:
                if (index_start > idx):
                    index_start = idx
            else:
                pass
        ## run mrqap for all articles
        time_thunk_size = int((len(article_meta_data) - index_start) // 60)
        beta_list = []
        time_lines = []
        if experiment_ba:
            article_graph = build_ba_article_meta_graph(article_meta_data,article_graph,type)
        if experiment_er:
            article_graph = build_er_article_meta_graph(article_graph)
        else:
            pass


        for i in tqdm(range(index_start,len(article_meta_data),time_thunk_size),
                    desc=f"run qap for all articles: {type}"): 
            article_meta_data_time =  dict(list(article_meta_data.items())[:i+time_thunk_size])

            group_number_array,countrys_list = get_countrys_list(article_meta_data_time,article_graph,map_index,type)
            article_graph_sub = article_graph.subgraph(list(article_graph.nodes)[:i+time_thunk_size])
            
            if experiment_ba or experiment_er:
                topic_citation_array = build_citation_group_array_from_citation(article_graph_sub,
                                                         countrys_list,
                                                         type)
            else:
                topic_citation_array = build_citation_group_array(
                                                                article_meta_data_time,
                                                                author_info,
                                                                countrys_list,
                                                            type)
                
            topic_relevance_array = build_group_relevance_array(relevance_array,
                                            # article_meta_data_time,
                                            article_graph_sub,
                                            author_info,
                                            article_graph,
                                            countrys_list,
                                            type,
                                            map_index)
            # topic_group_number_array = build_group_number_array(article_meta_data_time,
            #                                                     article_graph,
            #                                                     countrys_list,
            #                                                     map_index)

            # clear i,i?
            # for i in range(topic_citation_array.shape[0]):
            #     topic_citation_array[i][i] = 0
            #     group_number_array[i][i] = 0
            #     topic_relevance_array[i][i] = 0
            # X = {'CITATION':topic_citation_array, "NUMBER":group_number_array}

            # betas = run_mrqap(topic_relevance_array,X)
            betas = run_qap(topic_relevance_array,topic_citation_array,type = method)
            # betas = run_ols(topic_relevance_array,topic_citation_array)
            if betas is None and not np.isnan(betas[0]):
                continue
            beta_list.append(betas)
            time_lines.append(list(article_meta_data_time.values())[-1]["time"])
        beta_dict["all"] = {"y":beta_list,
                            "x":time_lines}
        
        writeinfo(beta_save_root,beta_dict)

    if group: ## 是否计算分topic的beta
        for topic, sub_article_meta_data in article_meta_data_grouped.items():
            sub_article_meta_data = dict(sorted(sub_article_meta_data.items(),
                                                key=lambda x:x[1]["time"]))
            topic_beta_list = []
            time_lines = []
            time_thunk_size = len(sub_article_meta_data) // 10

            group_number_array,countrys_list = get_countrys_list(sub_article_meta_data,article_graph,map_index)
            topic_citation_array = build_citation_group_array(article_graph,
                                                                sub_article_meta_data,
                                                                countrys_list,
                                                            type)
            try:
                for i in tqdm(range(0,len(sub_article_meta_data),time_thunk_size)):  
                    sub_article_meta_data_time = dict(list(sub_article_meta_data.items())[:i+time_thunk_size])
                    
                    topic_group_relevance_array = build_group_relevance_array(relevance_array,
                                            sub_article_meta_data_time,
                                            author_info,
                                            article_graph,
                                            countrys_list,
                                            type)
                    if topic_group_relevance_array.shape[0] ==0:
                        continue
                    betas = run_mrqap(topic_group_relevance_array,topic_citation_array)
                    topic_beta_list.append(betas)
                    time_lines.append(list(sub_article_meta_data_time.values())[-1]["time"])
                beta_dict[topic] = {"y":topic_beta_list,
                                    "x":time_lines}
            except:
                continue
    
        writeinfo(beta_save_root,beta_dict)



def run_mrqap(relevance_array: np.ndarray,
              X):
    # citation_array = citation_array[citation_array != 0]
    # relevance_array = relevance_array[relevance_array != 0]
    # relevance_array = np.zeros_like(relevance_array)
    
    Y = {'RELEVANCE':relevance_array}
    
    np.random.seed(1)
    NPERMUTATIONS = 100
    # DIRECTED = True
    DIRECTED = False
    #######################################################################
    # QAP
    #######################################################################
    start_time = time.time()
    mrqap = MRQAP(Y=Y, X=X, npermutations=NPERMUTATIONS, diagonal=False, directed=DIRECTED)
    mrqap.mrqap()
    betas = mrqap.return_betas()
    # mrqap.plot("test_gini.pdf")
    return betas
    # mrqap.summary()
    # print("--- {}, {}: {} seconds ---".format('directed' if DIRECTED else 'undirected', NPERMUTATIONS, time.time() - start_time))
    # mrqap.plot('betas')
    # mrqap.plot('tvalues')


def run_qap(relevance_array: np.ndarray,
              X,
              type = 'pearson'):
    
    np.random.seed(1)
    NPERMUTATIONS = 100
    #######################################################################
    # QAP
    #######################################################################
    start_time = time.time()
    qap = QAP(Y=relevance_array, X=X, npermutations=NPERMUTATIONS,  diagonal=False,
              type=type)
    qap.qap()
    
    return np.average(qap.betas), np.array(qap.betas).std()

    #ols
    # return qap.beta, np.std(qap.betas)
    # qap.summary()
    # mrqap.plot("test_gini.pdf")
    

def run_ols(relevance_array: np.ndarray,
              X):
    
    np.random.seed(1)
    NPERMUTATIONS = 100

    #######################################################################
    # QAP
    #######################################################################
    start_time = time.time()
    qap = QAP(Y=relevance_array, X=X, npermutations=NPERMUTATIONS,  diagonal=False)
    qap.qap()
    return np.average(qap.betas), np.array(qap.betas).std()

# from libs.ols import compute_beta_coefficient
# def run_ols(relevance_array: np.ndarray,
#               X):
    
    # mrqap.plot("test_gini.pdf")
    # return betas



def get_data(task,config):
    data_path = "LLMGraph/tasks/{task}/configs/{config}/data"
    article_meta_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                              "article_meta_info.pt"))
    author_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                       "author.pt"))
    return article_meta_info,author_info


def plt_distortion(beta_save_root:str,
                   article_meta_data:dict,
                   types:list,
                   save_dir:str ="evaluate/article/distortion"):
    betas_dict = {}
    for type in types:
        beta_save_path = os.path.join(beta_save_root,f"beta_dict_{type}.json")
        assert os.path.exists(beta_save_path)
        betas = readinfo(beta_save_path)
        betas_dict[type] = betas

    key = "CITATION"
    for topic in betas.keys():
        beta_data = []
        error_data = []
        try:
            for idx in range(len(betas[topic]["y"])):
                beta_data.append({
                        type_beta: betas_dict[type_beta][topic]["y"][idx][key][0] 
                        for type_beta in betas_dict.keys()}
                                )
                error_data.append({
                        type_beta: betas_dict[type_beta][topic]["y"][idx][key][1] 
                        for type_beta in betas_dict.keys()}
                                )
        except:
            continue
        time_data = betas_dict[types[0]][topic]["x"]
        # time_data = [f"{idx}-{time}" for idx,time in enumerate(time_data)]
        # plot_betas(time_data,beta_data,error=error_data,save_dir=save_dir,
        #            types=types,group_name=topic)
        plot_betas(time_data,beta_data,save_dir=save_dir,
                   types=types,group_name=topic)
        

def calculate_article_matrix(G:nx.DiGraph,
                            graph_name:str,
                            save_dir:str,
                            graph_true = None):
    if graph_name == "article_citation":
        calculate_matrix = [
                            "mmd",
                           "community",
                        #"control",
                        "base_info"            
                        ]
    elif graph_name in ["author_citation",
                        "co_authorship"]:
        calculate_matrix = [
                        "community",
                        "base_info"            
                        ]
    else:
        calculate_matrix = [
                          "community",
                            "base_info"            
                            ]
    # calculate_matrix = [
    #                       "community",
    #                         "base_info",
    #                         # "mmd"      
    #                         ]
        
    matrix = calculate_directed_graph_matrix( 
                                        G,
                                        graph_name=graph_name,
                                        type = "article",
                                        graph_true=graph_true,
                                        calculate_matrix = calculate_matrix)
    
    save_path = os.path.join(save_dir,f"{graph_name}_matrix.csv")
    os.makedirs(save_dir, exist_ok=True)
    matrix.to_csv(save_path)





def calculate_all_graph_matrix(
                generated_article_dir:str,
                article_meta_data:dict,
                author_info:dict,
                save_dir:str,
                task_name:str,
                val_dataset:str = None,
                article_num = None,
                graph_types:list = [
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ],
                xmin:int = 3):
    
    graphs = build_graphs(article_meta_data,
                 author_info, 
                 article_num = article_num,
                 graph_types = graph_types)
    
    article_meta_info_gt,author_info_gt = get_data(task_name,
                                                   "gt")
    
    graphs_gt = build_graphs(article_meta_info_gt,
                 author_info_gt, 
                 article_num = article_num,
                 graph_types = graph_types)
    
    for graph_type,graph in tqdm(graphs.items(),
                                 "calculate all graph matrix"):
        print("calculating", graph_type)
        if graph_type in graph_types:
            calculate_gcc_size(graph,graph_type,save_dir)
            calculate_article_matrix(graph,graph_type,save_dir,graph_true=graphs_gt[graph_type])
        
            
            
def save_degree_list(G:nx.DiGraph,
                    save_dir:str,
                    graph_name:str):
    save_degree_root = os.path.join(save_dir,"degree")
    os.makedirs(save_degree_root,exist_ok=True)
    degree_list = [G.in_degree(n) for n in G.nodes()]
    writeinfo(os.path.join(save_degree_root,f"{graph_name}.json"),degree_list)
    

def transfer_to_front_end(article_meta_data,
                          author_info:dict,
                          nodes,
                          edges,
                          paper_abstract_map:dict
                         ):
    author_num_limit = 3
    author_prompt_template = """{name}. \n
Citations: {cited}. \n
Research interest: {expertises}. \n
Interested topics: {topics}."""

    article_prompt_template = """{title}. \n
Citations: {cited}. \n
Time: {time}. \n
Abstract: {abstract}."""

    DIRECTED = True
    UNDIRECTED = False
    added_nodes = []
    added_edges = []
    paper_cites = {}
    author_cites = {}

    article_graph = build_citation_graph(article_meta_data)
    for node, node_info in article_graph.nodes(data=True):
        in_degree = article_graph.in_degree(node)
        paper_cites[node_info["title"]] = in_degree

    for title, paper_info in article_meta_data.items():
        for author in paper_info["author_ids"][:author_num_limit]:
            if author not in author_cites.keys():
                author_cites[author] = 0
            author_cites[author] += paper_cites[title]

            if author not in nodes:
                added_nodes.append(author)
            added_edges.append((author,title,"write",DIRECTED))

        for author_i in paper_info["author_ids"]:
            for author_j in paper_info["author_ids"]:
                if author_i != author_j:
                    added_edges.append((author_i,author_j,"co_authorship",UNDIRECTED))

        if title not in nodes:
            added_nodes.append(title)
        
    for title, paper_info in article_meta_data.items():
        for cited_paper in paper_info["cited_articles"]:
            if cited_paper in [*nodes, *added_nodes]:
                added_edges.append((title,cited_paper,"citation",DIRECTED))
    
    node_infos = {
        title:{"info":{"title":title,
                        "citation":paper_cites[title],
                        "time":info["time"],
                        "abstract":paper_abstract_map[title],
        },
                "type":"paper"
        }
        for title, info in article_meta_data.items()
    }
    author_node_infos = {
        author:{"info":{"name":author_info[author]["name"],
                        "citation":author_cites[author],
                        "expertises":author_info[author]["expertises"],
                        "topics":author_info[author]["topics"],
        },
                "type":"author"
            }
        for author in author_cites.keys()
    }

    added_nodes = list(set(added_nodes))
    added_edges = list(filter(lambda x:x not in edges, added_edges))
    return added_nodes, added_edges,{**node_infos,**author_node_infos}



def save_front_end_data(save_dir,author_info,article_meta_data):
    article_meta_data = dict(list(article_meta_data.items())[80:180])
    index_start = 10
    save_dir = os.path.join(save_dir,"front_end")
    os.makedirs(save_dir,exist_ok=True)

    time_author_logs = [] # {nodes:[], edges:[]}

    from LLMGraph.loader.article import DirectoryArticleLoader
    from langchain_community.document_loaders.text import TextLoader
    text_loader_kwargs={'autodetect_encoding': True}
    article_loader = DirectoryArticleLoader(
                         article_meta_data = article_meta_data,
                         path = "", 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
    docs = article_loader.load()
    paper_abstract_map = {}
    for title,idx in article_loader.doc_map.items():
        paper_abstract_map[title] = docs[idx].page_content

    start_meta_info = dict(list(article_meta_data.items())[:index_start])

    nodes = []
    edges = []
    nodes, edges, nodes_info = transfer_to_front_end(start_meta_info,
                                                     author_info,
                                                     nodes,
                                                     edges,
                                                     paper_abstract_map)
    
    
    # time_author_logs.append({"nodes":nodes, 
    #                          "edges":edges,
    #                          "nodes_info":nodes_info})
    
    # time_thunk_size = int((len(article_meta_data) - index_start) // 20)
    time_thunk_size = int((len(article_meta_data) - index_start) // 45)
    
    for i in tqdm(range(index_start,len(article_meta_data),time_thunk_size),
                desc=f"save front end data: {type}"): 
        article_meta_data_time =  dict(list(article_meta_data.items())[:i+time_thunk_size])
        added_nodes, added_edges, nodes_info = transfer_to_front_end(article_meta_data_time,author_info,nodes,
                                                                     edges,paper_abstract_map)
        time_author_logs.append({"nodes":added_nodes, "edges":added_edges, "nodes_info":nodes_info})
        edges.extend(added_edges)
        nodes.extend(added_nodes)

    save_path = os.path.join(save_dir,"nodes_edges.json")
    
    writeinfo(save_path, time_author_logs)




if __name__ == "__main__":
    args = parser.parse_args()  # 解析参数
    save_root = "LLMGraph/tasks/{task}/configs/{config}/evaluate".format(
        task = args.task,
        config = args.config
    )

    generated_article_dir = "LLMGraph/tasks/{task}/configs/{config}/data/generated_article".format(
        task = args.task,
        config = args.config
    )

    article_meta_info_path = "LLMGraph/tasks/{task}/configs/{config}/data/article_meta_info.pt".format(
        task = args.task,
        config = args.config
    )
    
    article_meta_info,author_info = get_data(args.task,args.config)

    calculate_all_graph_matrix(generated_article_dir,
                               article_meta_info,
                               author_info,
                               save_root,
                               task_name=args.task,
                               val_dataset=args.task,
                               graph_types=[
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ],
                            xmin=args.xmin)
    