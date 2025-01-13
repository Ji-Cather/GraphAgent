import networkx as nx
import os
import json 
import re
import numpy as np
import torch
import pickle
import random

def readinfo(data_dir):
    file_type = os.path.basename(data_dir).split('.')[1]
    try:
        if file_type == "pt":
            return torch.load(data_dir)
    except:
        data_dir = os.path.join(os.path.dirname(data_dir), f"{os.path.basename(data_dir).split('.')[0]}.json")
    
    print(data_dir)
    try:
        with open(data_dir, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            return data_list
    except:
        raise ValueError("file type not supported")


def build_country_citation_graph(article_meta_data:dict,
                                 author_info:dict,
                                 article_graph:nx.DiGraph) -> nx.DiGraph:

    """构建国家citation network
    Args:
        article_graph (nx.DiGraph): _description_
    Returns:
        nx.DiGraph: _description_
    """
    DG = nx.MultiDiGraph()
    # countrys = [author["country"] for author in author_info.values()]
    # countrys = list(set(countrys))

    countrys = readinfo("evaluate/article/country.json")
    countrys_list = []
    for v in countrys.values():
        for v_ in v:
            countrys_list.append(v_.lower())


    for node,node_info in article_graph.nodes().items():
        cited_nodes = article_graph.successors(node)
        for cited_node in cited_nodes:
            i_countrys = node_info["country"]
            j_countrys = article_graph.nodes(data=True)[cited_node]["country"]
            for i_country in i_countrys:
                for j_country in j_countrys:
                    try:
                        DG.add_edge(i_country.lower(),j_country.lower())
                    except:
                        continue
    DG = transfer_multidi_to_di(DG)
    return DG


def build_co_citation_graph(citation_network:nx.DiGraph):
    # 创建共引网络（无向图）
    co_citation_network = nx.Graph()

    # 初始化每篇论文被引用的集合
    cited_by = {}
    for cited, citing in citation_network.edges():
        if cited not in cited_by:
            cited_by[cited] = set()
        cited_by[cited].add(citing)

    # 遍历每篇论文，找到可能的共引论文对
    for paper in citation_network:
        citing_papers = cited_by.get(paper, [])
        for u in citing_papers:
            for v in citing_papers:
                if u != v:
                    co_citation_network.add_edge(u, v)
    
    return co_citation_network


def build_co_authorship_network(
                article_meta_data:dict = {},
                author_info:dict = {}):
    co_authorship_network = nx.Graph()
    for author_id,author in author_info.items():
        co_authorship_network.add_node(author_id,
                    name=author["name"],
                    institution=author["institution"],
                    country=author["country"],
                    topics = author["topics"])
    for title, article in article_meta_data.items():
        authors = article["author_ids"]
        for author_i in authors:
            for author_j in authors:
                if author_i == author_j:
                    continue
            co_authorship_network.add_edge(
                author_j, author_i)
    return co_authorship_network        
    

def build_bibliographic_coupling_network(citation_network:nx.DiGraph):
    # 创建引用共现网络（无向图）
    bibliographic_coupling_network = nx.Graph()

    # 初始化每篇论文引用的集合
    references = {}
    for citing, cited in citation_network.edges():
        if citing not in references:
            references[citing] = set()
        references[citing].add(cited)

    # 遍历每篇论文，找到可能的引用共现论文对
    for paper in citation_network:
        referenced_papers = references.get(paper, set())
        for other_paper in citation_network:
            if other_paper != paper and not bibliographic_coupling_network.has_edge(paper, other_paper):
                common_references = referenced_papers & references.get(other_paper, set())
                if common_references:
                    bibliographic_coupling_network.add_edge(paper, other_paper)
    return bibliographic_coupling_network





def build_author_citation_graph(article_meta_data:dict,
                                author_info:dict):
    """构建作者citation network
    Args:
        article_graph (nx.DiGraph): _description_
    Returns:
        nx.DiGraph: _description_
    """
    DG = nx.MultiDiGraph()
    group_id = 0
    group_author_ids = {} # author_id: group_id
    for author_id,author in author_info.items():
        author_ids = [author_id,* author["co_author_ids"]]
        found_group = None
        for group_author_id_ in author_ids:
            if group_author_id_ in group_author_ids.keys():
                found_group = group_author_ids[group_author_id_]
                break

        if found_group is None:
            found_group = group_id
            group_id += 1
        for group_author_id_ in author_ids:
            group_author_ids[group_author_id_] = found_group
    """add nodes"""
    for author_id,author in author_info.items():
        DG.add_node(author_id,
                    group=group_author_ids[author_id],
                    name=author["name"],
                    institution=author["institution"],
                    country=author["country"],
                    topics = author["topics"])

    """add edges"""
    for article_id, article_info in article_meta_data.items():
        author_ids = article_info["author_ids"]
        author_ids = list(filter(lambda a_id:a_id in author_info.keys(),
                        author_ids))
        for cited_article in article_info["cited_articles"]:
            if cited_article not in article_meta_data.keys():
                continue
            cited_author_ids = article_meta_data[cited_article]["author_ids"]
            cited_author_ids = list(filter(lambda a_id:a_id in author_info.keys(),
                       cited_author_ids))

            for cited_author_id in cited_author_ids:
                for author_id in author_ids:
                    DG.add_edge(str(author_id),str(cited_author_id))

    DG = transfer_multidi_to_di(DG)
    return DG


def transfer_multidi_to_di(multi_digraph:nx.MultiDiGraph):
    digraph = nx.DiGraph()

    # 针对MultiDiGraph中的每一对节点，计算边的数量，并将这个信息添加为DiGraph边的属性
    digraph.add_nodes_from(multi_digraph.nodes(data=True))
    for u, v in multi_digraph.edges():
        if not digraph.has_edge(u, v):
            # 如果这条边在DiGraph中尚不存在，新建边并初始化数量为1
            digraph.add_edge(u, v, count=1)
        else:
            # 如果这条边在DiGraph中已经存在，增加边的数量
            digraph[u][v]['count'] += 1

    return digraph

def build_citation_graph(article_meta_data:dict = {}):
    DG = nx.DiGraph()
    
    map_index = {
        title:str(idx) for idx,title in enumerate(article_meta_data.keys())}
    
    
    for title in article_meta_data.keys():
        cited_idx = map_index.get(title)
        time = article_meta_data[title]["time"]
        DG.add_node(cited_idx,title=title,time=time, topic = article_meta_data[title]["topic"])
    
    for title, article_info in article_meta_data.items():
        cited_articles = article_info.get("cited_articles",[])
        title_idx = map_index.get(title)
        
        edges = []
        for cite_title in cited_articles:
            cited_idx = map_index.get(cite_title)
            if cited_idx is not None:
                edges.append((cited_idx,title_idx))            
        DG.add_edges_from(edges)
        
    return DG

def update_citation_graph(DG:nx.DiGraph,
                          article_meta_data:dict,
                          author_info:dict):
    """增加citation network 国家attribute

    Args:
        DG (nx.DiGraph): _description_
        article_meta_data (dict): _description_
    """
    countrys = readinfo("evaluate/article/country.json")
    countrys_list = []
    for v in countrys.values():
        for v_ in v:
            countrys_list.append(v_.lower())
    map_index = {
            title:str(idx) for idx,title in enumerate(article_meta_data.keys())}
    reverse_map = {str(idx):title for title,idx in map_index.items()}
    for node,node_info in DG.nodes().items():
        title = reverse_map[node]
        if "author_names" in article_meta_data[title].keys():
            author_ids = article_meta_data[title]["author_names"]
        else:
            author_ids = list(filter(lambda a_id:a_id in author_info.keys(),
                                 article_meta_data[title]["author_ids"]))
        DG.nodes[node]["country"] = []

        for author_id in author_ids:
            try:
                if author_info[author_id]["country"].lower() in countrys_list:
                    DG.nodes[node]["country"].append(author_info[author_id]["country"].lower())
            except:
                pass
    nodes_filtered = list(filter(lambda n:len(n[1]["country"])>0,DG.nodes(data=True)))
    DG = DG.subgraph([ node[0] for node in nodes_filtered])
    return DG

    

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

def generate_citation_graphs(num_graphs,
                             min_size, 
                             max_size, 
                             dataset = 'train', #
                             graph_path = "/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/citation/citeseer/raw/article_meta_info_abstract.pt",
                             seed=0):

    

    G = build_citation_graph(readinfo(graph_path))
    n = len(list(G.nodes()))
    if dataset == "train" or dataset == "val":
        G = G
    else: # 预测后续graph生长
        # G = G.subgraph(nodes[-2000:])
        Gs = []
        for _ in range(1,1+num_graphs):
            graph_gt_sub = nx.Graph(G.subgraph(list(G.nodes())[-1000-_:-_]))
            Gs.append(graph_gt_sub)
        return Gs
        
    # 随机选择3个节点
    graphs = []
    rng = np.random.default_rng(seed)
    max_size = max_size if max_size !=-1 else len(G.nodes())
    for i in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        # 随机选择起始索引
        start_index = random.randint(0, max_size - n)
        random_nodes = list(G.nodes())[start_index:start_index+n]
        H = G.subgraph(random_nodes)
        H = nx.Graph(H)
        graphs.append(H)
    return graphs

def generate_citation():
    graph_generator = generate_citation_graphs

    key = "citation"
    
    train_size = 160
    val_size = 32
    test_size = 20
    min_size = 64
    max_size = 512
    
    graph_path = "LLMGraph/tasks/citeseer/data/article_meta_info.pt"
    train = graph_generator(
                num_graphs=train_size,
                min_size=min_size,
                max_size=max_size,
                dataset="train",
                graph_path= graph_path,
                seed=0,
            )
    train = list(filter(lambda G:len(G.subgraph(max(nx.connected_components(G), key=len)).nodes())> 2, train))
            
    validation = graph_generator(
        num_graphs=val_size,
        min_size=min_size,
        max_size=max_size,
        dataset="val",
        graph_path= graph_path,
        seed=1,
    )
    test = graph_generator(
        num_graphs=test_size,
        min_size=min_size,
        max_size=max_size,
        dataset="test",
        graph_path= graph_path,
        seed=2,
    )
    dataset = {
        "train": train,
        "val": validation,
        "test": test,
    }

    # data_path = Path(".data")
    data_path = "LLMGraph/baselines/baseline_checkpoints"
    with open(data_path / f"llm{key}citeseer.pkl", "wb") as f:
        pickle.dump(dataset, f)