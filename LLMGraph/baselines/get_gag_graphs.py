import networkx as nx
import os
import json 
import torch

def readinfo(data_dir):
    file_type = os.path.basename(data_dir).split('.')[-1]
    try:
        if file_type == "pt":
            return torch.load(data_dir)
    except:
        data_dir = os.path.join(os.path.dirname(data_dir), f"{os.path.join(os.path.basename(data_dir).split('.')[:-1])}.json")
    
    print(data_dir)
    try:
        with open(data_dir, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            return data_list
    except:
        raise ValueError("file type not supported")

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


import torch

generated_graphs = {}

test_config = "fast_vllm_445_test"
prefix_graph_sub = 445 # for big, set to the length of seed graph

# test_config = "big"
# prefix_graph_sub = 2997 # for big, set to the length of seed graph
graph_path_2 = f"LLMGraph/tasks/citeseer/configs/{test_config}/data/article_meta_info.pt"
G_2 = build_citation_graph(readinfo(graph_path_2))
generated_graphs["llmcitationciteseer"] = []
for idx in range(1,21):
    generated_graphs["llmcitationciteseer"].append(
    nx.Graph(G_2.subgraph(list(G_2.nodes())[prefix_graph_sub+idx:prefix_graph_sub+1000+idx])))

# graph_path_2 = "/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/cora_subgraph/article_meta_info.pt"
# G_2 = build_citation_graph(readinfo(graph_path_2))
# generated_graphs["llmcitationcora"] = []
# for idx in range(1,21):
#     generated_graphs["llmcitationcora"].append(
#     nx.Graph(G_2.subgraph(list(G_2.nodes())[prefix_graph_sub+idx:prefix_graph_sub+1000+idx])))


# tweet_path = "/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/tweet/llmtweet_generated.pkl"

# max_size = 512
# for key, graph_generator in zip(
#     ["friend","action","follow"],
#     [gg.data.generate_friend_graphs,
#     gg.data.generate_action_graphs,
#     gg.data.generate_follow_graphs
#     ]
# ):
#     test = graph_generator(
#                 num_graphs=20,
#                 min_size=1000,
#                 max_size=1000,
#                 dataset="test",
#                 seed=2,
#                 data_path = "/mnt/jiarui/graph_generation-main-6a992d0b151e7e9c0a23b3a351db730b4d6da666/data/netcraft/tweet/llmtweet_generated.pkl"
#             )
#     test_generated = list(filter(lambda G:len(G.subgraph(max(nx.connected_components(G), key=len)).nodes())> 2, test))
#     generated_graphs[f"llm{key}"] = test_generated


torch.save(
    generated_graphs,
    "LLMGraph/baselines/baseline_checkpoints/gag_generated.pt"
)