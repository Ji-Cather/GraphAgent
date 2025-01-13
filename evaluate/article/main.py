import networkx as nx
import argparse
import os
from evaluate.article.gcc_size import calculate_gcc_size
from evaluate.article.build_graph import (build_author_citation_graph,
                                          update_citation_graph,
                                          build_country_citation_graph,
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
    

    




def get_data(task,config):
    data_path = "LLMGraph/tasks/{task}/configs/{config}/data"
    article_meta_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                              "article_meta_info.pt"))
    author_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                       "author.pt"))
    return article_meta_info,author_info


        

def calculate_article_matrix(G:nx.DiGraph,
                            graph_name:str,
                            save_dir:str):
    calculate_matrix = [
                        "community",
                        "base_info"            
                        ]
        
    matrix = calculate_directed_graph_matrix( 
                                        G,
                                        graph_name=graph_name,
                                        type = "article",
                                        calculate_matrix = calculate_matrix)
    
    save_path = os.path.join(save_dir,f"{graph_name}_matrix.csv")
    os.makedirs(save_dir, exist_ok=True)
    matrix.to_csv(save_path)





def calculate_all_graph_matrix(
                article_meta_data:dict,
                author_info:dict,
                save_dir:str,
                task_name:str,
                article_num = None,
                graph_types:list = [
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ],
                xmin:int = 3,
                threshold = 1000):
    article_meta_data = dict(list(article_meta_data.items())[:threshold])
    graphs = build_graphs(article_meta_data,
                 author_info, 
                 article_num = article_num,
                 graph_types = graph_types)
   
    
    for graph_type,graph in tqdm(graphs.items(),
                                 "calculate all graph matrix"):
        print("calculating", graph_type)
        if graph_type in graph_types:
            calculate_gcc_size(graph,graph_type,save_dir)
            calculate_article_matrix(graph,graph_type,save_dir)
            calculate_article_power_law(graph,graph_type,save_dir,plt_flag=True,xmin=xmin)
            

def calculate_article_power_law(G:nx.DiGraph,
                            graph_name:str,
                            save_dir:str,
                             plt_flag=True,
                             xmin:int = 3):
    from evaluate.matrix import calculate_power_law
    power_law_dfs = calculate_power_law( 
                            G,
                            save_dir=save_dir,
                            graph_name=graph_name,
                            plt_flag=plt_flag,
                            xmin=xmin)
    for degree_type, df in power_law_dfs.items():
        save_path = os.path.join(save_dir,
                                 f"{graph_name}_{degree_type}_power_law.csv")
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_path)

            
def save_degree_list(G:nx.DiGraph,
                    save_dir:str,
                    graph_name:str):
    save_degree_root = os.path.join(save_dir,"degree")
    os.makedirs(save_degree_root,exist_ok=True)
    degree_list = [G.in_degree(n) for n in G.nodes()]
    writeinfo(os.path.join(save_degree_root,f"{graph_name}.json"),degree_list)
    


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

    calculate_all_graph_matrix(article_meta_info,
                               author_info,
                               save_root,
                               task_name=args.task,
                               graph_types=[
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ],
                            xmin=args.xmin,
                            threshold=-1)
    