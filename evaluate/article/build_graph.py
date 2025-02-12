import networkx as nx
from LLMGraph.utils.io import readinfo,writeinfo
import os
import json 
import re
import numpy as np
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm



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


def assign_topic(article_meta_data,
                 author_info,
                 article_meta_info_path:str):
    # assign_topic_label
    from collections import Counter
    for title,article_info in article_meta_data.items():
        if "topic" not in article_info.keys():
            authors = article_info["author_ids"]
            expertises = []
            for author in authors:
                assert isinstance(author,str)
                if author not in author_info.keys():
                    continue
                expertises.extend(author_info[author].get("expertises",[]))
            # 使用Counter计算每个元素的出现次数
            counter = Counter(expertises)
            try:
                # 找出出现次数最多的元素
                topic = counter.most_common(1)[0][0]
            except:
                topic = "Machine Learning"
            article_info["topic"] = topic
    
    from langchain_openai import ChatOpenAI
    # 按照topic进行分组
    topics = set([article["topic"] for article in article_meta_data.values()])
    print("len topics", len(topics))
    
    # regroup these topics
    prompt = """
Here're some topics:
{topics}
Group these topics into {num_group} groups, and return in dict format:
{{
    "(topic group, should be a keyword topic, e.g. 'Machine Learning')": [(a list of topics)]
}}
now respond:
"""
    if os.path.exists("evaluate/article/topic_group_map.json"):
        topic_group_map = readinfo("evaluate/article/topic_group_map.json")
    # else:
        # llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, max_tokens=2000)
        # prompt = prompt.format(topics=",".join(topics),
        #                     num_group=6)
        # response = llm.invoke(prompt).content
        # topic_group_map = json.loads(response)

        for topic_par,topics_list in topic_group_map.items():
            topics =[]
            for topic in topics_list:
                
                topics.append(topic.lower())
            topic_group_map[topic_par] = topics
        writeinfo("evaluate/article/topic_group_map.json", topic_group_map)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=200)
    for idx,title in enumerate(article_meta_data.keys()):
        article = article_meta_data[title]
        topic = article["topic"]
        article["node"] = str(idx)
        for topic_par in topic_group_map.keys():
            if topic in topic_group_map[topic_par] or topic.lower() in topic_group_map[topic_par]:
                article["topic_par"] = topic_par
                break
        if "topic_par" not in article.keys():
            prompt = """give me a group name of this topic: {topic}
Choosing from:
[{group_topics}]

respond one integer (index):
"""         
            prompt = prompt.format(topic=topic,
                                group_topics=",".join(topic_group_map.keys()))
            response = llm.invoke(prompt).content
            try:
                regex = "(\d+)"
                idx = re.search(regex, response).group(1)
                topic_par = list(topic_group_map.keys())[idx-1]
            except:
                topic_par = "Machine Learning"
                for candidate in topic_group_map.keys():
                    if candidate.lower() in response.lower():
                        topic_par = candidate
                        break
            topic_group_map[topic_par].append(topic.lower())
            article["topic_par"] = topic_par

    writeinfo("evaluate/article/topic_group_map.json", topic_group_map)
    # writeinfo(article_meta_info_path,article_meta_data)
    return article_meta_data

    
def build_relevance_array(article_meta_data:dict):
    """暂时只考虑 每个country所有的文章 其内容相似度 (基于embedding)
    """
    
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders.text import TextLoader
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from LLMGraph.loader.article import DirectoryArticleLoader
    text_loader_kwargs={'autodetect_encoding': True}
    article_loader = DirectoryArticleLoader(
                         article_meta_data = article_meta_data,
                         path = "", 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
    docs = article_loader.load()
    prompt_template = PromptTemplate.from_template("""
Title: {title}
Cited: {cited}
Publish Time: {time}
Content: {page_content}""")
    docs_str = [prompt_template.format(**doc.metadata,
                                       page_content = doc.page_content) 
                                       for doc in docs]
    docs_embed = embeddings.embed_documents(docs_str)
    
    relevance_array = []

    for i, article_i in tqdm(enumerate(article_meta_data.keys()),
                             "building relevance array..."):
        array_sub = []
        for j, article_j in enumerate(article_meta_data.keys()):
            if i==j:
                array_sub.append(0)
                continue
            embed_i = docs_embed[i]
            embed_j = docs_embed[j]
            similarity = cosine_similarity([embed_i], [embed_j])[0][0]
            array_sub.append(similarity)
        relevance_array.append(array_sub)
    relevance_array = np.array(relevance_array)
    return relevance_array

    


def build_group_relevance_array(relevance_array,
                              article_sub_graph,
                            author_info:dict,
                            article_graph:nx.DiGraph,
                            countrys_list:list,
                            type = "country_all",
                            map_index={}):
    """type: country_all/ country_core"""
    
    group_articles = {}
    ## 计算分组的文章相似度

    
    for node_,node_info_  in article_sub_graph.nodes(data=True):
        node = map_index[node_info_["title"]]
        try:
            countrys = article_graph.nodes[node]["country"]
        except:
            continue
        for country in countrys:
            country = country.lower()
            if country not in countrys_list:
                continue
            if country not in group_articles.keys():
                group_articles[country] = []
            group_articles[country].append(node)
    for country in group_articles.keys():
        group_articles[country] = list(set(group_articles[country]))

    group_relevance_array = np.zeros((len(countrys_list),len(countrys_list)))

    """new method"""
    # for idx, title in enumerate(article_meta_data.keys()):
    #     cited_titles = article_meta_data[title]["cited_articles"]
    #     for cited_title in cited_titles:
    #         if cited_title not in article_meta_data.keys():
    #             continue
    #         cited_node = map_index[cited_title]
    #         i_countrys = article_graph.nodes(data=True)[str(idx)]["country"]
    #         j_countrys = article_graph.nodes(data=True)[cited_node]["country"]
    #         for i_country in i_countrys:
    #             for j_country in j_countrys:
    #                 try:
    #                     i_countey_index = countrys_list.index(i_country.lower())
    #                     j_country_index = countrys_list.index(j_country.lower())
    #                     group_relevance_array[j_country_index,i_countey_index] += relevance_array[int(cited_node),
    #                                                                                               int(map_index[title])]
    #                     if i_countey_index!=j_country_index:
    #                        group_relevance_array[i_countey_index,j_country_index] += relevance_array[int(cited_node),
    #                                                                                               int(map_index[title])]
                           
    #                 except Exception as e:
    #                     if type == "country_all":
    #                         raise Exception(f"unsupported country {i_country}, {j_country}")
    #                     continue


    # """original method"""
    

    for idx_i, country_i in enumerate(group_articles.keys()):
        for idx_j,country_j in enumerate(group_articles.keys()):

            i_countey_index = countrys_list.index(country_i.lower())
            j_country_index = countrys_list.index(country_j.lower())
            # if country_i==country_j:
            #     group_relevance_array[i_countey_index,j_country_index] += 1
            #     continue
            articles_i = group_articles[country_i]
            articles_j = group_articles[country_j]
            rele_all = []
            for article_i in articles_i:
                for article_j in articles_j:
                    rele_all.append(relevance_array[int(article_i),int(article_j)])
            # group_relevance_array[i_countey_index,j_country_index] += sum(rele_all)/len(rele_all)
            group_relevance_array[i_countey_index,j_country_index] += sum(rele_all)
    X = group_relevance_array
    min_val = np.min(X)
    max_val = np.max(X)

    # 全局归一化
    normalized_array = (X - min_val) / (max_val - min_val)
    normalized_array = normalized_array.T
    normalized_array = np.nan_to_num(normalized_array )
    # # 按行进行Z得分归一化
    return normalized_array
    # X = group_relevance_array
    # # X_mean = X.mean(axis=1, keepdims=True)
    # # X_std = X.std(axis=1, keepdims=True)
    # # X_standardized = (X - X_mean) / X_std
    # # X_normalized = X_standardized
    # X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    # try:
    #     X_normalized = X / X_norm
    # except:
    #     X_normalized = X
    # X_normalized = np.nan_to_num(X_normalized , nan=0.0)
    # return X_normalized



# def build_group_number_array(article_meta_data:dict,
#                             countrys_list:list,
#                             article_graph,
#                             map_index={}):
#     """type: country_all/ country_core"""
    
#     group_articles = {}
#     ## 计算分组的文章相似度

#     for idx, article_info in enumerate(article_meta_data.values()):
#         node = map_index[article_info["title"]]
#         try:
#             countrys = article_graph.nodes[node]["country"]
#         except:
#             continue
#         for country in countrys:
#             if country not in countrys_list:
#                 continue
#             if country not in group_articles.keys():
#                 group_articles[country] = []
#             group_articles[country].append(node)
#     for country in group_articles.keys():
#         group_articles[country] = list(set(group_articles[country]))
    
    
#     for idx_i, country_i in enumerate(group_articles.keys()):
#         for idx_j,country_j in enumerate(group_articles.keys()):

#             i_countey_index = countrys_list.index(country_i.lower())
#             j_country_index = countrys_list.index(country_j.lower())
            
#             articles_i = group_articles[country_i]
#             articles_j = group_articles[country_j]
#             rele_all = []
#             for article_i in articles_i:
#                 for article_j in articles_j:
#                     rele_all.append(relevance_array[int(article_i),int(article_j)])
                
#             group_relevance_array[i_countey_index,j_country_index] += sum(rele_all)/len(rele_all)

#     return group_number_array


        

def build_citation_group_array(article_meta_data:dict = {},
                               author_info:dict = {},
                               countrys_list =[],
                               type = "country_all") -> np.ndarray:
    """type: country_all/country_core/country_used"""
    # use all_countrys
    
    citation_array = np.zeros((len(countrys_list),len(countrys_list)))
    article_graph = build_citation_graph(article_meta_data)
    article_graph = update_citation_graph(article_graph,article_meta_data,author_info)

    map_index = {title:str(idx) for idx,title in enumerate(article_meta_data.keys())}

    for idx, title in enumerate(article_meta_data.keys()):

        # # for country_i in article["country"]:
        # node = map_index[title]
        # article = article_meta_data[title]
        # try:
        #     node_info = article_graph.nodes(data=True)[node]
        # except:
        #     continue
        # cited_nodes = article_graph.successors(node)
        cited_titles = article_meta_data[title]["cited_articles"]
        for cited_title in cited_titles:
            if cited_title not in article_meta_data.keys():
                continue
            cited_node = map_index[cited_title]
            i_countrys = article_graph.nodes(data=True)[str(idx)]["country"]
            j_countrys = article_graph.nodes(data=True)[cited_node]["country"]
            for i_country in i_countrys:
                for j_country in j_countrys:
                    try:
                        i_countey_index = countrys_list.index(i_country.lower())
                        j_country_index = countrys_list.index(j_country.lower())
                        citation_array[j_country_index,i_countey_index] += 1
                        if i_countey_index!=j_country_index:
                            citation_array[i_countey_index,j_country_index] += 1
                    except Exception as e:
                        if type == "country_all":
                            raise Exception(f"unsupported country {i_country}, {j_country}")
                        continue
    # 按行进行Z得分归一化
    X = citation_array
    # return X
    # X_mean = X.mean(axis=1, keepdims=True)
    # X_std = X.std(axis=1, keepdims=True)
    # X_standardized = (X - X_mean) / X_std

    # X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    # X_normalized = X / X_norm
    # X_normalized = np.nan_to_num(X_normalized , nan=0.0)
    # return X_normalized, countrys_list    
    # return X_normalized, countrys_list
    # 计算最小值和最大值
    min_val = np.min(X)
    max_val = np.max(X)

    # 全局归一化
    normalized_array = (X - min_val) / (max_val - min_val)
    # return X
    # return normalized_array
    return normalized_array



def build_citation_group_array_from_citation(article_graph,
                               countrys_list =[],
                               type = "country_all") -> np.ndarray:
    """type: country_all/country_core/country_used"""
    # use all_countrys
    
    citation_array = np.zeros((len(countrys_list),len(countrys_list)))
   
    for node,node_info in article_graph.nodes(data=True):

        # # for country_i in article["country"]:
        # node = map_index[title]
        # article = article_meta_data[title]
        # try:
        #     node_info = article_graph.nodes(data=True)[node]
        # except:
        #     continue
        # cited_nodes = article_graph.successors(node)
        cited_nodes = article_graph.successors(node)
        for cited_node in cited_nodes:
            i_countrys = article_graph.nodes(data=True)[node]["country"]
            j_countrys = article_graph.nodes(data=True)[cited_node]["country"]
            for i_country in i_countrys:
                for j_country in j_countrys:
                    try:
                        i_countey_index = countrys_list.index(i_country.lower())
                        j_country_index = countrys_list.index(j_country.lower())
                        citation_array[j_country_index,i_countey_index] += 1
                        if i_countey_index!=j_country_index:
                            citation_array[i_countey_index,j_country_index] += 1
                    except Exception as e:
                        if type == "country_all":
                            raise Exception(f"unsupported country {i_country}, {j_country}")
                        continue
    # 按行进行Z得分归一化
    X = citation_array
    # return X
    # X_mean = X.mean(axis=1, keepdims=True)
    # X_std = X.std(axis=1, keepdims=True)
    # X_standardized = (X - X_mean) / X_std

    # X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    # X_normalized = X / X_norm
    # X_normalized = np.nan_to_num(X_normalized , nan=0.0)
    # return X_normalized, countrys_list    
    # return X_normalized, countrys_list
    # 计算最小值和最大值
    min_val = np.min(X)
    max_val = np.max(X)

    # 全局归一化
    normalized_array = (X - min_val) / (max_val - min_val)
    # return X
    # return normalized_array
    normalized_array = np.nan_to_num(normalized_array , nan=0.0)
    return normalized_array


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

    