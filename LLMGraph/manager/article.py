""" load the basic infos of authors"""

import os
from LLMGraph.manager.base import BaseManager
from agentscope.message import Msg
from . import manager_registry as ManagerRgistry
from typing import List,Union,Any
from copy import deepcopy
from langchain_community.document_loaders import TextLoader
from LLMGraph.loader.article import DirectoryArticleLoader

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import random
import networkx as nx
from LLMGraph.utils.io import readinfo,writeinfo
from LLMGraph.retriever import retriever_registry

from LLMGraph.output_parser import find_and_load_json

from datetime import datetime,date,timedelta
import copy
from agentscope.models import  ModelWrapperBase
import time
import numpy as np

def parse_date_str(date_str):
    """
    将包含年份和月份或年份、月份和日的字符串（如'2401'或'240101')转换为datetime对象。
    date_str: 一个字符串，格式为'YYYYMM'或'YYYYMMDD'。
    返回: 对应的datetime对象。
    """
    if len(date_str) == 4:
        return datetime.strptime(date_str, "%y%m").date()
    if len(date_str) == 7:
        return datetime.strptime(date_str, "%Y-%m").date()
    if len(date_str) == 6:  # 年份和月份
        return datetime.strptime(date_str, "%Y%m").date()
    elif len(date_str) == 8:  # 年份、月份和日
        return datetime.strptime(date_str, "%Y%m%d").date()
    elif len(date_str) == 10:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    else:   
        raise ValueError(f"Unsupported date string format {date_str}")


def transfer_time(json_data:dict):
    for k,v in json_data.items():
        v["time"] = parse_date_str(v["time"])
    return json_data

def pre_process(article_meta_data:dict,
                author_data:dict,
                cur_time:date,
                load_history:bool=False):
    
    """返回按照时间排序的article_meta_data和author_data,以及切分到当前时间步的ratings矩阵
    article_meta_data: dict 切分到cur_time, author_data: dict不切分
    """
    article_meta_data = transfer_time(article_meta_data)
    author_data = transfer_time(author_data)
    if load_history:
        return article_meta_data, author_data
    article_meta_data = sorted(article_meta_data.items(), key=lambda x: x[1]["time"])
    author_data = dict(sorted(author_data.items(), key=lambda x: x[1]["time"]))
    article_meta_data = dict(filter(lambda x: x[1]["time"] <= cur_time, article_meta_data))
    return article_meta_data, author_data


@ManagerRgistry.register("article")
class ArticleManager(BaseManager):
    """
        manage infos of different community.
    """
    article_meta_data :dict = {}
    author_data :dict = {}
    retriever : Any
    online_retriever : Any = None
    article_dir: str
    generated_article_dir: str
    meta_article_path:str
    author_path:str
    article_loader: Union[DirectoryArticleLoader,None]
    db: Any
    dataset:str
    experiment:list = [], # default/shuffle/false cite
    topic_agents:dict = {} # topic: [agent_name]
    simulation_time: int = 0
    author_pointer_args:dict = {
        "cur_author_ids": [],
        "cnr_time":  date.min,
        "overloaded_ids":[],
        "overloaded_threshold":100
    }
    start_time:date = date.min 
    big_name_list:list = []
    countrys :dict = {}
    llm: ModelWrapperBase
    embeddings: Any
    article_written_num:int = 0
    article_write_configs:dict = {}

    class Config:
        arbitrary_types_allowed = True
    
    
    
    @classmethod
    def load_data(cls,
                  article_meta_path,
                  author_path,
                  article_dir,
                  generated_article_dir,
                  task_path,
                  config_path,
                  retriever_kwargs,
                  llm,
                  cur_time: date,
                  tool_kwargs,
                  control_profile,
                  online_retriever_kwargs:dict = {},
                  experiment:list = [], # default/shuffle/false cite
                  ):
        
        retriever_kwargs = copy.deepcopy(retriever_kwargs)
        try:
            article_meta_path_join = os.path.join(os.path.dirname(config_path),article_meta_path)
            author_path_join = os.path.join(os.path.dirname(config_path),author_path)
            assert os.path.exists(article_meta_path_join) and os.path.exists(author_path_join)
        except:
            article_meta_path_join = os.path.join(task_path,article_meta_path)
            author_path_join =  os.path.join(task_path,author_path)

        article_dir = os.path.join(task_path,article_dir)
        if "llm_agent" in article_dir:
            dataset = "llm_agent"
        elif "citeseer" in article_dir:
            dataset = "citeseer"
        elif "cora" in article_dir:
            dataset = "cora"
        else:raise ValueError("unknown dataset: {}".format(article_dir))

        article_meta_data = readinfo(article_meta_path_join)
        author_data = readinfo(author_path_join)
        
        countrys = readinfo("evaluate/article/country.json")
        
        assert os.path.exists(article_dir),"no such file path: {}".format(article_dir)
        generated_article_dir = os.path.join(os.path.dirname(config_path),
                                             generated_article_dir)
        
        log_info_path = os.path.join(os.path.dirname(generated_article_dir),
                                    "log_info.json")
        topic_agents ={}
        if os.path.exists(log_info_path):
            log_info = readinfo(log_info_path)
            article_write_configs = log_info.get("article_write_configs",{})
            simulation_time = log_info["simulation_time"]
            cur_time = parse_date_str(log_info["last_added_time"])
            article_written_num = len(os.listdir(generated_article_dir))
            author_pointer_args={
                "cur_time":cur_time,
                "cur_author_ids": log_info["cur_author_ids"],
                "overloaded_ids": log_info["overloaded_ids"],
                "overloaded_threshold": log_info["overloaded_threshold"]
                }
            filter_key = "topics"
            for agent_name in [*log_info["cur_author_ids"],
                               ]:
                agent_info = author_data[agent_name]
                for topic in agent_info.get(filter_key,[]):
                    if topic not in topic_agents.keys():
                        topic_agents[topic] = [agent_name]
                    elif agent_name not in topic_agents[topic]:
                        topic_agents[topic].append(agent_name)

            start_time = parse_date_str(log_info["cur_time"])
            article_meta_data, author_data = pre_process(article_meta_data, 
                                                         author_data, 
                                                         cur_time,
                                                         load_history=True)
        else:
            simulation_time = 0
            article_written_num = 0
            author_pointer_args:dict = {
                "cur_time":  date.min,
                "cur_author_ids": [],
                "overloaded_ids":[],
                "overloaded_threshold":100
            }        
            article_write_configs = {}
            start_time = date.min
            article_meta_data, author_data = pre_process(article_meta_data, 
                                                         author_data, 
                                                         cur_time,
                                                         load_history= False)
        if online_retriever_kwargs!={}:
            online_retriever = retriever_registry.from_db(**online_retriever_kwargs)
        else:online_retriever = None
        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if not os.path.exists(generated_article_dir):
            os.makedirs(generated_article_dir)
            
        
        
    
        return cls(
           article_meta_data = article_meta_data,
           author_data = author_data,
           retriever = None,
           retriever_kwargs = retriever_kwargs, 
           db = None,
           article_loader = None,
           article_dir = article_dir,
           generated_article_dir = generated_article_dir,
           author_path = author_path,
           meta_article_path = article_meta_path,
           embeddings = embeddings,
           llm = llm,
           simulation_time = simulation_time,
           countrys = countrys,
           start_time = start_time,
           article_written_num = article_written_num,
           author_pointer_args =  author_pointer_args,
           tool_kwargs = tool_kwargs,
           control_profile = control_profile,
           experiment = experiment,
           online_retriever = online_retriever,
           dataset = dataset,
           topic_agents = topic_agents,
           article_write_configs = article_write_configs
           )
    
    def calculate_avg_degree_citation(self):
        DG = nx.DiGraph()
        
        map_index = {
            title:idx for idx,title in enumerate(self.article_meta_data.keys())}
        
        for title in self.article_meta_data.keys():
            cited_idx = map_index.get(title)
            time = self.article_meta_data[title]["time"].strftime("%Y-%m")
            DG.add_node(cited_idx,title=title,time=time)
        
        for title, article_info in self.article_meta_data.items():
            cited_articles = article_info.get("cited_articles",[])
            title_idx = map_index.get(title)
            
            edges = []
            for cite_title in cited_articles:
                cited_idx = map_index.get(cite_title)
                if cited_idx is not None:
                    edges.append((cited_idx,title_idx))            
            DG.add_edges_from(edges)
        degree_list = list(dict(DG.out_degree).values())
        return np.mean(degree_list), np.std(degree_list, ddof=0)
    
    def get_article_write_configs(self):
        return self.article_write_configs

    def get_start_time(self):
        return datetime.strftime(self.start_time,"%Y-%m-%d")
        
    def update_db(self):
        assert os.path.exists(self.generated_article_dir)
        docs_update = self.article_loader.add_doc_dir(self.article_meta_data)
        docs = self.article_loader.load()
        # self.db = FAISS.from_documents(docs, self.embeddings)
        if self.db is None and len(docs) > 0:
            self.db = FAISS.from_documents(docs, self.embeddings)
        elif len(docs_update) > 0:
            db_update = FAISS.from_documents(docs_update, self.embeddings)
            self.db.merge_from(db_update)
            # assert (len(self.db.docstore._dict) == len(self.article_meta_data)), \
            #     "db size not match article meta data size"
        else:
            assert self.db is not None, "error in db"

        self.retriever_kwargs["vectorstore"] = self.db
        self.retriever_kwargs["compare_function_type"] = "article"
        self.retriever = retriever_registry.from_db(**self.retriever_kwargs)
        
    def get_retriever(self):
        if self.retriever is None:
            text_loader_kwargs={'autodetect_encoding': True}
            self.article_loader = DirectoryArticleLoader(
                         article_meta_data = self.article_meta_data,
                         path = self.article_dir, 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
            self.update_db()
        return self.retriever

    

    def add_author(self,
                   cur_time,
                   author_time_delta,
                   author_num_per_delta
                   ):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        author_time_delta = timedelta(seconds=author_time_delta)
        author_num_per_delta = int(author_num_per_delta)

        if cur_time - self.author_pointer_args["cur_time"] >  author_time_delta:
            added_round_num = 0

            filtered_keys = []
            for key, value in self.author_data.items():
                if value["time"] <= cur_time:
                    filtered_keys.append(key)

            added_round_num += len(filtered_keys) - len(self.author_pointer_args["cur_author_ids"])

            remain_add = author_num_per_delta - added_round_num
            if remain_add > 0:
                for i in range(0, remain_add, 5):
                    add_indexs = self.update_author_db(cur_time, 5)
                    filtered_keys.extend(add_indexs)

            overloaded_keys = []
            other_keys = []
            for filter_key in filtered_keys:
                if len(self.author_data[filter_key]["articles"]) > \
                self.author_pointer_args["overloaded_threshold"]:
                    overloaded_keys.append(filter_key)
                else:
                    other_keys.append(filter_key)
            
            self.author_pointer_args["cur_author_ids"] = other_keys
            self.author_pointer_args["overloaded_ids"] = list(set(
                                                        [*overloaded_keys,
                                                        *self.author_pointer_args["overloaded_ids"]]))

            filter_key = "topics"
            for agent_name in filtered_keys:
                agent_info = self.author_data[agent_name]
                for topic in agent_info.get(filter_key,[]):
                    if topic not in self.topic_agents.keys():
                        self.topic_agents[topic] = [agent_name]
                    elif agent_name not in self.topic_agents[topic]:
                        self.topic_agents[topic].append(agent_name)

            self.author_pointer_args["cur_time"] = cur_time
        self.update_big_name_list()
        
        
        
    def update_author_db(self,
                         cur_time,
                         author_num_per_delta):
        template = """
I would like you to generate a series of random author's personal information.
These authors are interested in computer science, they are experts in various fields of CS.

I need you to give a list of author infos with the constraints for each attribute as follows:

  (1) Name: Author's name
  (2) Expertises: a list, The author's areas of expertises can be selected from the following areas:{expertises_list}
  (3) Institution: The author's institution, you can choose whatever institution you want, just give me one institution name
  (4) Country: The author's country, you can choose whatever institution you want,just give me one country name corresponding to the institution
  (5) Topics: a list, The topics this author is interested in, can be selected from the following topics:{topics_list}
  Here's some common used countrys you can infer to:
  {countrys}

  Please generate me a list of {author_num} different authors, which can be loaded by eval function in python:
  [{{
  "name":"",
  "expertises":[],
  "institution":"",
  "country":"",
  "topics":[]
  }},
  ...,
  {{
  "name":"",
  "expertises":[],
  "institution":"",
  "country":"",
  "topics":[]
  }}]

  Now please generate:
"""
        topics_available = self.get_topics_available()
        countrys_available = []
        upper_str = 20
        for v in self.countrys.values():
            for country_one in v:
                countrys_available.append(country_one.lower())
        
        
        countrys = ",".join(countrys_available[:upper_str])

        expertises = ",".join(self.get_expertises())[:500]
        prompt_inputs = {
            "expertises_list": "[{expertises}]".format(expertises = expertises),
            "topics_list":"[{topics_available}]".format(topics_available = ",".join(
                topics_available)),
            "author_num":author_num_per_delta, # the number of the authors added per round
            "countrys": countrys
        }
        
        countrys_available = []
        for v in self.countrys.values():
            for country_one in v:
                countrys_available.append(country_one.lower())
        

        from agentscope.message import Msg
        prompt = template.format_map(prompt_inputs)
        prompt = self.llm.format(
            Msg("system","You're a helpful assistant","system"),
            Msg("user",prompt,"user"))
        authors_str = self.llm(prompt)
        add_indexs = []
        from LLMGraph.output_parser.base_parser import find_and_load_json
        try:
            authors = find_and_load_json(authors_str.text,"list")
            index = 0
            for idx_author,author_info in self.author_data.items():
                if idx_author.isdigit():
                    try:
                        index = int(idx_author) if int(idx_author) > index else \
                            index
                    except:continue
            index += 1
            
            for author_info in authors:
                assert str(index) not in self.author_data.keys()
                author_info["name"] = str(author_info["name"])
                if isinstance(author_info["expertises"], str):
                    try:
                        author_info["expertises"] = eval(author_info["expertises"])
                        assert isinstance(author_info["expertises"], list)
                    except:
                        author_info["expertises"] = [author_info["expertises"]]
                country = author_info["country"].lower()
                try:
                    assert country in countrys_available
                    institution = author_info["institution"].lower()
                    country = author_info["country"].lower()
                except:
                    country = random.choice(countrys_available)
                    institution = country

                try:
                    topics = list(filter(lambda x: x.strip() in topics_available,
                                         author_info["topics"]))
                    topics = [x.strip() for x in topics]
                    assert len(topics) > 0
                except:
                    topics = [random.choice(topics_available)]

                self.author_data[str(index)] ={
                    "name":author_info["name"],
                    "articles":[],
                    'institution':institution,
                    'expertises':author_info["expertises"],
                    'topics':topics,
                    "publications":0,
                    "cited":0,
                    "co_author_ids":[],
                    "country":country,
                    "time":cur_time
                }
                add_indexs.append(str(index))
                index += 1
        except Exception as e:
            print(e, "update_author_db")
        return add_indexs
        
    def get_expertises(self):
        """get all the expertises in the author DB"""
        expertises = []
        for author_info in self.author_data.values():
            expertises.extend(author_info["expertises"])
        return list(set(expertises))
    
    def write_generated_articles(self, articles:list =[]):
        
        root = self.generated_article_dir
        if not os.path.exists(root):
            os.makedirs(root)
        max_idx = 0
        num_article = 0
        import re
        appendix = re.search(r"(.*)_(\d+).txt", os.listdir(self.article_dir)[0]).group(1)
        for config_path in os.listdir(root):
            regex = r"_(\d+).txt"
            try:
                idx = int(re.search(regex,config_path).group(1))
                max_idx = idx if max_idx<idx else max_idx
            except:
                continue
            
        for config_path in os.listdir(self.article_dir):
            regex = r"_(\d+).txt"
            import re
            try:
                idx = int(re.search(regex,config_path).group(1))
                max_idx = idx if max_idx<idx else max_idx
            except:
                continue
            
        for idx,article in enumerate(articles):
            if not article["success"]:
                continue
            title = article["title"]
            abstract = article["abstract"]
            del article["abstract"]
            path_new_article = os.path.join(root,f"{appendix}_{max_idx+idx+1}.txt")
            article.update({
                "path": path_new_article,
                "cited":0,
                "cited_articles":[]
            })
            if title not in self.article_meta_data.keys():
                assert not os.path.exists(path_new_article),f"{path_new_article} exists!!"
                with open(path_new_article,"w") as f:
                    f.write(title + " "+ abstract)
               
                
                co_authors_article = article["author_ids"]
                
                for author_id in article["author_ids"]:
                    assert author_id in self.author_data.keys(),f"unknown author {author_id}"

                    self.author_data[author_id]["articles"].append(title)
                    self.author_data[author_id]["publications"] +=1
                    co_authors = self.author_data[author_id].get("co_author_ids",[])
                    for co_author_a in co_authors_article:
                        if co_author_a != author_id and \
                            co_author_a not in co_authors:
                                co_authors.append(author_id)
                    self.author_data[author_id]["co_author_ids"] = list(set(co_authors))
                
                    
                """update citations"""
                for cite_article in article["citations"]:
                    if cite_article not in self.article_meta_data.keys():
                        continue
                    self.article_meta_data[cite_article]["cited"] +=1
                    self.article_meta_data[cite_article]["cited_articles"].append(title)
                    for author_id in self.article_meta_data[cite_article]["author_ids"]:
                        if author_id in self.author_data.keys():
                            self.author_data[author_id]["cited"] +=1
                try:
                    del article["all_citation"]
                except:pass
                article["time"] = parse_date_str(article["time"])

                self.article_meta_data[title] = article
                
                num_article +=1
                
            else:
                """update publication"""
                print(f"generated replicate article!! \n {title}")
                continue

        meta_article_path = os.path.join(os.path.dirname(self.generated_article_dir),
                                                 "article_meta_info.pt")     
        searlized_article_meta_data = copy.deepcopy(self.article_meta_data)
        
        for title,article_info in searlized_article_meta_data.items():
            if isinstance(article_info["time"],datetime) or isinstance(article_info["time"],date):
                searlized_article_meta_data[title]["time"] = article_info["time"].strftime("%Y-%m")
            elif isinstance(article_info["time"],str):
                searlized_article_meta_data[title]["time"] = article_info["time"]
            else:
                raise Exception(f"unknown type {type(article_info['time'])}")
            
        writeinfo(meta_article_path, searlized_article_meta_data)
        
        searlized_author_data = copy.deepcopy(self.author_data)
        
        for author_id,author_info in searlized_author_data.items():
            if isinstance(author_info["time"],datetime) or isinstance(author_info["time"],date):
                author_info["time"] = author_info["time"].strftime("%Y-%m")
            elif isinstance(author_info["time"],str):
                author_info["time"] = author_info["time"]
            else:
                raise Exception(f"unknown type {type(author_info['time'])}")
            searlized_author_data[author_id] = {
                k:author_info[k] for k in ["co_author_ids",
                                            "cited",
                                            "publications",
                                            "expertises",
                                    "articles",
                                    "institution", 
                                    "name",
                                    "time",
                                    "country",
                                    "topics"]
            }
            
        
        author_path = os.path.join(os.path.dirname(self.generated_article_dir),
                                            "author.pt")
        writeinfo(author_path, searlized_author_data)
        
        return num_article
        
    def write_and_update_db(self,
                            articles:list =[]):
        num = self.write_generated_articles(articles=articles)
        self.update_db()
        self.update_big_name_list()
        self.article_written_num += num

        
    def get_article_written_num(self):
        return self.article_written_num
     
    def get_author_description(self,
                             agent_name):
        try:
            infos = self.author_data[agent_name]

            template="""\
You are a researcher. Your name is {name}. Your research interest is {expertises}.
You often write about articles about these topics: {topics}.
\
"""         
            if isinstance(infos["expertises"],str):
                expertises = infos["expertises"][:100]
            else:
                expertises = ",".join(list(infos.get("expertises",[])))[:100]
            role_description = template.format(
                expertises = expertises,
                name = infos["name"],
                topics = ",".join(list(infos.get("topics",[])))
            )
            return role_description
        except Exception as e:
            print(e)
            return ""
        
    
        
    def filter_citations(self,
                         citations: Union[str,list]) -> List[str]:
        if isinstance(citations,str):
            citations_articles = citations.split("\n")
        else:
            citations_articles = citations
        citation_names = []
        for citation_article in citations_articles:
            if citation_article.strip() == "":
                continue
            for doc_candidate in self.article_meta_data.keys():
                if doc_candidate.lower() in citation_article.lower():
                    citation_names.append(doc_candidate)
                    break
                
        return citation_names
    
    def get_topic_agents(self,topic) -> list:
        
        return self.topic_agents.get(topic,[])
    
    def get_author(self,author_id):
        author_info = self.author_data[author_id]
        keys = ["name",
                "institution",
                "expertises",
                "publications",
                "cited",
                "topics"]
        infos = {k:author_info[k] for k in keys}
        for k,v in infos.items():
            if isinstance(v,str):
                infos[k] = v[:200]
            elif isinstance(v,list):
                infos[k] = v[:10]
        return infos

    def get_most_cooperated_author(self,
                                   topic:str,
                                   author_num:int = 5):
        authors_topic = self.get_topic_agents(topic)
        """这里可以做一个llm版本的"""
        author_dict_filtered = {author_id:self.author_data[author_id] for author_id in 
                                self.author_pointer_args["cur_author_ids"]}

        if len(authors_topic) == 0:
            first_author = random.choice(list(author_dict_filtered.keys()))
        else:
            authors_topic = list(filter(lambda x:x in author_dict_filtered.keys(), authors_topic))
            first_author = random.choice(authors_topic)
        authors = [first_author]
        
        ## bfs 
        queue_authors = [first_author]
        while len(authors) < author_num and len(queue_authors) > 0:
            author_name = queue_authors.pop()
            co_author_ids = author_dict_filtered[author_name].get("co_author_ids",[])
            """每次bfs 优先度为co_author -> tags """
            for co_author in co_author_ids:
                if co_author not in authors and \
                    co_author in author_dict_filtered.keys():
                    authors.append(co_author)
                    queue_authors.append(co_author)

        if len(authors) < author_num:
            tags = ["topics", 
                    "expertises",
                    "cited",
                    "institution",
                    "country"]
            for tag in tags:
                same_tag_author_ids = [author_id for author_id,author_info in author_dict_filtered.items() 
                                       if author_info.get(tag,None) == author_dict_filtered[first_author].get(tag)]
                
                for same_tag_author in same_tag_author_ids:
                    if same_tag_author not in authors:
                        authors.append(same_tag_author)
                    if len(authors) >= author_num:
                        return authors
        if len(authors) < author_num:
            num_add = author_num - len(authors)
            added_authors = random.sample(list(author_dict_filtered.keys()),num_add)
            for add_author in added_authors:
                if add_author not in authors:
                    authors.append(add_author)
        return authors
    

    def get_llm_author(self,
                       llm,
                        topic:str,
                        author_num:int = 5):
        
        authors_topic = self.get_topic_agents(topic)
        
        author_dict_filtered = {author_id:self.author_data[author_id] for author_id in 
                                self.author_pointer_args["cur_author_ids"]}
        authors_topic = list(filter(lambda x:x in author_dict_filtered.keys(), authors_topic))
        if len(authors_topic) == 0:
            first_author = random.choice(list(author_dict_filtered.keys()))
        else:
            first_author = random.choice(authors_topic)
        if author_num == 1: 
            return [first_author]
        
        threshold = 30
        ## bfs 
        queue_authors = [first_author]
        authors = [first_author]
        while len(authors) < threshold and len(queue_authors) > 0:
            author_name = queue_authors.pop()
            try:
                co_author_ids = author_dict_filtered[author_name].get("co_author_ids",[])
                """每次bfs 优先度为co_author -> tags """
                for co_author in co_author_ids:
                    if co_author not in authors and \
                        co_author in author_dict_filtered.keys():
                        authors.append(co_author)
                        queue_authors.append(co_author)
            except:
                continue

        if len(authors) < threshold:
            if "no_author_country" in self.experiment:
                tags = ["topics", 
                    "expertises",
                    "cited",]
            else:
                tags = ["topics", 
                    "expertises",
                    "cited",
                    "institution",
                    "country"]
            for tag in tags:
                same_tag_author_ids = [author_id for author_id,author_info in author_dict_filtered.items() 
                                       if author_info.get(tag,None) == author_dict_filtered[first_author].get(tag)]
                
                for same_tag_author in same_tag_author_ids:
                    if same_tag_author not in authors:
                        authors.append(same_tag_author)
                    if len(authors) >= threshold:
                        break

        template ="""
Here's some of the authors in topic:{topic}

The author appears at the begining of the list would be the first author.
I want you to select some authors from this candidate list to complete the next paper:

{author_infos}

Now respond the authors index, a list of integers, return your answer in this format:
Thought:(your reason for selecting these authors)
Authors:(a list, like([1,2,3],the author index you selected)

Think step by step before you act.
Now respond:
"""
        if len(authors) > threshold:
            authors = random.sample(authors, threshold)
        
        random.shuffle(authors)
        if "no_author_cite" in self.experiment:
            author_template = """
{idx}. {name}, country:{country},. often write about articles: {topics}.
"""     
        elif "no_author_topic" in self.experiment:
            author_template = """
{idx}. {name}, country:{country}, citations: {cited}.
"""     
        elif "no_author_country" in self.experiment:
            author_template = """
{idx}. {name}, citations: {cited}. often write about articles: {topics}.
"""
        else:
            author_template = """
    {idx}. {name}, country:{country}, citations: {cited}. often write about articles: {topics}.
    """
        prompt = template.format(topic=topic,
                               author_infos="\n".join(
                                   [author_template.format_map({"idx":idx,
                                    **self.get_author(author_id)}
                                       ) 
                                    for idx, author_id in enumerate(authors)])
                                    )
        prompt_msg = llm.format(Msg("user",prompt,"user"))
        response = llm(prompt_msg)
        content = response.text
        try:
            candidate_list = find_and_load_json(content,"list")
            authors_r = [authors[candidate] for candidate in candidate_list]
        except:
            authors_r = random.sample(authors, author_num)

        return authors_r
       
    def save_infos(self, 
                   cur_time,
                   start_time,
                   article_write_configs
                   ):
        simulation_time = time.time() - start_time
        simulation_time += self.simulation_time
        generated_article_num = len(os.listdir(self.generated_article_dir))
        log_info = {"cur_time":datetime.strftime(cur_time, '%Y-%m-%d'),
                    "last_added_time": datetime.strftime(self.author_pointer_args["cur_time"], 
                                                         '%Y-%m-%d'),
                    "generated_articles": generated_article_num,
                    "simulation_time":int(simulation_time),
                    "cur_author_num":len(self.author_pointer_args["cur_author_ids"]),
                    "cur_author_ids":self.author_pointer_args["cur_author_ids"],
                    "overloaded_ids":self.author_pointer_args["overloaded_ids"],
                    "overloaded_threshold":self.author_pointer_args["overloaded_threshold"],
                    "article_write_configs":article_write_configs
                    }
        
        from LLMGraph.utils.io import writeinfo
        log_info_path = os.path.join(os.path.dirname(self.generated_article_dir),
                                    "log_info.json")
        writeinfo(log_info_path, log_info)


    def save_networkx_graph(self):
        # 创建一个空的有向图
        DG = nx.DiGraph()
        model_root = os.path.join(os.path.dirname(self.generated_article_dir),\
            "model")
        
        map_index = {
            title:idx for idx,title in enumerate(self.article_meta_data.keys())}
        
        for title in self.article_meta_data.keys():
            cited_idx = map_index.get(title)
            time = self.article_meta_data[title]["time"].strftime("%Y-%m")
            DG.add_node(cited_idx,title=title,time=time)
        
        for title, article_info in self.article_meta_data.items():
            cited_articles = article_info.get("cited_articles",[])
            title_idx = map_index.get(title)
            
            edges = []
            for cite_title in cited_articles:
                cited_idx = map_index.get(cite_title)
                if cited_idx is not None:
                    edges.append((cited_idx,title_idx))            
            DG.add_edges_from(edges)
                    
        model_path = os.path.join(model_root,"graph.graphml")
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        nx.write_graphml(DG, model_path)
    
    def get_article_infos(self,
                          titles:list =[],
                          ):
        return self.article_loader.get_article_docs(titles,
                                    self.article_meta_data,
                                    self.author_data)

    def plot_article_lda(self):
        from evaluate.visualize.article import plt_topic_png

        docs = self.article_loader.load()

        documents = [doc.page_content for doc in docs]

        save_dir = os.path.join(self.generated_article_dir,"visualize")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir,f"article_kmeans_{len(docs)}.pdf")

        topic_num = int(np.sqrt(len(docs)))
        topic_num = topic_num if topic_num >5 else 5
        n_clusters = int(np.sqrt(topic_num))
        n_clusters = n_clusters if n_clusters >3 else 3

        plt_topic_png(save_path,
                      documents,
                      topic_num,
                      n_clusters
                      )
        
    def get_topics(self):
        
        return list(self.topic_agents.keys())

    def update_big_name_list(self):
        """get big name list: top hub_rate authors"""
        threshold_ratio = self.control_profile.get("hub_rate",0.2)
        cur_agent_ids = self.author_pointer_args["cur_author_ids"]

        threshold = int(len(cur_agent_ids)*threshold_ratio)
        cur_agent_ids = sorted(cur_agent_ids,
                               key = lambda x: self.author_data[x]["cited"],
                               reverse=True)
        
        self.big_name_list = cur_agent_ids[:threshold]

    def get_topics_available(self):
        if "llm_agent" in self.dataset:
            return [
                "Artificial Intelligence",
                "Machine Learning",
                "Computational Sciences",
                "Social and Cognitive Sciences",
                "Software and Systems",
                "Emerging Technologies"
            ]
        elif "citeseer" in self.dataset:
            return ["Agents","AI","DB","IR","ML","HCI"]
        elif "cora" in self.dataset:
            return ["Case_Based",
                    "Genetic_Algorithms",
                    "Neural_Networks",
                    "Probabilistic_Methods",
                    "Reinforcement_Learning",
                    "Rule_Learning",
                    "Theory"]
        else:
            return []
        
    def save_encoded_features(self,
                              article_num = None,
                              embeddings = "default"
                              ):
        
        
        # embeddings_list = []
        # from sentence_transformers import SentenceTransformer as SBert
        # model = SBert("albert-base-v1")   # 模型大小1.31G
        # assert len(self.article_loader.docs) == len(self.article_meta_data), "not available article number"
        doc_str_list = []
        if article_num is not None:
            docs = self.article_loader.docs[:article_num]
        else:
            docs = self.article_loader.docs
            article_num = len(self.article_meta_data)

        for doc in docs:
            doc_str = self.article_loader.format_document(
                                                          self.article_meta_data,
                                                          self.author_data,
                                                          doc)
            doc_str_list.append(doc_str)
            # embedding = model.encode(doc_str)
        if embeddings == "default":
            embeddings_list = self.embeddings.embed_documents(doc_str_list)
            # 转换为numpy数组
            embeddings_array = np.vstack(embeddings_list)
            save_npy_root = os.path.join(os.path.dirname(self.generated_article_dir),\
            "feature","default",f"{article_num}")
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            # 创建一个CountVectorizer对象
            vectorizer = CountVectorizer()
            # 将文本数据转换成词频矩阵
            embeddings_array = vectorizer.fit_transform(doc_str_list)
            save_npy_root = os.path.join(os.path.dirname(self.generated_article_dir),\
            "feature","BoW",f"{article_num}")
    
        
        os.makedirs(save_npy_root, exist_ok=True)
        assert embeddings_array.shape[0] == article_num
        np.save(os.path.join(save_npy_root, "embeddings.npy"), 
                    embeddings_array)
        
        meta_article_path = os.path.join(save_npy_root, "article_meta_info.pt")     
        searlized_article_meta_data = copy.deepcopy(self.article_meta_data)
        searlized_article_meta_data = dict(list(searlized_article_meta_data.items())[:article_num])

        for title, article_info in searlized_article_meta_data.items():
            if isinstance(article_info["time"],datetime) or isinstance(article_info["time"],date):
                searlized_article_meta_data[title]["time"] = article_info["time"].strftime("%Y-%m")
            elif isinstance(article_info["time"],str):
                searlized_article_meta_data[title]["time"] = article_info["time"]
            else:
                raise Exception(f"unknown type {type(article_info['time'])}")
            
        assert len(searlized_article_meta_data) == article_num
        writeinfo(meta_article_path, searlized_article_meta_data)
        