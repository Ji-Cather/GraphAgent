import json
import os

from . import manager_registry as ManagerRgistry
from typing import List,Union,Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from LLMGraph.retriever import retriever_registry
import random
import networkx as nx
from datetime import datetime,date,timedelta
import copy
from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from LLMGraph.tool import create_movie_retriever_tool
from LLMGraph.loader import Movie1MDatLoader
from LLMGraph.manager.base import BaseManager
import numpy as np

import time
from LLMGraph.utils.process_time import transfer_time
from LLMGraph.tool import create_movie_retriever_tool,create_get_movie_html_tool

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list


def process_array_time(array):
    """处理最后一列的timestamp"""
    for idx,row in enumerate(array):
        timestamp = row[-1]
        time = transfer_time(timestamp)
        assert isinstance(time,date)
        array[idx][-1] = time
    return array

def pre_process(movie_array:np.ndarray,
                ratings:np.ndarray,
                users:np.ndarray,
                start_time,
                cur_time,
                filter_initial_train = True):
    
    """返回按照时间排序的movie,以及切分到当前时间步的ratings矩阵"""
    
    movie_array = process_array_time(movie_array)
    users = process_array_time(users)
    ratings = process_array_time(ratings)
    
    sorted_indices = np.argsort(movie_array[:, -1])
    movies = movie_array[sorted_indices]
    
    sorted_indices = np.argsort(users[:, -1])
    users = users[sorted_indices]
    
    ratings = ratings[np.isin(ratings[:,0], users[:,0])] 
    ratings = ratings[np.isin(ratings[:,1], movies[:,0])] 
    
    if filter_initial_train:
        ratings = ratings[ratings[:,-1] <= cur_time]
        movies = movies[movies[:,-1] >= start_time]
        users = users[users[:,-1] >= start_time]
    return movies, ratings, users


@ManagerRgistry.register("movie")
class MovieManager(BaseManager):
    """,
        manage infos of movie db
    """,
    
    link_movie_path:str
    movie_data_dir:str
    simulation_time:int = 0
    
    html_tool_kwargs:dict ={
        "upper_token": 1000,
        "url_keys": ["imdbId_url",
                    "tmdbId_url"],
        "llm_kdb_summary": True,
        "llm_url_summary": False,
    }
    
    ratings_log:list = []
    
    movie_loader: Movie1MDatLoader
    
    db: Any = None # 存储历史电影 DB为None时表示没有可看的电影
    db_cur_movies: Any = None # 存储正在热映的电影
    retriever:Any = None
    retriever_cur:Any = None

    watcher_data: np.ndarray
    ratings_data: np.ndarray
    movie_scores: dict = {} # movie_id: score
    
    # llm: OpenAI
    embeddings: Any
    
    generated_data_dir: str
    
    
    retriever_kwargs_last:dict = {}

    watcher_pointer_args:dict = {
        "cnt_p": -1,
        "cnt_watcher_ids":[]
    }
    
    age_map:dict = {
            1:  "Under 18",
            18:  "18-24",
            25:  "25-34",
            35:  "35-44",
            45:  "45-49",
            50:  "50-55",
            56:  "56+"}
        
    occupation_map:dict = {
    0:  "other",
	1:  "academic/educator",
	2:  "artist",
	3:  "clerical/admin",
	4:  "college/grad student",
	5:  "customer service",
	6:  "doctor/health care",
	7:  "executive/managerial",
	8:  "farmer",
	9:  "homemaker",
	10:  "K-12 student",
	11:  "lawyer",
	12:  "programmer",
	13:  "retired",
	14:  "sales/marketing",
	15:  "scientist",
	16:  "self-employed",
	17:  "technician/engineer",
	18:  "tradesman/craftsman",
	19:  "unemployed",
	20:  "writer"}
    
    

    class Config:
        arbitrary_types_allowed = True
    
    
    
    @classmethod
    def load_data(cls,
                  movie_data_dir,
                  retriever_kwargs,
                  html_tool_kwargs,
                  ratings_data_name,
                  generated_data_dir, # store all the generated movies, watcher and rating infos
                  cur_time: datetime,
                  start_time: datetime,
                  movie_time_delta: timedelta,
                  tool_kwargs,
                  control_profile,
                  link_movie_path:str = "LLMGraph/tasks/movielens/data/ml-25m/links.csv"
                  ):
        
        if os.path.exists(os.path.join(generated_data_dir,"data")):
            movie_path = os.path.join(movie_data_dir,"movies.npy") 
            ratings_path = os.path.join(generated_data_dir,"data","ratings.npy")
            users_path = os.path.join(generated_data_dir,"data","users.npy")
            ratings_log_path = os.path.join(generated_data_dir,"data","ratings_log.npy")
            ratings_log = np.load(ratings_log_path, allow_pickle=True).tolist()
            filter_initial_train = False
            
        else:
            movie_path = os.path.join(movie_data_dir,"movies.npy") 
            ratings_path = os.path.join(movie_data_dir,ratings_data_name) 
            users_path = os.path.join(movie_data_dir,"users.npy")
            ratings_log = []
            filter_initial_train = True

        
           
        movies = np.load(movie_path,allow_pickle=True)
        ratings = np.load(ratings_path,allow_pickle=True)
        watcher_data = np.load(users_path,allow_pickle=True)
        
        movies, ratings, watcher_data = pre_process(movies,
                                            ratings,
                                            watcher_data,
                                            start_time,
                                            cur_time,
                                            filter_initial_train)
            
        movie_loader = Movie1MDatLoader(movie_data_array = movies,
                        link_movie_path = link_movie_path,
                        cur_time=cur_time,
                        load_movie_html=False)
        
        movie_loader.init(cur_time,
                          movie_time_delta)

        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # embeddings = HuggingFaceEmbeddings(model_name ="thenlper/gte-small")
        db, db_cur_movies = None, None
        if len(movie_loader.docs)>0:
            db = FAISS.from_documents(movie_loader.docs, 
                                       embeddings)
        else:
            raise Exception("empty online movie DB!")
        if len(movie_loader.cur_movie_docs)>0:
            db_cur_movies = FAISS.from_documents(movie_loader.cur_movie_docs, 
                                                  embeddings)

        watcher_pointer_args ={
            "cnt_p":0,
            "cnt_watcher_ids":[]
        }
        
        movie_scores = {}
        
        for movie_id in movie_loader.movie_data_array[:,0]:
            ratings_sub = ratings[ratings[:, 1] == int(movie_id)]
            assert isinstance(ratings_sub,np.ndarray)
            if ratings_sub[:,2].sum() == 0:
                movie_scores[movie_id] = 0
            else:
                movie_scores[movie_id] = ratings_sub[:,2].mean()
            
        
        return cls(
            link_movie_path = link_movie_path,
            movie_data_dir = movie_data_dir,
            embeddings = embeddings,
            watcher_pointer_args = watcher_pointer_args,
            watcher_data = watcher_data,
            ratings_data = ratings,
            movie_loader = movie_loader,
            db = db,
            db_cur_movies = db_cur_movies,
            retriever_kwargs = retriever_kwargs,
            generated_data_dir = generated_data_dir,
            movie_scores = movie_scores,
            html_tool_kwargs = html_tool_kwargs,
            ratings_log = ratings_log,
            tool_kwargs = tool_kwargs,
           control_profile = control_profile
        )
        
    def load_history(self):
        log_info_path = os.path.join(self.generated_data_dir,
                                     "data",
                                     "log_info.json")
        content = None
        if os.path.exists(log_info_path):
            log_info = readinfo(log_info_path)
            cur_time = log_info.get("cur_time")
            cur_time = datetime.strptime(cur_time,'%Y-%m-%d').date()
            self.simulation_time = log_info.get("simulation_time",0)
            # self.watcher_pointer_args["cnt_p"] = log_info.get("cur_watcher_num",0)
            content = {"cur_time":cur_time.strftime('%Y-%m-%d'), 
                       "cur_rate":len(self.ratings_log)}
        
        return content
       
    def update_movie_scores(self,
                            movie_scores:dict = {} # movie_id: avg(all rating movie_score)
                            ):
        self.movie_scores.update(movie_scores)
        
        
    def get_movie_retriever(self,
                            movie_scores = {},
                            online = True,
                           **retriever_kargs_update):
        """offline的时候不进行filter"""
        
        self.update_movie_scores(movie_scores)
        retriever_args = copy.deepcopy(self.retriever_kwargs)
        retriever_args.update(retriever_kargs_update)

        if online: 
            retriever_args["vectorstore"] = self.db
            retriever_args["compare_function_type"] = "movie"
            retriever_args["movie_scores"] = self.movie_scores
            self.retriever = retriever_registry.from_db(**retriever_args)
        else:
            retriever_args["vectorstore"] = self.db_cur_movies    
            retriever_args["compare_function_type"] = "movie"
            retriever_args["movie_scores"] = self.movie_scores
            self.retriever_cur = retriever_registry.from_db(**retriever_args)    

       
    
    
    def get_movie_retriever_tool(self,
                                 online = True,
                                interested_genres:list = [],
                                watched_movie_ids:list = [],
                                **retriever_kargs_update):
        self.get_movie_retriever(online = online,
                                **retriever_kargs_update)
        
        if online: 
            retriever = self.retriever
        else:
            retriever = self.retriever_cur

        if retriever is None:return
        
        document_prompt = PromptTemplate.from_template("""
Title: {Title}
Genres: {Genres}
Content: {page_content}""")
        tool_func,func_dict = create_movie_retriever_tool(
            retriever,
            "SearchMovie",
            "Search for movies, you should provide some keywords for the movie you want to watch (like genres, plots and directors...)",
            document_prompt = document_prompt,
            filter_keys=self.tool_kwargs["filter_keys"],
            interested_genres=interested_genres,
            watched_movie_ids=watched_movie_ids)

        return tool_func, func_dict
    
    def get_movie_html_tool(self,
                            online = True,
                            **retriever_kargs_update):
        self.get_movie_retriever(online=online,
                                             **retriever_kargs_update)
        
        if online: 
            retriever = self.retriever
        else:
            retriever = self.retriever_cur

        if retriever is None: return
        movie_html_tool = create_get_movie_html_tool(
            retriever = retriever,
            movie_scores = self.movie_scores,
            name = "GetOneMovie",
            description="You can get the movie html information of one movie you want to watch using this tool.\
[!Important!] You should always give your rating after using this tool!! you should give one movie name",
            **self.html_tool_kwargs
            )
        return movie_html_tool
    
    
    
    def get_cur_movie_docs_len(self):
        return len(self.movie_loader.cur_movie_docs)
    
    def add_movies(self,
                  cur_time:str,
                  movie_time_delta:int):
        """update movie DB"""
        cur_time = datetime.strptime(cur_time,'%Y-%m-%d').date()
        movie_time_delta = timedelta(days=movie_time_delta)

        if self.no_available_movies():
            return
        
        if (cur_time - self.movie_loader.cur_time) < movie_time_delta:
            return
        
        self.movie_loader.update(cur_time)
        
        if len(self.movie_loader.docs)>0:
            self.db = FAISS.from_documents(self.movie_loader.docs, 
                                       self.embeddings)
        else:
            self.db = None
        if len(self.movie_loader.cur_movie_docs)>0:
            self.db_cur_movies = FAISS.from_documents(self.movie_loader.cur_movie_docs, 
                                                  self.embeddings)
        else:
            self.db_cur_movies = None

        
    def filter_rating_movie(self,
                            movie_rating:dict = {},
                            online = True):
        
        try:
            movie_title = movie_rating["movie"]
            if self.retriever is not None:
                online_docs = self.retriever.invoke(movie_title)
            else:online_docs = []
            if self.retriever_cur is not None:
                offline_docs = self.retriever_cur.invoke(movie_title)
            else: offline_docs = []
            retrived_docs = [*online_docs,*offline_docs]

            for movie_doc in retrived_docs:
                if movie_title.strip().lower() in movie_doc.metadata["Title"].lower():
                    movie_rating.update({
                        "movie": movie_doc.metadata["Title"],
                        "movie_id": movie_doc.metadata["MovieId"],
                        "genres": movie_doc.metadata["Genres"],  
                    })
                    return movie_rating
        except Exception as e:
            pass  
        return {}
        
    
    def add_watcher(self,
                    cur_time,
                    watcher_num:int =-1,
                    watcher_add:bool = False):
        if not watcher_add:
            if self.watcher_pointer_args["cnt_watcher_ids"] ==[]:
                self.watcher_pointer_args["cnt_watcher_ids"] = list(range(len(self.watcher_data[:watcher_num])))
                self.watcher_pointer_args["cnt_p"] = len(self.watcher_data[:watcher_num])
            return
            
        left_p = self.watcher_pointer_args["cnt_p"]
        upper_idx = np.argmax(cur_time <= self.watcher_data[:,-1])
        # if upper_idx ==0:upper_idx = len(self.watcher_data)
        
        self.watcher_pointer_args["cnt_p"] = upper_idx
        self.watcher_pointer_args["cnt_watcher_ids"] = list(range(left_p,upper_idx))
    
    def add_and_return_watcher_profiles(self,
                                        cur_time:str,
                                        watcher_num:int =-1,
                                        watcher_add:bool = False):
        cur_time = datetime.strptime(cur_time,'%Y-%m-%d').date()
        self.add_watcher(cur_time,
                         watcher_num = watcher_num,
                         watcher_add= watcher_add)
        return self.return_cur_watcher_profiles()
        
    def return_cur_watcher_profiles(self):
       
        return [{
                "infos":
                  {
                    "gender": "Female" if self.watcher_data[idx][1]=="F" else "Male",
                    "age": self.age_map.get(self.watcher_data[idx][2]),
                    "job": self.occupation_map.get(self.watcher_data[idx][3])
                },
                "id":int(self.watcher_data[idx][0]),
                } 
                for idx in self.watcher_pointer_args["cnt_watcher_ids"]]
        
    def update_db_ratings(self,
                          ratings,
                          agent_ids:list = []):
        """ 需要update rating 矩阵"""
        ratings_cur_turn = []
        for rating_agent, agent_id in zip(ratings,agent_ids):
            for rating_one_movie in rating_agent:
                timestamp = rating_one_movie["timestamp"]
                timestamp = transfer_time(timestamp)
                ratings_cur_turn.append([int(agent_id),
                                        int(rating_one_movie["movie_id"]),
                                        int(rating_one_movie["rating"]),
                                        timestamp])
                self.ratings_log.append(rating_one_movie)
        if len(ratings_cur_turn) ==0: return 0
        ratings_cur_turn = np.asarray(ratings_cur_turn)
        assert self.ratings_data.shape[1] == ratings_cur_turn.shape[1]
        self.ratings_data = np.concatenate([self.ratings_data,ratings_cur_turn])
        ratings_len = ratings_cur_turn.shape[0]
        if ratings_len is None:
            return 0
        else:
            return ratings_len
        
    def get_watcher_rating_infos(self, watcher_id) -> dict: 
        # count 不同movie的观看频率 以及平均打分
        # 现在的做法会time_consuming 仅仅在创建的时候进行调用
        
        arr =  self.ratings_data
        ratings_sub = arr[arr[:, 0] == int(watcher_id)]
        rating_count = {}
        movie_array = self.movie_loader.movie_data_array
        assert isinstance(ratings_sub,np.ndarray)
        for rating in ratings_sub:
            movie_id = rating[1]
            try:
                movie_info = movie_array[movie_array[:,0] == movie_id][0] 
            except: continue
            
            rating_time = rating[3]
            if isinstance(rating_time,date):
                rating_time = rating_time.strftime("%Y-%m-%d")
            rating_info = [(movie_id,rating[2],rating_time)]
            
            genres = movie_info[2].split("|")
            for genre in genres:
                if genre not in rating_count:
                    
                    rating_count[genre] = rating_info
                else: 
                    rating_count[genre].extend(rating_info)

        return rating_count
   
    
    def get_movie_description(self):
        return self.movie_loader.get_movie_description()
    
    def get_movie_types(self):
        return self.movie_loader.get_movie_types()
    
    
    def get_watcher_infos(self,
                          watcher_id,
                          first_person = True):
        if isinstance(watcher_id,str):
            watcher_id = int(watcher_id)
        if first_person:
            prompt_template = """,
I'm a viewer. \
I'm {gender}.
I'm {age} years old, my job is {job}.\
Now I want to watch a movie and give a rating score of the movie I get.\
My task is to give a rating score. """
        else:
            prompt_template = """
You're a viewer. You're {gender}. You're {age} years old, your job is {job}. \
Now you want to watch a movie and give a rating score of the movie you get. \
Your task is to give a rating score. """
        watcher_idx = np.where(self.watcher_data[:,0] == watcher_id)[0][0]
        infos = self.watcher_data[watcher_idx,:]

        age_info = infos[2]
        if self.age_map.get(infos[2]) is not None:
            age_info = self.age_map.get(infos[2])
        infos = self.watcher_data[watcher_idx]
        
        id_oc = infos[3]
        if isinstance(id_oc,int):
            occupation = self.occupation_map.get(id_oc)
        else: occupation = id_oc
        
        infos_dict = {
            "gender": "Female" if infos[1]=="F" else "Male",
            "age": age_info,
            "job": occupation
        }

        return prompt_template.format_map(infos_dict)
    
    
    def save_infos(self,
                   cur_time,
                   start_time):
        data_dir = os.path.join(self.generated_data_dir,"data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        np.save(os.path.join(data_dir,"users.npy"), self.watcher_data)
        np.save(os.path.join(data_dir,"ratings.npy"), self.ratings_data)
        np.save(os.path.join(data_dir,"ratings_log.npy"), self.ratings_log)
        # np.save(os.path.join(data_dir,"movies.npy"), self.movie_loader.movie_data_array)
        simulation_time = time.time() - start_time
        simulation_time += self.simulation_time
        log_info = {"cur_time":datetime.strftime(cur_time, '%Y-%m-%d'),
                    "generated_ratings": len(self.ratings_log),
                    "simulation_time":int(simulation_time),
                    "cur_watcher_num":int(self.watcher_pointer_args["cnt_p"]),
                    "cur_movie_num":int(self.movie_loader.data_ptr)}
        
        writeinfo(os.path.join(data_dir,"log_info.json"), log_info)
        

    
    
    def save_networkx_graph(self):
        model_dir = os.path.join(os.path.dirname(self.generated_data_dir),\
            "model")
        
        # 创建一个空的二部图
        B = nx.Graph()

        # 添加节点，节点可以有属性。这里我们用节点属性'bipartite'标记属于哪个集合
        for watcher_idx in self.watcher_pointer_args["cnt_watcher_ids"]:
            watcher_info = self.watcher_data[watcher_idx]
            watcher_id = watcher_info[0]
            id_oc = watcher_info[3]
            if isinstance(id_oc,int):
                occupation = self.occupation_map.get(id_oc)
            else: occupation = id_oc
            B.add_node(f"watcher_{watcher_id}", 
                       bipartite=0, 
                       gender= "Female" if watcher_info[1]=="F" else "Male",
                       age = watcher_info[2],
                       occupation = occupation
                       )
        docs = self.movie_loader.docs
        cur_docs = self.movie_loader.cur_movie_docs
        docs_all =[*docs, *cur_docs]
        for doc in docs_all:
            movie_id = doc.metadata["MovieId"]
            B.add_node(f"movie_{movie_id}", 
                       bipartite=1,
                       title = doc.metadata["Title"],
                       genres = ", ".join(doc.metadata["Genres"]),
                       timestamp = doc.metadata["Timestamp"].strftime("%Y%m%d")
                       )    
        from tqdm import tqdm
        for row in tqdm(self.ratings_data):
            timestamp = row[3]
            if isinstance(timestamp,int):
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
            if isinstance(timestamp,datetime):
                timestamp_str = timestamp.strftime("%Y%m%d")
            elif isinstance(timestamp,date):
                timestamp_str = timestamp.strftime("%Y%m%d")
            edge_kargs ={
                "rating": row[2],
                "timestamp": timestamp_str
            }
            try:
                assert f"watcher_{int(row[0])}" in B.nodes().keys(), f"watcher_{int(row[0])} not available"
                assert f"movie_{int(row[1])}" in B.nodes().keys(), f"movie_{int(row[1])} not available"
                B.add_edge(f"watcher_{int(row[0])}",
                            f"movie_{int(row[1])}",
                            **edge_kargs)
            except Exception as e:
                continue
        # 添加边，连接集合0和集合1的节点
        # B.add_edges_from([('a', 1), ('b', 2), ('c', 3), ('a', 3), ('a', 4), ('b', 4)])
        model_path = os.path.join(model_dir,"graph.graphml")
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        nx.write_graphml(B, model_path)
    
    def get_movie_rating_score(self,
                               movie_id):
        arr =  self.ratings_data
        ratings_sub = arr[arr[:, 1] == int(movie_id)]
        assert isinstance(ratings_sub,np.ndarray)
        if ratings_sub[:,2].sum() == 0:
            rating_score = 0
        else:
            rating_score = ratings_sub[:,2].mean()
            
        return rating_score
    
    def update_watcher(self,
                       llm):
        prompt ="""
Your task is to give me a list of watcher's profiles. Respond in this format:
[
{
"gender": (F/M)
"age":(the age of the watcher)
"job":(the job of the watcher)
}
]

Now respond:
"""
        response = llm.invoke(prompt)
        content = response.content
        id_prefix = self.watcher_data.shape[0]
        try:
            watcher_data_update = json.loads(content)
            watcher_data_update_array = []
            for idx,watcher in enumerate(watcher_data_update):
                try:
                    gender = watcher["gender"]
                    if gender not in ["F",'M']:
                        gender = random.choice(["F","M"])
                    occupation = watcher["job"]
                    age = int(watcher["age"])   
                    watcher_profile = [id_prefix+idx,
                                    gender,
                                    age,
                                    occupation,
                                    np.nan # No zip:code
                                    ]
                    watcher_data_update_array.append(watcher_profile)
                except:continue
                
            self.watcher_data = np.concatenate([self.watcher_data,
                                                watcher_data_update_array])
        except Exception as e:return
        
    def no_available_movies(self):
        if self.movie_loader.movie_data_array.shape[0] == \
            self.movie_loader.data_ptr:
            return True
        return False
    
    def get_movie_array(self):
        return self.movie_loader.movie_data_array
    
    def get_docs_len(self):
        return len(self.movie_loader.docs)
    
    def get_movie_available_num(self,
                                watched_movie_ids:list = []):
        if self.movie_loader.data_ptr >= self.movie_loader.movie_data_array.shape[0]:
            movie_ids = self.movie_loader.movie_data_array[:,0]
        else:
            movie_ids = self.movie_loader.movie_data_array[
                    :self.movie_loader.data_ptr,0]
        return len(list(filter(lambda movie_id: movie_id not in watched_movie_ids, movie_ids)))
    

    def get_rating_counts(self,
                          rating_counts_id:dict = {}):
        rating_counts = {}
        ratings = {}
        for genre, movie_ratings in rating_counts_id.items():
            rating_counts[genre] = [movie_rating[1] for movie_rating in \
                movie_ratings]
            for movie_rating in movie_ratings:
                movie_id, rating, timestamp = movie_rating
                try:
                    movie_info = self.movie_loader.movie_data_array[
                        self.movie_loader.movie_data_array[:,0] == movie_id]
                    # time = transfer_time(timestamp)
                    ratings[movie_id] = {
                        "movie":movie_info[0][1],
                        "thought": "",
                        "rating": rating,
                        "timestamp": timestamp
                    }
                except:pass

        return {
            "rating_counts":rating_counts,
            "ratings":ratings
        }


    def get_offline_movie_info(self,
                                  filter: dict = {}, 
                                  max_movie_number:int = 20):
        movie_template = """{Title}: {page_content}"""
        cur_movie_doc_len = self.get_cur_movie_docs_len()
        if cur_movie_doc_len == 0:
            return ""
        self.get_movie_retriever(
               online = False
            )
        all_movies = []
        for genre in filter["interested_genres"]:
            sub_docs = self.retriever_cur.invoke(genre)
            if len(sub_docs) > 5:
                sub_docs = sub_docs[:5]
            all_movies.extend(sub_docs)
        

        if len(all_movies) > max_movie_number:
            all_movies = random.sample(all_movies, max_movie_number)

        searched_movie_info = []
        for doc in all_movies:
            searched_movie_info.append(movie_template.format_map({
                "page_content":doc.page_content,
                **doc.metadata
            }))
            
        searched_movie_info = "\n".join(searched_movie_info)
        return searched_movie_info
    
    def get_role_description(self):
        pass