""" load the basic infos of authors""",

import json
import os
from LLMGraph.manager.base import BaseManager
from . import manager_registry as ManagerRgistry
from typing import List, Union, Any, Optional
from langchain_core.prompts import BasePromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import random
import networkx as nx
from LLMGraph.retriever import retriever_registry
from LLMGraph.tool import create_forum_retriever_tool
from datetime import datetime
import copy
from langchain_core.prompts import PromptTemplate
import pandas as pd
from datetime import datetime, date, timedelta
from agentscope.models import ModelWrapperBase
from LLMGraph.loader.social import SocialLoader
from LLMGraph.output_parser.base_parser import find_and_load_json
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name
from collections import Counter

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

@ManagerRgistry.register("social")
class SocialManager(BaseManager):
    """,
        manage infos of social db
    """,
    
    class Config:
        arbitrary_types_allowed = True
    
    social_member_data: pd.DataFrame

    forum_loader: SocialLoader
    action_logs: list = [] # 存储 ac_id, f_id, timestamp, ac_type
    
    embeddings: Any
    generated_data_dir:str
    
    last_added_time:date = date.min
    last_added_index:int = 0
    
    follow_map:dict = {} # 大v:num
    
    start_time: date = None
    simulation_time:int =0
    db:Any = None
    retriever_kargs_update:dict = {}
    retriever:Any = None
    big_name_list:list = []
    
    # debug 
    transitive_agent_log = [] # 记录每一轮的agent增减ids
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def load_data(cls,
                  cur_time,
                  social_data_dir,
                  generated_data_dir,
                  data_name,
                  retriever_kwargs,
                  control_profile,
                  tool_kwargs:dict = {
                        "filter_keys": ["follow", "big_name", "topic"],
                        "hub_connect": False
                    }
                  ):
        social_member_data_path = os.path.join(social_data_dir,f"{data_name}_members.csv")
        data_path = os.path.join(social_data_dir,f"{data_name}.csv")
        forum_df = pd.read_csv(data_path,index_col=None)
        simulation_time = 0
        # load history experiment
        social_member_dir = os.path.join(generated_data_dir,"data","social_network")
        last_added_time = date.min
        if os.path.exists(social_member_dir):
            paths = os.listdir(social_member_dir)
            paths = sorted(paths)
            social_member_path = os.path.join(social_member_dir,
                                              paths[-1])
            cur_time = datetime.strptime(os.path.basename(paths[-1]).split(".")[0][-8:],"%Y%m%d").date()
            social_member_data = pd.read_csv(social_member_path,index_col=None)
            list_cols = ["follow","friend"]
            for list_col in list_cols:
                social_member_data[list_col]  = [json.loads(sub_data) for sub_data in\
                                social_member_data[list_col]]
            action_log_path = os.path.join(generated_data_dir,"data","action_logs.json")
            action_logs = readinfo(action_log_path)
            
            transitive_agent_log_path = os.path.join(generated_data_dir,"data","transitive_agent_log.json")
            transitive_agent_log = readinfo(transitive_agent_log_path)

            for ac_log in action_logs:
                ac_log[-1] = datetime.strptime(ac_log[-1],"%Y-%m-%d").date()
            
            forum_path = os.path.join(generated_data_dir,"data","forum.json")
            forum_loader = SocialLoader(data_path=forum_path)
            ex_logs_path = os.path.join(generated_data_dir,"data","ex_logs.json")
            ex_logs = readinfo(ex_logs_path)
            # last_added_time = datetime.strptime(ex_logs["last_added_time"],"%Y%m%d").date()
            simulation_time = ex_logs.get("simulation_time",0)
        else:
            # forum_df['user_index'] = forum_df.index
            # social_member_data = forum_df[['user_index',
            #                             "user_name",
            #                             "user_description",
            #                             "user_followers"]].drop_duplicates(
            #                                 subset="user_name", keep='first')
            # social_member_data["follow"] = [[] for i in range(social_member_data.shape[0])]
            # # 这个部分可能要增加一个action，如果 不是大v，理论上来说有人关注你很大概率是会互关的
            # social_member_data["friend"] = [[] for i in range(social_member_data.shape[0])]
            social_member_data  = pd.read_csv(social_member_data_path,index_col = None)
            social_member_data["follow"] = [eval(follow_list) for follow_list in social_member_data["follow"]]
            social_member_data["friend"] = [eval(friend_list) for friend_list in social_member_data["friend"]]
            forum_loader = SocialLoader(social_data=forum_df)
            action_logs = []
            transitive_agent_log = []
        
        follow_map = {}
        for user_index in social_member_data['user_index']:
            follow_map[user_index] = len(social_member_data.loc[user_index,"follow"]) + \
                len(social_member_data.loc[user_index,"friend"])
        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",cache_folder="LLMGraph/utils/model")
        #embeddings = HuggingFaceEmbeddings(model_name ="thenlper/gte-small",cache_folder="LLMGraph/utils/model")
  
        
        return cls(
            forum_loader = forum_loader,
            social_member_data = social_member_data,
            retriever_kwargs =retriever_kwargs,
            embeddings = embeddings,
            generated_data_dir=generated_data_dir,
            action_logs = action_logs,
            start_time = cur_time,
            last_added_time = last_added_time,
            simulation_time = simulation_time,
            follow_map = follow_map,
            transitive_agent_log = transitive_agent_log,
            tool_kwargs = tool_kwargs,
            control_profile = control_profile
        )
        
    def get_start_time(self):
        date = datetime.strftime(self.start_time,"%Y-%m-%d")
        return date

    
    def get_user_role_description(self,user_index):
        if isinstance(user_index,str):
            user_index = int(user_index)
        
        template ="""
You are a visitor on Twitter. You are {user_description}. 
You have {user_followers} followers.

{follower_description}
"""      
        follower_des = self.get_follower_description(user_index)
        try:
            infos = self.social_member_data.loc[user_index,:].to_dict()
            infos["user_followers"] = self.follow_map.get(user_index,0)
            infos["follower_description"] = follower_des
        except:
            pass
        
        return template.format_map(infos)
    
    def get_user_short_description(self,user_index):
        if isinstance(user_index,str):
            user_index = int(user_index)
        template ="""{user_name}, {user_description}"""
        infos = self.social_member_data.loc[user_index,:].to_dict()
        return template.format_map(infos)
    
    def get_user_friend_info(self,
                             user_index,
                             threshold:int = 30 # 超过30, 不显示friend信息
                             ):
        if isinstance(user_index,str):
            user_index = int(user_index)
        template ="""
Your friend include: 
{friends}
"""      
        friend_template = """{idx}:{short_description}"""
        
        infos = self.social_member_data.loc[user_index,:].to_dict()
        
        if len(infos["friend"]) > threshold:
            return f"""You have {len(infos["friend"])} friends, which are too many to show. You may consider not to follow others."""
        friend_infos = []
        for idx, friend_id in enumerate(infos["friend"]):
            friend_des = friend_template.format(idx = idx,
            short_description = self.get_user_short_description(friend_id))
            friend_infos.append(friend_des)
        
        return template.format(friends = "\n".join(friend_infos))
    

    def delete_user_profiles(self,
                            cur_time:str,
                            add_user_time_delta:int,
                            num_delete:int = 5) -> list:
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        add_user_time_delta = timedelta(days=add_user_time_delta)
        num_agents = self.social_member_data.shape[0]
        num_delete = min(num_agents - 1,num_delete)
        if num_delete < 1:return []
        if self.last_added_time == date.min or \
            add_user_time_delta > (cur_time - self.last_added_time):
            return []
        else:
            ids_delete = random.sample(self.social_member_data.index.to_list(),
                                       num_delete)
            return ids_delete
        
    def rerun_set_time(self,last_added_time):
        if isinstance(last_added_time,str):
            last_added_time = datetime.strptime(last_added_time,"%Y-%m-%d").date()
        self.last_added_time = last_added_time

    def rerun(self):
        return len(self.transitive_agent_log)>0

    def denote_transitive_log(self, 
                              delete_ids,
                              add_ids):
        self.transitive_agent_log.append({
            "delete_ids":delete_ids,
            "add_ids":add_ids
        })
        

    def update_docs(self):
        docs = self.forum_loader.load()
        self.db = FAISS.from_documents(docs, 
                                    self.embeddings)
    
    def get_forum_retriever(self,
                            **retriever_kargs_update):
        """offline的时候不进行filter"""
        if self.db is None:
            docs = self.forum_loader.load()
            self.db = FAISS.from_documents(docs, 
                                    self.embeddings)
        if (self.retriever_kargs_update != retriever_kargs_update) or \
            self.retriever is None:
            retriever_args = copy.deepcopy(self.retriever_kwargs)
            retriever_args["vectorstore"] = self.db
            retriever_args.update(retriever_kargs_update)
            retriever_args["compare_function_type"] = "social"
            self.retriever = retriever_registry.from_db(**retriever_args)
        return self.retriever
    
    def get_forum_retriever_tool(self,
                            document_prompt: Optional[BasePromptTemplate] = None,
                            social_follow_map:dict = {
                                "follow_ids": [],
                                "friend_ids": []
                            },
                            max_search:int = 5,
                            interested_topics:list = [],
                            **retriever_kargs_update):
        
        retriever = self.get_forum_retriever(**retriever_kargs_update)
        
        if retriever is None:return
        
        document_prompt = PromptTemplate.from_template("""
{tweet_idx}:
    user: {user_name}
    topic: {topic}
    tweet: {page_content}""")
        retriever_tool = create_forum_retriever_tool(
                    retriever,
                    "search_forum",
                    "You can search for anything you are interested on this platform.",
                    document_prompt = document_prompt,
                    big_name_list = self.big_name_list,
                    filter_keys = self.tool_kwargs["filter_keys"],
                    social_follow_map = social_follow_map,
                    interested_topics = interested_topics,
                    max_search = max_search,
                    hub_connect = self.tool_kwargs.get("hub_connect",True))
        return retriever_tool
    

    def update_add_user_time(self,
                             cur_time:str):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        self.last_added_time = cur_time


    def add_and_return_user_profiles(self, 
                                     cur_time:str,
                                     add_user_time_delta:int,
                                     num_added:int = 5) -> dict:
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        add_user_time_delta = timedelta(days=add_user_time_delta)
        # llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613",
        #                  temperature=0.7,
        #                  max_tokens=2000)
        llm = load_model_by_config_name("default")
        profiles = {}
        
        if self.last_added_time == date.min:
            if self.social_member_data.shape[0] < self.last_added_index:
                return {}
            # init data
            message_threshold = 10000
            profiles = self.social_member_data.iloc[self.last_added_index:]
            if profiles.shape[0] > message_threshold:
                profiles = profiles.iloc[:message_threshold]
                self.last_added_index += profiles.shape[0]
            return profiles.to_dict()

        elif add_user_time_delta > (cur_time - self.last_added_time):
            profiles = {}
        else:
            try:
                ids_added_num = 0
                step = num_added if num_added < 5 else 5
                for i in range(0,num_added,step):
                    ids_sub = self.update_person(llm, step)
                    ids_added_num += len(ids_sub)
                    if ids_added_num >= num_added:
                        break
                ids = self.social_member_data.index.to_list()[-ids_added_num:]
                profiles_df = self.social_member_data.loc[ids,:]
                friend_ids = profiles_df['user_index'].to_list()

                for index, profile_row in profiles_df.iterrows():
                    friend_ids_cp = copy.deepcopy(friend_ids)
                    friend_ids_cp.remove(profile_row['user_index'])
                    for friend_id_ in friend_ids_cp:
                        if friend_id_ not in profiles_df.loc[index, "friend"]:
                            profiles_df.loc[index, "friend"].append(friend_id_)
                try:
                    assert profiles_df.shape[0]<= num_added, print("error", f"update_agents \
                                                                   {num_added}/{profiles_df.shape[0]}")
                except:
                    return {}
                
                profiles = profiles_df.to_dict()
            except Exception as e:
                profiles = {}

        return profiles
    


    def add_tweets(self, 
                   agent_id: Union[str,int],
                   cur_time:str,
                   twitters:list = []):
        
        if isinstance(agent_id,str):
            agent_id = int(agent_id)
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        """update action logs"""
        tweets = []
        action_logs = []
        available_actions = ["tweet","retweet","reply"]
        for tweet in twitters:
            try:
                action = tweet.get("action","retweet").lower()
                if action == "tweet":
                    send_tweet = {
                        "text": tweet.get("input"),
                        "user_name": self.social_member_data.loc[agent_id,
                                                                 "user_name"],
                        "user_index": agent_id,
                        "topic":tweet.get("topic"),
                        "action":action,
                        "origin_tweet_idx": -1,
                        "owner_user_index": agent_id,
                    }
                    tweets.append(send_tweet)
                
                elif action in available_actions:
                    t_id = int(tweet.get("tweet_index"))
                    tweet_info_db = self.forum_loader.docs[int(t_id)]

                    if action == "reply":
                        reply_tweet = {
                            "text":tweet.get("input"),
                            "user_name": self.social_member_data.loc[agent_id,
                                                                    "user_name"],
                            "user_index": agent_id,
                            "topic": tweet_info_db.metadata.get("topic","default"),
                            "action":action,
                            "origin_tweet_idx": tweet_info_db.metadata.get("tweet_idx",-1),
                            "owner_user_index": tweet_info_db.metadata.get("owner_user_index",agent_id),
                        }
                        tweets.append(reply_tweet)

                    owner_id = tweet_info_db.metadata.get("owner_user_index")
                    assert owner_id is not None
                    if tweet.get("follow"):
                        if owner_id not in self.social_member_data.loc[agent_id,"follow"] and \
                            owner_id not in self.social_member_data.loc[agent_id,"friend"] and\
                            owner_id != agent_id:
                            if agent_id in self.social_member_data.loc[owner_id,"follow"]:
                                # 互关了，需要改成friend
                                self.social_member_data.loc[owner_id,"friend"].append(agent_id)
                                self.social_member_data.loc[owner_id,"follow"].remove(agent_id)
                                self.social_member_data.loc[agent_id,"friend"].append(owner_id)
                            else:
                                self.social_member_data.loc[agent_id,"follow"].append(owner_id) 

                            action_logs.append([agent_id,owner_id,"follow",cur_time]) 
                 
                if action in available_actions:   
                    action_logs.append([agent_id,owner_id,action,cur_time])
                
                for mention_id in tweet.get("mention",[]):
                    mention_id = int(mention_id)
                    if mention_id != agent_id:
                        action_logs.append([agent_id,mention_id,"mention",cur_time])

            except Exception as e:
                continue
            
        self.action_logs.extend(action_logs)
        if len(tweets) > 0:
            tweets = pd.DataFrame(tweets)
            docs = self.forum_loader.add_social(tweets)
            db_update = FAISS.from_documents(docs, self.embeddings)
            self.db.merge_from(db_update)
       
        return len(tweets)
            
    def update_person(self,
                       llm:ModelWrapperBase,
                       num_added:int = 5):
        prompt ="""
Your task is to give me a list of {num_added} person's profiles for twitter users . Respond in this format:
[
{{
"user_name": "(str;The name of this user)",
"user_description":"(str;short and concise, a general description of this user, ordinary users or super \
large users and the topics this person interested in)"
}}
]

Now please generate:
"""

        prompt = prompt.format(num_added = num_added)
        prompt_msg = llm.format(Msg("user",prompt,"user"))
        response = llm(prompt_msg)
        content = response.text
        followers_update = []
        ids_added = []
        
        try:
            # person_data_update = json.loads(content)
            person_data_update = find_and_load_json(content, "list")
            for idx,person in enumerate(person_data_update):
                try:
                    name = person["user_name"]
                    description = person["user_description"]
                    # followers = int(person["user_followers"])
                    followers_update.append({
                        "user_name":name,
                        "user_description":description,
                        # "user_followers":followers,
                        "follow":[],
                        "friend":[]
                    })

                except:continue
            followers_update = pd.DataFrame(followers_update)
            self.social_member_data = pd.concat([self.social_member_data,followers_update],ignore_index=True)
            ids_added = self.social_member_data.index.to_list()[-followers_update.shape[0]:]
            self.social_member_data['user_index'] = self.social_member_data.index
        except Exception as e:
            ids_added = []
            pass
        return ids_added
    
    def save_infos(self,
                   cur_time:str, 
                   start_time):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d")
        data_dir = os.path.join(self.generated_data_dir,"data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        social_member_data_dir = os.path.join(data_dir,"social_network")
        if not os.path.exists(social_member_data_dir):
            os.makedirs(social_member_data_dir)
            
        save_path = os.path.join(social_member_data_dir,
                                    "social_member_data_{day}.csv".format(
                                        day = cur_time.strftime("%Y%m%d"))
                                     )            
        self.social_member_data.to_csv(save_path,index=False)
        
        self.forum_loader.save(os.path.join(data_dir,"forum.json"))
        
        action_logs_searlize = copy.deepcopy(self.action_logs)
        for ac_log in action_logs_searlize:
            ac_log[-1] = ac_log[-1].strftime("%Y-%m-%d")
        
        writeinfo(os.path.join(data_dir,"action_logs.json"),
                  action_logs_searlize)
        writeinfo(os.path.join(data_dir,"transitive_agent_log.json"),
                  self.transitive_agent_log)
        
        ex_logs_path = os.path.join(data_dir,"ex_logs.json")
        import time
        simulation_time = time.time() - start_time
        simulation_time += self.simulation_time
        
        if os.path.exists(ex_logs_path):
            ex_logs = readinfo(ex_logs_path)
            start_time = ex_logs["simulation_time"]
            this_round = int(simulation_time) - int(start_time)
            round_times = ex_logs.get("round_times",[])
            round_times.append(this_round)
        else:
            this_round = int(simulation_time)
            round_times = [this_round]

        user_indexs = self.social_member_data['user_index'].to_list()
        delete_indexs = []
        for  transitive_agent_log_ in self.transitive_agent_log:
            delete_indexs.extend(transitive_agent_log_["delete_ids"])

        user_indexs = list(filter(lambda x: x not in delete_indexs, user_indexs))
        ex_logs = {
            "last_added_time":self.last_added_time.strftime("%Y%m%d"),
            "simulation_time":int(simulation_time),
            "twitters":len(self.action_logs),
            "cur_user_num":len(user_indexs),
            "round_times":round_times
        }
        writeinfo(os.path.join(data_dir,"ex_logs.json"),
                  ex_logs)
        
    def get_follow_ids(self, 
                     agent_id):
        if isinstance(agent_id,str):
            agent_id = int(agent_id)

        follow_ids = self.social_member_data.loc[
            agent_id,
            "follow"]
        friend_ids = self.social_member_data.loc[
            agent_id,
            "friend"]

        return {
            "follow_ids":follow_ids,
            "friend_ids":friend_ids
        }
            
                
                
    def save_networkx_graph(self):
        pass
    
    
   

    def sample_cur_agents(self, 
                          cur_agent_ids:list = [],
                          sample_ratio:float = 0.1,
                          sample_big_name_ratio:float = 0.3):
        
        hot_agent_ids = copy.deepcopy(self.big_name_list)
        common_agent_ids = list(filter(lambda x: x not in hot_agent_ids, cur_agent_ids))
       
        common_num = min(max(int(sample_ratio*len(common_agent_ids)),5),
                         len(common_agent_ids))
        hot_num = min(max(int(sample_big_name_ratio*len(hot_agent_ids)),5),
                         len(hot_agent_ids))
        return random.sample(common_agent_ids,common_num), \
            random.sample(hot_agent_ids,hot_num)
    
    def sample_cur_agents_llmplan(self, 
                                agent_plans_map:dict ={}
                                ):
        """按照二项分布采样"""
        hot_agent_ids = []
        common_agent_ids = []
        import numpy as np
        for day, agent_ids in agent_plans_map.items():
            p = int(day)/30
            p = 1 if p > 1 else p
            sampled_list = np.array(agent_ids)[np.random.rand(len(agent_ids)) <= p]
            for agent_id in sampled_list:
                if agent_id in self.big_name_list:
                    hot_agent_ids.append(int(agent_id))
                else:
                    common_agent_ids.append(int(agent_id))
        return common_agent_ids, hot_agent_ids


    def get_user_num_followers(self, user_index:int):
        if isinstance(user_index,str):
            user_index = int(user_index)
        return self.follow_map.get(user_index,0)
    
    def update_big_name_list(self):
        """get big name list: follower count > threshold"""
        self.update_follow_map()
        threshold_ratio = self.control_profile.get("hub_rate",0.2)
        threshold = int(self.social_member_data.shape[0]*threshold_ratio)
        follower_count_list = [(agent_id, self.follow_map.get(agent_id,0))
                          for idx, agent_id
                          in enumerate(self.social_member_data.index)]
        social_member_filter = sorted(follower_count_list, key=lambda x: x[1], reverse=True)[:threshold]
        social_member_filter_ids = [x[0] for x in social_member_filter]
        self.big_name_list = social_member_filter_ids
        

    def update_follow_map(self):
        for user_index in self.social_member_data['user_index']:
            for follow_id in self.social_member_data.loc[user_index,
                                                     "follow"]:
                self.follow_map[follow_id] = self.follow_map.get(follow_id,0) + 1
            for friend_id in self.social_member_data.loc[user_index,
                                                     "friend"]:
                self.follow_map[friend_id] = self.follow_map.get(friend_id,0) + 1    
            

    def get_memory_init_kwargs(self, user_index:int):
        seen_tweets = []
        posted_tweets = []
        
        action_counts = {}
        posted_topics = []
        document_prompt = PromptTemplate.from_template("""
{tweet_idx}:
    user: {user_name}
    topic: {topic}
    tweet: {page_content}""")
        
        user_tweets = list(filter(lambda doc: doc.metadata["user_index"] == user_index,
                                  self.forum_loader.docs))
        actions = [tweet.metadata["action"] for tweet in user_tweets]
        for tweet in user_tweets:
            posted_tweets.append(document_prompt.format(**tweet.metadata,
                                                        page_content=tweet.page_content))
            posted_topics.append(tweet.metadata["topic"])
        
        topic_memory = {
            "posted_topics":posted_topics,
            "followed_topics":[]
        } 

        action_counter = Counter(actions)
        action_counts = dict(action_counter.items()) 
        return {
            "seen_tweets":seen_tweets,
            "posted_tweets":posted_tweets,
            "topic_memory":topic_memory,
            "action_counts":action_counts
        }
    
    def get_follower_description(self,user_index:str):
        if isinstance(user_index,str):
            user_index = int(user_index)
        template = """
Here's the list of people you followed:

{user_names}
"""
        followed_agent_ids = self.social_member_data.loc[user_index,
                                                     "follow"]
        followed_agent_names = [
            self.social_member_data.loc[followed_agent_id,
                                        "user_name"]
            for followed_agent_id in followed_agent_ids
        ]
        return template.format(user_names =
                               ",".join(followed_agent_names))
    
    def get_user_big_name(self,user_index:str):
        if isinstance(user_index,str):
            user_index = int(user_index)
        big_name = user_index in self.big_name_list
        return big_name
    
    def return_deleted_agent_ids(self):
        delete_ids = []
        for transitive_agent_log in self.transitive_agent_log:
            delete_ids.extend(transitive_agent_log["delete_ids"])
        return delete_ids
    
    def plot_agent_plan_distribution(self,
                                     cur_time:str,
                                     sampled_agent_ids:dict = {},
                                     agent_plans_map:dict = {}):
        data_dir = os.path.join(self.generated_data_dir,"data","plans")
        os.makedirs(data_dir,exist_ok=True)
        from LLMGraph.utils.io import writeinfo
        data_dir = os.path.join(data_dir,f"agent_plans_{cur_time}.json")
        writeinfo(data_dir,
                  {"sampled_agent_ids":sampled_agent_ids,
                   "agent_plans_map":agent_plans_map,
                   "big_name_list":self.big_name_list})
        
        ## plot fig
        # data_dir = os.path.join(data_dir,f"agent_plans_{cur_time}.pdf")
        import matplotlib.pyplot as plt
        

        # # 绘制折线图
        # plt.figure(figsize=(10, 5))  # 可以调整图像的大小
        # plt.plot(days, plan_lengths, marker='o')  # 使用'o'标记每个点

        # # 添加标题和轴标签
        # plt.ylabel('Agent Number')
        # plt.xlabel('Activity Frequency')

        # # 显示图形
        # plt.xticks(rotation=45)  # 可能需要旋转x轴标签，以防它们互相重叠
        # plt.grid(True)  # 显示网格线
        # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        # plt.savefig(data_dir)
