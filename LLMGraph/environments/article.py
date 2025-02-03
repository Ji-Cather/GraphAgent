from typing import List, Deque, Optional, Any



import json

import threading

from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import os
import random
from LLMGraph.wrapper.agent_group import GroupDiscussAgent
from LLMGraph.wrapper import ArticleAgentWrapper
from LLMGraph.agent.article import ArticleAgent, ArticleManagerAgent
from LLMGraph.retriever import retriever_registry

from datetime import datetime, timedelta, date
from agentscope.message import Msg, PlaceholderMessage

from agentscope.agents.rpc_agent import RpcAgentServerLauncher,RpcAgent
        
@EnvironmentRegistry.register("article")
class ArticleEnvironment(BaseEnvironment):
    """
    A environment implementing the logic of conversation.
    """
    agent_groups:list = [] # 存储生成各个article的agents
    
    author_agents:dict = {} # id: article_agent
    
    agent_configs:dict = {} # 存储agent的config
    
    

    
    time_configs:dict ={
        "cur_time": datetime.now().date(),
        "end_time": datetime.now().date() + timedelta(days=365),
        "round_time_delta": timedelta(days=10),
        "author_time_delta": timedelta(days=30), # 以天为单位,
        "author_num_per_delta": 10,
        "article_num_per_delta": 3,
    }
    
    article_written_num:int = 0
    max_paper_num:int  = 20
    save_encoded_features:int = -1
    
    class Config:
        arbitrary_types_allowed = True
    
    

    def __init__(self, 
                 launcher_args:list = [],
                 **kwargs):
        to_dist = len(launcher_args) > 0
        article_manager_configs = kwargs.pop("managers").pop("article")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")

        time_configs = kwargs.pop("time_configs",{})
        time_configs["start_time"] = time_configs["cur_time"]
        time_configs["round_time_delta"] = timedelta(days=time_configs["round_time_delta"])
        time_configs["author_time_delta"] = timedelta(days=time_configs["author_time_delta"])
        
    
        article_write_configs = kwargs.pop("article_write_configs")
        
        if to_dist:
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_args[0]["host"],
                    "port":launcher_args[0]["port"]
                }
            }
        else:
            to_dist_kwargs = {}

        model_config_name = article_manager_configs.pop("model_config_name",
                                                        "vllm")
        manager_agent = ArticleManagerAgent(name = "article_manager",
                                     article_manager_configs = article_manager_configs,
                                    model_config_name = model_config_name,
                                    task_path = task_path,
                                    config_path = config_path,
                                    cur_time = time_configs["cur_time"].strftime("%Y-%m-%d"),
                                    article_write_configs = article_write_configs,
                                    **copy.deepcopy(to_dist_kwargs)
                                    )

        super().__init__(
                        manager_agent = manager_agent,
                        agent_configs = agent_configs,
                        time_configs = time_configs,
                        launcher_args = launcher_args,
                        to_dist = to_dist,
                        **kwargs
                        )
        
        cur_time = self.call_manager_agent_func(
            "get_start_time"
        ).content
        try:
            cur_time = datetime.strptime(cur_time, "%Y-%m-%d").date()
            self.time_configs["cur_time"] = cur_time
        except:
            pass
    
    
        
    def reset(self) -> None:
        """Reset the environment"""
        pass

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        self.article_written_num = self.call_manager_agent_func(
            func_name = "get_article_written_num").content
        return self.article_written_num >= self.max_paper_num \
            or self.time_configs["cur_time"] >= self.time_configs["end_time"]
        
        
    
    # def generate_topic_one_group(self,
    #                             group_id,
    #                             communication_num = 10):
                                      
    #     """
    #     the async run parse of tenant(tenant_id) communication.
    #     return: the receivers, self(if continue_communication)
    #     """
    #     agents = self.agent_groups[group_id]["authors"]
    #     research_content = self.agent_groups[group_id]["content"]
    #     idx = 0
        
    #     for agent in agents:
    #         self.call_agent_func(agent,"clear_discussion_cur")

    #     from agentscope.msghub import msghub
    #     with msghub(participants=agents) as hub:
    #         for idx in range(communication_num):
    #             agent = agents[idx%len(agents)]
    #             role_description = self.call_manager_agent_func("get_author_description",
    #                         kwargs={"agent_name":agent.name}).content
    #             # print(role_description)
    #             if ((idx+1)%2==0):
    #                 research_content = self.call_agent_func(agent,
    #                                                          "idea_generation",
    #                                                          kwargs={"role_description":role_description,
    #                                                      "research_content":research_content}).content
    #                 finish_generation = research_content["finish"]
    #                 if finish_generation: 
    #                     self.agent_groups[group_id]["content"] = research_content
    #                     break
                
    #             candidate_id_msg = self.call_agent_func(agent, "choose_researcher",
    #                                  kwargs={"role_description":role_description,
    #                                          "research_content":research_content})
                
    #             candidate_id = candidate_id_msg.content
    #             role_description_2 = self.call_manager_agent_func("get_author_description",
    #                         kwargs={"agent_name":candidate_id}).content
                
    #             group_discussion_msg = self.call_agent_func(agent, "group_discuss",
    #                                  kwargs={"role_description_1":role_description,
    #                                          "role_description_2":role_description_2,
    #                                          "research_content":research_content,
    #                                          "author_id": candidate_id})
   
    

            
    def write_article_one_group(self,
                                group_id):
                                      
            """
            the async run parse of tenant(tenant_id) communication.
            return: the receivers, self(if continue_communication)
            """
            agent_first_author = self.agent_groups[group_id]["authors"][0]
            research_content = self.agent_groups[group_id]["content"]

            research_content = self.call_agent_func(agent_first_author, 
                                "write_article",
                                 kwargs={"research_content":research_content})
            
            
            research_content = research_content.content
            # research_content = self.call_agent_func(agent_first_author, 
            #                                          "choose_reason",
            #                                         kwargs={"research_content":research_content,
            #                                                 "cur_time_str":self.time_configs["cur_time"].strftime("%Y-%m-%d")})
            # research_content = research_content.content
            
            
            
    #test 测试用 要改
    def communication(self):
        
        research_msgs = []
        for group_id, group_info in enumerate(self.agent_groups):
            group = group_info["authors"]
            assert isinstance(group,GroupDiscussAgent) or isinstance(group, RpcAgent)

            research_content_msg = group(Msg(
                content = self.agent_groups[group_id]["content"],
                role="user",
                name ="user",
                func = "communication",
                kwargs={"research_content":self.agent_groups[group_id]["content"]}))
            research_msgs.append(research_content_msg)
        
        for group_id, research_msg in enumerate(research_msgs):
            self.agent_groups[group_id]["content"] = research_msg.content
            
            

    def write(self):
        research_msgs = []
        
        for group_id, group_info in enumerate(self.agent_groups):
            group = group_info["authors"]
            assert isinstance(group,GroupDiscussAgent) or isinstance(group, RpcAgent)
            research_content_msg = group(Msg(
                content = "call_function",
                role="user",
                name ="user",
                func = "write",
                kwargs={"research_content":self.agent_groups[group_id]["content"],
                        "cur_time_str":self.time_configs["cur_time"].strftime("%Y-%m-%d")
                        }))
            research_msgs.append(research_content_msg)

        for group_id, research_msg in enumerate(research_msgs):
            self.agent_groups[group_id]["content"] = {**research_msg.content,
                                                      "time":self.time_configs["cur_time"].strftime("%Y-%m-%d")}
            
        
    
    
    



    def init_agent(self, author, author_info, sn, launcher_id = 0) -> ArticleAgentWrapper:
       
        if self.to_dist:
            launcher_arg = self.launcher_args[launcher_id]
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_arg["host"],
                    "port":launcher_arg["port"]
                }
            }
        else:
            to_dist_kwargs = {}
        
        agent = ArticleAgent(name = author,
                            infos = author_info,
                            agent_configs = self.agent_configs,
                            social_network = sn,
                            **copy.deepcopy(to_dist_kwargs))
                    # print("article_created")
        author_agent = ArticleAgentWrapper(
                            name = author,
                            agent = agent,
                            manager_agent = self.manager_agent,
                            max_tool_iters = 1,
                            **copy.deepcopy(to_dist_kwargs))
        
        return author_agent

    def group_assign_topic(self,
                   article_number = 10,
                   author_number = 5):
        """_summary_

        Args:
            article_number (int, optional): the number of articles to be generated. Defaults to 10.
            author_number (int, optional): the number of authors for topic discussion (per article). Defaults to 5.
        """
        article_write_configs = self.call_manager_agent_func(func_name="get_article_write_configs").content
        topics = self.call_manager_agent_func("get_topics").content
        group_agents = []
        for i in range(article_number):
            #random sample
            if len(topics) == 0:
                raise Exception("no topic available")
            else:
                topic = random.choice(topics)
            authors = []
            iter_time = 0
            max_iter = 3
            if author_number >1:
                while(len(authors)<2 and iter_time<max_iter):
                    authors = self.call_manager_agent_func(
                        "get_most_cooperated_author",
                        kwargs={
                            "topic":topic,
                            "author_num":author_number
                        }
                    ).content 
                    # authors = self.call_manager_agent_func(
                    #     "get_llm_author",
                    #     kwargs={
                    #         "topic":topic,
                    #         "author_num":author_number
                    #     }
                    # ).content # 暂不支持单作者
                    iter_time += 1
            else:
                authors = self.call_manager_agent_func(
                        "get_llm_author",
                        kwargs={
                            "topic":topic,
                            "author_num":1
                        }
                    ).content 
                authors = [authors[0]]


            agent_ids =[]
            author_agents = []
            for author in authors:
                sn = {}
                for s_id in authors:
                    if author != s_id:
                        sn[s_id] = self.call_manager_agent_func(
                            "get_author",
                            kwargs={
                                "author_id":s_id
                            }
                        ).content
                
                if author in self.author_agents.keys():
                    author_agent = self.author_agents[author]
                    add_sn_msg = Msg("user",
                                     content="call_function",
                                     kwargs={
                                            "social_network":sn
                                           },
                                     func="add_social_network"
                                       )
                    author_agent(add_sn_msg).content
                    # assert isinstance(author_agent,ArticleAgent)
                    
                else:
                    agent_ids.append(author)
                    author_info = self.call_manager_agent_func(
                            "get_author",
                            kwargs = {
                                "author_id":author
                            }
                        ).content
                    
                    """initialize agent_wrapper(tool(to_dist), agent(to_dist))"""
                    if self.to_dist:
                        launcher_id = i% len(self.launcher_args)
                    else:
                        launcher_id = 0
                    author_agent = self.init_agent(author,author_info,sn,launcher_id = launcher_id) 
                   
                   
                    self.author_agents[author] = author_agent
                    
                author_agents.append(author_agent)
            
            for agent in author_agents:
                self.call_agent_func(agent,"clear_discussion_cur").content
            
            # another layer of wrapper
            if self.to_dist:
                launcher_id = i% len(self.launcher_args)
                launcher = self.launcher_args[launcher_id]
                to_dist_kwargs = {
                    "to_dist":{
                        "host":launcher["host"],
                        "port":launcher["port"]
                    }
                }
            else:
                to_dist_kwargs = {}

            group_one = GroupDiscussAgent(name = f"group_{i}",
                        communication_num = article_write_configs["communication_num"],
                        agents = author_agents,
                        manager_agent = self.manager_agent,
                        **copy.deepcopy(to_dist_kwargs))

                
            article ={
                "authors":group_one,
                "content":{
                    "topic": topic,
                    "keywords":[],
                    "abstract":"",
                    "citations":[],
                    "title":"",
                    "author_ids":[agent.name for agent in author_agents],
                    "success":False
                    }
            }
            group_agents.append(article)
           
        self.agent_groups = group_agents
        
    
    

            
    def update_time(self):
        self.time_configs["cur_time"] += self.time_configs["round_time_delta"]
        

    def add_author(self):
        """这边需要按照活跃比例 去修改cur_agent_ids"""
        add_author_msg = Msg(
            "user",
            content="call_function",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "author_time_delta":self.time_configs["author_time_delta"].total_seconds(),
                "author_num_per_delta":self.time_configs["author_num_per_delta"]
                },
            func="add_author"
        )
        return_msg = self.manager_agent(add_author_msg)
        if isinstance(return_msg, PlaceholderMessage):
            return_msg.update_value()# 堵塞
            
    def step(self):       
        self.update_time()
        
        self.add_author()

        article_write_configs = self.call_manager_agent_func(func_name="get_article_write_configs").content
        self.group_assign_topic(
            article_number = self.time_configs["article_num_per_delta"],
            author_number = article_write_configs["author_num"])
        
        # if article_write_configs["author_num"] >1:
        #     self.communication()
        
        # 接下来是写论文的过程
        self.write()
        
        articles = [agent_group["content"] for agent_group in self.agent_groups]

        for idx in range(0, len(articles), 20):
            sub_articles = articles[idx:idx+20]
            return_msg = self.call_manager_agent_func(func_name="update",
                                                 kwargs = {
                                                     "articles":sub_articles,
                                                     "update_retriever":False
                            }).content # 初始化,在to_dist之后初始化
        self.call_manager_agent_func(func_name="update",
                                     kwargs = {
                                         "update_retriever":True
                            }).content # 初始化,在to_dist之后初始化)


        self.agent_groups = []
        
            

    def save(self,
             start_time):
        self.call_manager_agent_func(func_name="save",
                                     kwargs = {
                                         "start_time":start_time,
                                         "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                                         "save_encoded_features":self.save_encoded_features
                                }).content
        
        
    def eval(self):
        """visualize LDA kmeans png"""
        self.call_manager_agent_func(func_name="plot_article_lda").content