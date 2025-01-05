

from typing import List, Union
import json
from LLMGraph.manager import MovieManager


from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import os
import random
from agentscope.message import Msg
from LLMGraph.agent.movie import MovieAgent, MovieManagerAgent
from LLMGraph.wrapper import MovieAgentWrapper
from datetime import datetime, timedelta
from agentscope.agents.rpc_agent import RpcAgentServerLauncher,RpcAgent
 
@EnvironmentRegistry.register("movie")
class MovieEnvironment(BaseEnvironment):
    """
    A environment implementing the logic of conversation.
    
    Args:
        agents: tenant_manager
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """

    cur_agents:dict = {} # 存储所有agent
    
    agent_configs:dict = {} # 存储agent的config
    
    movie_rate_configs:dict = {
        "watch_plan":"SYSTEM",
        "min_rate_all": 200 ,# 总体最少做多少个评价
        }
    
    time_configs:dict ={
        "start_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "cur_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "end_time": datetime.strptime("2001-12-31", '%Y-%m-%d'),
        "movie_time_delta": timedelta(days=4*30) ,# 4个月,
        "watcher_time_delta": timedelta(days=30),
        "watcher_add": False, # 所有时间点都是 n个watcher进行rate
        # 如果是true,则按照watcher_time_delta分批进入
    }
    
    
    cur_rate:int = 0
    
    movie_agents:dict = {} # id: movie_agent 存储所有的agent 暂时没用到
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 launcher_args:list = [],
                 **kwargs):
        
        to_dist = len(launcher_args) > 0
        if to_dist:
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_args[0]["host"],
                    "port":launcher_args[0]["port"]
                }
            }
        else:
            to_dist_kwargs = {}
            
        movie_manager_configs = kwargs.pop("managers").pop("movie")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")
        movie_data_dir = os.path.join(task_path,
                                      movie_manager_configs.pop("movie_data_dir"))
       
        link_movie_path = os.path.join(task_path,
                                          movie_manager_configs.pop("link_movie_path"))
        generated_data_dir = os.path.join(os.path.dirname(config_path),
                                          movie_manager_configs.pop("generated_data_dir"))
        
        time_configs = kwargs.pop("time_configs",{})
        time_configs["movie_time_delta"] = timedelta(days=30*time_configs["movie_time_delta"])
        time_configs["watcher_time_delta"] = timedelta(days=30*time_configs["watcher_time_delta"])
        
        manager_agent = MovieManagerAgent(
            name = "movie_manager",
            model_config_name="default",
            movie_data_dir = movie_data_dir,
            link_movie_path=link_movie_path,
            generated_data_dir=generated_data_dir,
            movie_manager_configs = movie_manager_configs,
            start_time = time_configs["start_time"].strftime("%Y-%m-%d"),
            cur_time = time_configs["cur_time"].strftime("%Y-%m-%d"),
            movie_time_delta = time_configs["movie_time_delta"].days,
            **copy.deepcopy(to_dist_kwargs)
        )
        movie_rate_configs = kwargs.pop("movie_rate_configs")
    

        call_func_msg = Msg("user",
                            role="user",
                            func = "load_history",
                            content="call_function")
        return_value = manager_agent(call_func_msg).content
        if return_value is not None:
            cur_time = return_value.get("cur_time",time_configs["cur_time"]) 
            cur_rate = return_value.get("cur_rate",0)
            if isinstance(cur_time,str):
                time_configs["cur_time"] = datetime.strptime(cur_time, '%Y-%m-%d').date()
        else:
            cur_rate = 0
            
        print(f"Generated ratings number: {cur_rate}")
        
        super().__init__(manager_agent = manager_agent,
                         agent_configs = agent_configs,
                         movie_rate_configs = movie_rate_configs,
                         cur_rate = cur_rate,
                         time_configs = time_configs,
                         to_dist = to_dist, 
                         launcher_args = launcher_args,
                         **kwargs)
        

    def initialize(self):
        self.update_agents()
        
    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        
        return self.cur_rate >= self.movie_rate_configs["min_rate_all"] \
            or self.time_configs["cur_time"] >= self.time_configs["end_time"]
   
            

    def rate_movie(self):
        def run_parallel():
            if len(self.cur_agents)==0: return {}
            rating_msgs:List[Msg] =[]
            for agent_id,agent in list(self.cur_agents.items()):
                rate_msg = self.call_agent_func(agent, 
                                                 "rate_movie_process", 
                                        kwargs = {"cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                                                "max_retry_rate":2})
                rating_msgs.append(rate_msg)

            ratings = []
            thunk_size = len(self.launcher_args)
            if thunk_size == 0:
                thunk_size = 1
            for i in range(0,len(rating_msgs),thunk_size):
                thunk = rating_msgs[i:i+thunk_size]
                ratings_sub = [msg.content for msg in thunk]
                ratings.extend(ratings_sub)
            # ratings = [rating_msg.content for rating_msg in rating_msgs]
            return ratings
        
        return run_parallel() # 需要进行讨论的tenant
    
    
    def update_time(self):
        self.time_configs["cur_time"] = self.time_configs["cur_time"] + self.time_configs["watcher_time_delta"]
        if isinstance(self.time_configs["cur_time"],datetime):
            self.time_configs["cur_time"] = self.time_configs["cur_time"].date()
        
    
    
    
    def init_agent(self, 
                   name: str,
                   infos:dict,  
                   launcher_id = 0,
                   ) -> MovieAgentWrapper:
        rating_counts_id = self.call_manager_agent_func(
                "get_watcher_rating_infos",
                kwargs={
                    "watcher_id":name,
                }
            ).content
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
        
        agent_rating_counts = self.call_manager_agent_func(
            "get_rating_counts",
            kwargs={
                "rating_counts_id": rating_counts_id
            }
        ).content
        
        agent = MovieAgent(name = name,
                           infos = infos,
                            agent_configs=self.agent_configs,
                            rating_counts=agent_rating_counts["rating_counts"],
                            ratings=agent_rating_counts["ratings"],    
                            **copy.deepcopy(to_dist_kwargs))

        wrapper_agent = MovieAgentWrapper(
                            name = name,
                            agent = agent,
                            manager_agent = self.manager_agent,
                            max_tool_iters = 2,
                            max_retrys = 3,
                            **copy.deepcopy(to_dist_kwargs))
        return wrapper_agent


    def update_agents(self):
        """initialize the agents and plans"""
        
        agent_profiles = self.call_manager_agent_func(
                func_name="add_and_return_watcher_profiles",
                kwargs = {
        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
        "watcher_add":self.time_configs["watcher_add"],
        "watcher_num":self.time_configs["watcher_num"]
                }).content

        agents_added = []
        if len(agent_profiles) == 0 :return

        for idx,agent_profile in enumerate(agent_profiles):
            assert agent_profile["id"] not in self.cur_agents.keys(),"error!"
            

            if self.to_dist:
                launcher_id = idx%len(self.launcher_args)
            else:
                launcher_id = 0
            agent = self.init_agent(agent_profile["id"], 
                                    agent_profile, 
                                    launcher_id=launcher_id,)
           
            agents_added.append(agent)
            self.cur_agents[agent_profile["id"]] = agent
        
        def run_parallel(task,
                         len_v:int,
                        agents,
                        requirement:str = "You should watch a lot of movies and rate them."):
            watch_msgs :List[Msg] = []
            for agent in agents:
                # system/llm control plan
                watch_msg = self.call_agent_func(agent, 
                                                 "get_movie_watch_plan", 
                                        kwargs = {"task":task,
                                                "len_v":len_v,
                                                "use_default_plan":self.movie_rate_configs.get("watch_plan",
                                                                                               "SYSTEM")== "SYSTEM",
                                                "requirement":requirement})
                
                watch_msgs.append(watch_msg)
            
            # 堵塞
            [watch_msg.content for watch_msg in watch_msgs]
        
        
        number_years = (self.time_configs["end_time"] - self.time_configs["cur_time"])\
            //timedelta(days=365)
        round_delta = self.time_configs["watcher_time_delta"]
        
        template = """You should make a plan to watch a few movies every {month_num} \
months for the next {year_num} years, \
You can allocate your time reasonably without going to see it every month.\
So you need to give your plan, and respond in the format of a vector with a length of {len_v}."""
        month_num = round_delta//timedelta(days=30)
        len_v = number_years* (12//month_num + 1)# 为了防止越界
        
        task = template.format(month_num = month_num,
                               len_v = len_v,
                               year_num = number_years)
        run_parallel(task, len_v, agents_added) # 需要进行讨论的tenant
        
        

    def step(self):
        
        self.call_manager_agent_func(
            "add_movies",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "movie_time_delta":self.time_configs["movie_time_delta"].days,
            }
        ).content

        self.call_manager_agent_func("update").content

        # self.update_agents() # 一开始加入所有agent
        # rate movie process

        ratings = self.rate_movie()
        
        # update rating DB
        self.update_movie_manager(ratings)
                
        # update movie/watcher DB and Time
        self.update_time()
        
        
    def update_movie_manager(self,
                             ratings
                             ):
        num = self.call_manager_agent_func("update_db_ratings",
                                           kwargs={
                                               "ratings":ratings,
                                               "agent_ids":list(self.cur_agents.keys()),
                                           }).content
        self.cur_rate += num
        
        
    
    def save(self, start_time):
        self.call_manager_agent_func(
            "save",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "start_time":start_time
            }
        )

        
    def test(self):
        """define unittest"""
        pass

    def eval(self):
        pass