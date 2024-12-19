import asyncio

from typing import List, Any, Union

from LLMGraph.agent.social import SocialAgent,SocialManagerAgent
from LLMGraph.wrapper.social import SocialAgentWrapper

import json
from LLMGraph.manager import SocialManager
import pandas as pd
from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import os
import random
from dateutil.relativedelta import relativedelta

from agentscope.agents.rpc_agent import RpcAgentServerLauncher
from datetime import datetime, timedelta
from agentscope.message import Msg, PlaceholderMessage

@EnvironmentRegistry.register("social")
class SocialEnvironment(BaseEnvironment):
    """social environment for twitter

    Args:
        BaseEnvironment (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    cur_agents:dict = {} # 存储所有agent

    sampled_agent_ids:dict ={
        "big_name":[],
        "common":[]
    }
    
    agent_configs:dict # 存储agent的config
    
    time_configs:dict ={
        "start_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "cur_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "end_time": datetime.strptime("2001-12-31", '%Y-%m-%d'),
        "social_time_delta": timedelta(days = 1) ,# 一天
    }
    
    social_configs:dict ={
        "max_people":100,
        "add_people_rate":0.1,
        "delete_people_rate":0.1
    }
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 launcher_args=[],
                 **kwargs):
        to_dist = len(launcher_args) > 0
        social_manager_configs = kwargs.pop("managers").pop("social")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")
        social_data_dir = os.path.join(task_path,
                                      social_manager_configs.pop("social_data_dir"))
       
        generated_data_dir = os.path.join(os.path.dirname(config_path),
                                          social_manager_configs.pop("generated_data_dir"))
        
        time_configs = kwargs.pop("time_configs",{})
        time_configs["social_time_delta"] = timedelta(days=time_configs["social_time_delta"])
        time_configs["people_add_delta"] = timedelta(days=time_configs["people_add_delta"])

       
        if to_dist:
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_args[0]["host"],
                    "port":launcher_args[0]["port"]
                }
            }
        else:
            to_dist_kwargs = {}

        manager_agent = SocialManagerAgent( # pylint: disable=E1123
                                            name = "socialmanager",
                                           social_data_dir = social_data_dir,
                                            generated_data_dir = generated_data_dir,
                                            social_manager_configs=social_manager_configs,
                                            cur_time=time_configs["cur_time"].strftime("%Y-%m-%d"),
                                            # to_dist = {
                                            #         "host":"localhost",
                                            #         "port":"2333"
                                            #     } 
                                            **copy.deepcopy(to_dist_kwargs)
                                            )
        cur_time = manager_agent(
            Msg("user",
                content="get_start_time",
                kwargs={},
                func="get_start_time"
            )
        ).content
        
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        time_configs["cur_time"] = cur_time

        super().__init__(manager_agent = manager_agent,
                         agent_configs = agent_configs,
                         time_configs = time_configs,
                         launcher_args = launcher_args,
                         to_dist = to_dist,
                         **kwargs
                         )

    
    
        
    def reset(self) -> None:
        """Reset the environment"""
        pass

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        
        return self.time_configs["cur_time"] >= self.time_configs["end_time"] or \
        len(self.cur_agents) < 1
   
            
    def social_one_agent(self,
                        agent,
                        big_name:bool = False) -> Msg:
                                      
        """
        the async run parse of tenant(tenant_id) communication.
        return: the receivers, self(if continue_communication)
        """
        # assert isinstance(agent, SocialAgentWrapper)
        social_content_msg = self.call_agent_func(agent,
                                            "twitter_process",
                                            kwargs={
                                                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                                                "big_name": big_name
                                            })
        return social_content_msg
            
            
    def social(self,
               agent_plans_map:dict = {}):
        """按照随机采样"""
        # common_agent_ids, hot_agent_ids = self.call_manager_agent_func(
        #     "sample_cur_agents",
        #                             kwargs ={
        #                             "cur_agent_ids":list(self.cur_agents.keys()),
        #                             "sample_ratio":0.1,
        #                             "sample_big_name_ratio":0.3
        #                             }).content

        """按照llm生成的概率采样"""
        
        common_agent_ids, hot_agent_ids = self.call_manager_agent_func(
            "sample_cur_agents_llmplan",
            kwargs ={
            "agent_plans_map": agent_plans_map
            }).content
    
        self.sampled_agent_ids["big_name"] = hot_agent_ids
        self.sampled_agent_ids["common"] = common_agent_ids
        
        self.call_manager_agent_func(
            "plot_agent_plan_distribution",
            kwargs={
                "cur_time": datetime.strftime(self.time_configs["cur_time"],"%Y-%m-%d"),
                "agent_plans_map": agent_plans_map,
                "sampled_agent_ids": self.sampled_agent_ids
            }
        ).content

        def run_parallel(time_out_seconds = 300):
            if len(self.cur_agents)==0: return []

            agent_ids_cur = []
            for agent_id in self.sampled_agent_ids.get("big_name",[]):
                agent_ids_cur.append([agent_id,True])
            for agent_id in self.sampled_agent_ids.get("common",[]):
                agent_ids_cur.append([agent_id,False])
            
            print(len(self.sampled_agent_ids.get("big_name",[])), "big_name_len", 
                  len(self.sampled_agent_ids.get("common",[])), "user_len")
            
            twitters_content = []
            for idx in range(0, len(agent_ids_cur), 100):
                sub_agent_ids = agent_ids_cur[idx:idx+100]
                sub_msgs = [
                    self.social_one_agent(self.cur_agents[agent_id[0]], agent_id[1])
                    for agent_id in sub_agent_ids
                ]
                sub_twitters_content = [msg.content for msg in sub_msgs]
                twitters_content.extend(sub_twitters_content)


            for content in twitters_content:
                assert content is not None
            return twitters_content
        
        return run_parallel() # 需要进行讨论的tenant
    
    def collect_agent_plans(self):
        def run_parallel():
            if len(self.cur_agents)==0: return []

            cur_agents_keys = list(self.cur_agents.keys())
            plans_content = []

            for idx in range(0, len(self.cur_agents), 100):
                sub_plans = []
                cur_agents_keys_sub = cur_agents_keys[idx:idx+100]
                for agent_id in cur_agents_keys_sub:
                    agent = self.cur_agents[agent_id]
                    msg = self.call_agent_func(agent,
                                     "get_acting_plan"
                                     )
                    sub_plans.append(msg)

                sub_plans_content = [plan.content for plan in sub_plans]
                plans_content.extend(sub_plans_content)

            for content in plans_content:
                assert content is not None
            return plans_content
        
        plans_all_agent = run_parallel()
        agent_plans_map = {i:[] for i in range(1, 31)}
        for plan_agent, agent_id in zip(plans_all_agent, 
                                        self.cur_agents.keys()):
            log_in_days = plan_agent.get("log_in_days", 8)
            agent_plans_map[log_in_days].append(agent_id)

        return agent_plans_map
    
    def update_time(self):
        self.time_configs["cur_time"] = self.time_configs["cur_time"] \
            + self.time_configs["social_time_delta"]
        if isinstance(self.time_configs["cur_time"], datetime):
            self.time_configs["cur_time"] = self.time_configs["cur_time"].date()

    def init_agent(self, 
                   name:Union[int, str],
                   infos:dict,  
                   launcher_id = 0) -> SocialAgentWrapper:
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

        memory_init_kwargs = self.call_manager_agent_func(
            "get_memory_init_kwargs",
            kwargs={
                "user_index":name
            }
        ).content

        agent = SocialAgent(name = name,
                            infos = infos,
                            agent_configs=self.agent_configs,
                            memory_init_kwargs = memory_init_kwargs,
                            **copy.deepcopy(to_dist_kwargs))

        wrapper_agent = SocialAgentWrapper(
                            name = name,
                            agent = agent,
                            manager_agent = self.manager_agent,
                            max_tool_iters = 1,
                            **copy.deepcopy(to_dist_kwargs))
        return wrapper_agent
    
    def update_agents(self,
                      denote_transitive_log:bool = True):
        """initialize the agents and plans"""
        num_added = int(len(self.cur_agents) * \
            self.social_configs["add_people_rate"])
        
        num_deleted = int(len(self.cur_agents) * \
            self.social_configs["delete_people_rate"])

        delete_agent_ids = self.call_manager_agent_func(
            "delete_user_profiles",                                                                 
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "add_user_time_delta":self.time_configs["people_add_delta"].days,
                "num_delete":num_deleted
            }
        ).content

        for cur_agent_id in delete_agent_ids:
            if cur_agent_id in self.cur_agents.keys():
                self.cur_agents.pop(cur_agent_id)
                print(f"Agent {cur_agent_id} deleted")
        
        agent_profiles = None
        """add agents (num_added)"""
        if num_added == 0 and len(self.cur_agents)==0:
            agent_profiles_df_list =[]

            max_iter = 1 # social_member_size / threshold
            for i in range(max_iter):
                agent_profiles_cur = self.call_manager_agent_func(
                    "add_and_return_user_profiles",
                    kwargs={
                        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                        "add_user_time_delta":self.time_configs["people_add_delta"].days,
                        "num_added":num_added
                    }
                ).content
                if len(agent_profiles_cur) =={}:break
                agent_profiles_df = pd.DataFrame.from_dict(agent_profiles_cur)
                agent_profiles_df_list.append(agent_profiles_df)
                print(f"updated {agent_profiles_df.shape[0]} agent profiles")
                
            agent_profiles = pd.concat(agent_profiles_df_list)

        else:
            agent_profiles_dfs = []
            step = num_added if num_added < 20 else 20
            step = 5 if step < 5 else step
            added_num = 0
            for i in range(0,num_added,step):
                add_num_per_round = step
                assert add_num_per_round <= 20, f"error add_num_per_round:{step}, {add_num_per_round}"
                agent_profiles_ = self.call_manager_agent_func(
                    "add_and_return_user_profiles",
                    kwargs={
                        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                        "add_user_time_delta":self.time_configs["people_add_delta"].days,
                        "num_added":add_num_per_round
                    }
                ).content
                if len(agent_profiles_) ==0:break
                agent_profiles_df = pd.DataFrame.from_dict(agent_profiles_)
                added_num += agent_profiles_df.shape[0]
                agent_profiles_dfs.append(agent_profiles_df)
                if added_num > num_added:
                    break
            if len(agent_profiles_dfs)==0:
                return
            agent_profiles = pd.concat(agent_profiles_dfs)

        self.call_manager_agent_func(
            "update_add_user_time",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d")
            }
        ).content
        
        agents_added = []
        if agent_profiles is None or agent_profiles.shape[0]==0: return

        for idx, agent_profile_info in enumerate(agent_profiles.iterrows()):
            index, agent_profile = agent_profile_info
            user_index = agent_profile["user_index"]
            # assert agent_profile["user_index"] not in self.cur_agents.keys(),\
            #       f"{index}, {user_index}"
            if user_index in self.cur_agents.keys():
                continue
            if self.to_dist:
                launcher_id = idx%len(self.launcher_args)
            else:
                launcher_id = 0
            agent = self.init_agent(agent_profile["user_index"],
                                   agent_profile.to_dict(),
                                   launcher_id = launcher_id)
            agents_added.append(agent)
            self.cur_agents[agent_profile["user_index"]] = agent
            print(f"Agent {agent_profile['user_index']} added")
        
        if denote_transitive_log:
            self.call_manager_agent_func("denote_transitive_log",
                                     kwargs={
                                         "delete_ids":delete_agent_ids,
                                         "add_ids":agent_profiles["user_index"].to_list()
                                     })
        
            
        

    def step(self):
        if self.time_configs["people_add_delta"] > timedelta(days=0):
            # 暂时不往网络内添加agent
            self.update_agents()
        
        """adopt llm generated plans"""
        agent_plans_map = self.collect_agent_plans()
        twitters = self.social(agent_plans_map)
        
        # update rating DB
        add_num = self.update_social_manager(twitters)
        self.call_manager_agent_func("update_big_name_list").content

        print(f"added {add_num} twitters for {self.time_configs['cur_time'].strftime('%Y-%m-%d')}")
        # update social/watcher DB and Time
        self.update_time()
        
        
    def update_social_manager(self,
                             twitters:list = []
                             ):
        
        num  = 0
        agent_ids = [*self.sampled_agent_ids["big_name"],
                     *self.sampled_agent_ids["common"]
                     ]
        zipped_agent_twitters = list(zip(twitters,agent_ids))
        
        for idx in range(0, len(zipped_agent_twitters), 100):
            zipped_agent_twitters_sub = zipped_agent_twitters[idx:idx+100]
            sub_add_msgs = []
            for twitters_one, agent_id in zipped_agent_twitters_sub:
                add_msg = self.call_manager_agent_func(
                    "add_tweets",
                    kwargs={
                        "agent_id":agent_id,
                        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                        "twitters":twitters_one
                    }
                )
                sub_add_msgs.append(add_msg)
            sub_add_num = [add_msg.content for add_msg in sub_add_msgs]
            num += sum(sub_add_num)        
        # self.call_manager_agent_func("update_docs").content
        return num
        
    
    def save(self, start_time):
        # self.social_manager.save_networkx_graph()
        self.call_manager_agent_func(
            "save_infos",
            kwargs={
                "cur_time": self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "start_time": start_time
            }
        ).content


    def initialize(self):
        if self.call_manager_agent_func("rerun").content:
            self.update_agents(denote_transitive_log=False)
            delete_agent_ids = self.call_manager_agent_func("return_deleted_agent_ids").content
            for cur_agent_id in delete_agent_ids:
                if cur_agent_id in self.cur_agents.keys():
                    self.cur_agents.pop(cur_agent_id)
                    print(f"Agent {cur_agent_id} deleted")
            time_s = self.time_configs["cur_time"] - \
                                                  self.time_configs["social_time_delta"] 
            self.call_manager_agent_func("rerun_set_time",
                                         kwargs={
                                            "last_added_time": time_s.strftime("%Y-%m-%d")
                                         }).content
            # self.time_configs["cur_time"] = self.time_configs["cur_time"] - \
            #     self.time_configs["social_time_delta"]
            
        else:
            self.update_agents()
                
        print("Finish Initialization")

    def test(self):
        """define unittest"""
        pass