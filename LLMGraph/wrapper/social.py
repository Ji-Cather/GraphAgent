from typing import Dict,Sequence, Union
from agentscope.agents import AgentBase
from .base import BaseAgentWrapper
from agentscope.message import Msg, PlaceholderMessage
import time
from LLMGraph.utils.str_process import remove_before_first_space

from datetime import datetime


class SocialAgentWrapper(BaseAgentWrapper):
    
    def __init__(self, 
                 name: Union[str,int], 
                 agent:AgentBase,
                 manager_agent:AgentBase,
                 max_retrys: int = 3, 
                 max_tool_iters: int = 2,
                 to_dist = False, 
                 **kwargs) -> None:
        
        """tools: 需要已经to_dist后的agent"""
        
        
        super().__init__(name, 
                         agent, 
                         manager_agent, 
                         max_retrys, 
                         max_tool_iters,
                         to_dist = to_dist,
                          **kwargs)


    def reply(self, message:Msg = None) -> Msg:
        available_func_names = ["twitter_process",
                                "get_acting_plan"]
        func_name = message.get("func")
        assert func_name in available_func_names, f"{func_name} not in {available_func_names}"
        func = getattr(self,func_name)
        kwargs = message.get("kwargs",{})
        func_res = func(**kwargs)
        return func_res
        
    

    def twitter_process(self,
                        cur_time:str,
                        big_name:bool = False):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        follow_content = self.call_manager_func(
                "get_follow_ids",
                kwargs={
                    "agent_id":self.name,
                }
            ).content
        
        twitter_infos = self.get_twitter_search_batch(follow_content)
        
        if twitter_infos == "":
            return Msg(
            self.name,
            content=[],
            role="assistant"
        )

        if big_name:
            forum_actions = self.forum_action_bigname(
                twitter_infos=twitter_infos,
                )
        else:
            forum_actions = self.forum_action(
                twitter_infos=twitter_infos,
               )
        available_actions = ["tweet","retweet","reply"]
        forum_actions_filtered = []
        for forum_action in forum_actions:
            if isinstance(forum_action,dict):
                try:
                    if forum_action.get("action","").lower() in available_actions:
                        forum_actions_filtered.append(forum_action)
                except:
                    pass
        self.call_agent_func("add_twitter_action_memory",
                    kwargs={
                        "forum_actions":forum_actions_filtered
                    }).content

        return Msg(
            self.name,
            content=forum_actions_filtered,
            role="assistant"
        )
        

    def forum_action(self, 
                     twitter_infos:str) -> Msg:
        time_cache = {}
        time_start = time.time()
        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content
        friend_data = self.call_manager_func(
            "get_user_friend_info",
            kwargs={"user_index":self.name}            
        ).content
        
        time_cache["friend_data"] = time.time()-time_start
        time_start = time.time()

        num_followers = self.call_manager_func(
            "get_user_num_followers",
            kwargs={"user_index":self.name}            
        ).content

        time_cache["num_followers"] = time.time()-time_start
        time_start = time.time()

        forum_msg_content = self.call_agent_func("forum_action",
            kwargs={
                "role_description":role_description,
                "friend_data":friend_data,
                "twitter_data":twitter_infos,
                "num_followers":num_followers
            }).content
        
        assert forum_msg_content is not None
        time_all_end = time.time()-time_start
        time_cache["forum_action"] = time_all_end
        return forum_msg_content
    
    def forum_action_bigname(self, twitter_infos:str) -> Msg:
        time_start = time.time()
        time_start_all = time_start
        time_cache = {}

        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content
        
        friend_data = self.call_manager_func(
            "get_user_friend_info",
            kwargs={"user_index":self.name}            
        ).content
        time_cache["friend_data"] = time.time()-time_start
        time_start = time.time()
        
        num_followers = self.call_manager_func(
            "get_user_num_followers",
            kwargs={"user_index":self.name}            
        ).content
        time_cache["num_followers"] = time.time()-time_start
        time_start = time.time()

        forum_msg_content = self.call_agent_func("forum_action_bigname",
            kwargs={
                "role_description":role_description,
                "friend_data":friend_data,
                "twitter_data":twitter_infos,
                "num_followers":num_followers
            }).content
        assert forum_msg_content is not None
        time_all = time.time() - time_start_all
        time_cache["forum_action_bigname"] = time_all

        return forum_msg_content

    def get_twitter_search_batch(self,
                                 follow_content: dict = {},
                                 ) -> str:
        time_start_all = time.time()
        time_start = time_start_all
        time_cache = {}
        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content
        
        # update tool with follow content


        agent_msgs = self.call_agent_get_prompt(            
            "get_twitter_search_batch",
            kwargs={
                "role_description":role_description,
            }).content
        time_cache["prompt_inputs"] = time.time()-time_start
        time_start = time.time()

        interested_topics = self.call_agent_func(
            "return_interested_topics"
        ).content
        time_cache["interested_topics"] = time.time()-time_start
        time_start = time.time()

        self.call_manager_func("update",
                               kwargs={
                                    "social_follow_map":follow_content,
                                    "interested_topics":interested_topics,
                                })
        time_cache["update_tool"] = time.time()-time_start
        time_start = time.time()

        response = self.step(agent_msgs=agent_msgs,
                             use_tools=True,
                             return_tool_exec_only=True,
                             return_intermediate_steps=True)
        time_cache["step"] = time.time()-time_start
        time_start = time.time()
        intermediate_steps = response.get("intermediate_steps",[])
        if intermediate_steps== []:
            return ""

        searched_infos = ""
        searched_keywords = []
        for intermediate_step in intermediate_steps:
            action, observation = intermediate_step
            result = observation.get("result",{})
            keyword = action.get("kwargs",{}).get("keyword","")
            searched_keywords.append(keyword)
            try:
                searched_infos += result.get("output","")+ "\n"
            except:
                pass
        self.call_agent_func(
            "add_tweets",
            kwargs={
                "tweet_content": searched_infos,
                "type":"seen"
            }
        ).content

        time_cache["add_tweets"] = time.time()-time_start
        time_start = time.time()

        self.call_agent_func(
            "add_searched_keywords",
            kwargs={
                "keywords": searched_keywords
            }
        ).content
        
        time_all = time.time()-time_start_all
        time_cache["add_tweets"] = time_all
        return searched_infos
    

    def get_agent_memory_msgs(self):
        return self.call_agent_func(
            "get_short_memory"
        )
    
    def get_acting_plan(self):
        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content

        big_name = self.call_manager_func(
            "get_user_big_name",
            kwargs={"user_index":self.name}            
        ).content

        plan = self.call_agent_func(
            "get_acting_plan",
            kwargs={
                "role_description":role_description,
                "big_name":big_name
            }
        ).content

        return Msg(            
            self.name,
            content=plan,
            role="assistant"
        )