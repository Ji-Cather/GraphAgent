from typing import Sequence,Union

from LLMGraph.output_parser import social_output_parser_registry
from LLMGraph.prompt.social import social_prompt_registry

import json
from agentscope.message import Msg, PlaceholderMessage
from .. import agent_registry
from LLMGraph.agent.base_agent import BaseGraphAgent
from LLMGraph.prompt import MODEL
from LLMGraph.utils.count import count_prompt_len
from LLMGraph.memory import social_memory_registry
from LLMGraph.utils.count import select_to_last_period
import time
import pandas as pd

def remove_before_first_space(s:str):
    # 分割字符串，最大分割次数设为1
    parts = s.split(' ', 1)
    # 如果字符串中有空格，则返回第一个空格后的所有内容
    if len(parts) > 1:
        return parts[1].replace("\"","")
    else:
        # 如果没有空格，返回整个字符串
        return s.replace("\"","")


@agent_registry.register("social_agent")
class SocialAgent(BaseGraphAgent):
   
    infos : dict 

    mode : str = "choose_topic" # 控制llm_chain 状态（reset_state中改）
   
    
    def __init__(self, 
                 name,
                 infos,
                agent_configs,
                memory_init_kwargs:dict={},
                **kwargs):
        
        llm_config_name = agent_configs.get('llm').get("config_name")
        init_mode = "choose_topic"
        
        prompt = social_prompt_registry.build(init_mode)
        output_parser = social_output_parser_registry.build(init_mode)
        self.infos = infos

        memory_config = agent_configs.get('memory')
        memory_config["name"] = name
        self.action_memory = social_memory_registry.build(
            **memory_config,
            **memory_init_kwargs
        )
        super().__init__(name,
                         prompt_template=prompt,
                         output_parser=output_parser,
                         model_config_name=llm_config_name,
                         use_memory=False, # ban default memory
                         **kwargs)
    
    
    def reset_state(self,
                    mode = "forum_action"):
       
        if self.mode == mode : return
        self.mode = mode
        
        prompt = social_prompt_registry.build(mode)
        output_parser = social_output_parser_registry.build(mode)
        self.prompt_template = prompt
        self.output_parser = output_parser
    
    
    def get_acting_plan(self,
                        role_description:str,
                        big_name:bool = False,
                        update_threshold:int = 30):
        self.reset_state("forum_action_plan")
        if self.action_memory.action_plan["last_update_time"] < update_threshold:
            return Msg(self.name,
            self.action_memory.action_plan,
            "assistant"
        )
    

        action_history_hint = self.action_memory.get_action_history(
            action_days = True
        )

        respond_instructions = {
            "big_name":"As a social celebrity, \
And you should increse your log in days to twitter so as to increase your social media influence.\
You should at least be active on social media for 15-30 days a month.\
you now need to provide your next Twitter activity frequency plan in the following format:",
            "common":"As a ordinary user, \
you now need to provide your next Twitter activity frequency \
plan in the following format:"
        }

        prompt_inputs = {
            "role_description":role_description,
            "action_history": action_history_hint,
            "respond_instruction": respond_instructions.get(
                "big_name" if big_name else "common")
        }

        response = self.step(prompt_inputs = prompt_inputs).content
        try:
            action_plan = response.get("return_values",{}).get("action_plan",{})
        except:
            action_plan = {}
        self.action_memory.update_action_plan(action_plan)
        if big_name:
            log_in_days = self.action_memory.action_plan["log_in_days"]
            self.action_memory.action_plan["log_in_days"] = 20 if log_in_days < 20 else log_in_days
        return Msg(self.name,
            self.action_memory.action_plan,
            "assistant"
        )
    

        
    def forum_action(self,
                    role_description:str,
                    friend_data:str,
                    twitter_data:str,
                    num_followers:int):
        """这个100秒"""
        self.reset_state("forum_action")
        time_start = time.time()
        time_start_all = time_start
        time_cache = {}

        template = """
twitter data template:
{{tweet_idx}}:
    user: {{user_name}}
    topic: {{topic}}
    tweet: {{page_content}}
        

The current Twitter data is:


{searched_infos}



The end of Twitter data.       
"""
        searched_infos = select_to_last_period(
            twitter_data,
            4e3
        )
        time_cache["searched_infos"] = time.time()-time_start
        time_start = time.time()

        twitter_infos = template.format(searched_infos = 
                               searched_infos)

        forum_action_hint = self.action_memory.get_forum_action_hint()
        prompt_inputs = {
            "role_description":role_description,
            "twitter_data":twitter_infos,
            "memory": self.action_memory.retrieve_tweets_memory(upper_token=1e3),
            "friend_data": friend_data,
            "num_followers": num_followers,
            "forum_action_hint": forum_action_hint
        }
        time_cache["prompt_inputs"] = time.time()-time_start
        time_start = time.time()
        response = self.step(prompt_inputs = prompt_inputs).content

        time_cache["step"] = time.time()-time_start
        time_start = time.time()
        self.action_memory.add_message(searched_infos)

        try:
            actions = response.get("return_values",{}).get("actions",[])
        except:
            actions = []
        if actions is None:
            actions = []
        

        time_all_end = time.time()-time_start_all

        time_cache["all_process"] = time_all_end
        return Msg(name=self.name, 
                    role="assistant",
                    content = actions)
    
    def forum_action_bigname(self,
                    role_description:str,
                    friend_data:str,
                    twitter_data:str,
                    num_followers:int):
        self.reset_state("forum_action_bigname")
        
        template = """
twitter data template:
{{tweet_idx}}:
    user: {{user_name}}
    topic: {{topic}}
    tweet: {{page_content}}

The current Twitter data is:


{searched_infos}


The end of Twitter data.       
"""
        searched_infos = select_to_last_period(
            twitter_data,
            4e3
        )
        twitter_infos = template.format(searched_infos = 
                               searched_infos)
        forum_action_hint = self.action_memory.get_forum_action_hint()

        prompt_inputs = {
            "role_description":select_to_last_period(role_description,5e2),
            "twitter_data":twitter_infos,
            "memory": self.action_memory.retrieve_tweets_memory(upper_token=1e3),
            "friend_data":  select_to_last_period(friend_data,2e2),
            "num_followers": num_followers,
            "forum_action_hint":  select_to_last_period(forum_action_hint,5e2)
        }
        response = self.step(prompt_inputs = prompt_inputs).content
        self.action_memory.add_message(searched_infos)

        
        try:
            actions = response.get("return_values",{}).get("actions",[])
        except:
            actions = []        
        if actions is None:
            actions = []
        return Msg(name=self.name, 
                    role="assistant",
                    content = actions)
        
        
    def get_twitter_search_batch(self,
                                role_description,
                                    ):
        
        
        self.reset_state("search_forum")
        
        keywords = self.action_memory.get_searched_keywords()
        keywords_ndv = list(pd.Series(keywords).value_counts().to_dict().keys())
        
        topk = 5
        keywords_top = keywords_ndv[:topk]
        keyword_template = """
You often search for the following keywords:

[{keywords}]
"""

        searched_info = keyword_template.format(keywords = 
                            ",".join(keywords_top))
        prompt_inputs = {
            "role_description": role_description,
            # "memory":self.action_memory.retrieve_tweets_memory(
            #     upper_token=4e3
            # ),
            "memory":"", # 这个会导致问题
            "searched_info":searched_info
        }
        return prompt_inputs
        
    def return_interested_topics(self):
        interested_topics = self.action_memory.get_interested_topics()
        return Msg(
            self.name, 
            role="assistant",
            content = interested_topics
        )


    def observe(self, messages:Union[Msg]=None):
        if not isinstance(messages,Sequence):
            messages = [messages]
        for message in messages:
            if isinstance(message,PlaceholderMessage):
                message.update_value()
            # if not isinstance(message,Msg):continue
            if message.content == "":
                continue
            if message.name == self.name or message.name == "system":
                self.short_memory.add(messages)
                break

    def add_tweets(self,
                   tweet_content:str,
                   type = "seen"):
        """type: seen/ posted"""
        self.action_memory.add_tweets(
            tweet_content,
            type
        )

    def add_topics(self,
                   topics:list,
                   type = "posted"):
        """type: posted"""
        self.action_memory.add_topics(
            topics,
            type
        )

    def add_actions(self,
                    actions:list):
        self.action_memory.add_actions(
            actions
        )

    def add_twitter_action_memory(self,
                                  forum_actions:list =[]):
        posted_topics = [
            action.get("topic","default")
            for action in forum_actions
        ]

        posted_tweets = list(filter(lambda x: x["action"].lower() == "tweet", 
                               forum_actions))


        posted_tweets_str = "\n\n".join([tweet.get("input","") 
                                       for tweet in posted_tweets])

        actions = [forum_action["action"] for forum_action in forum_actions]
        
        self.add_actions(actions)
        self.add_topics(posted_topics, "posted")
        self.add_tweets(posted_tweets_str, "posted")

    def add_searched_keywords(self,
                              keywords:list):
        self.action_memory.add_searched_keywords(
            keywords
        )
    
    def get_searched_keywords(self):
        return self.action_memory.get_searched_keywords()