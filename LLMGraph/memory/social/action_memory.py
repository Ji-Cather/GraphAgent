from . import social_memory_registry
from LLMGraph.memory.base import BaseMemory
from typing import List, Sequence,Union,Dict


from agentscope.message import Msg
from .. import select_to_last_period
import random


from collections import Counter

def sort_by_frequency_and_value(items):
    # 使用Counter对象统计每个元素出现的次数
    count = Counter(items)
    
    # 按照出现次数进行排序，对于出现次数相同的元素，按照元素的值进行排序
    sorted_items = sorted(count, key=lambda x: (count[x], x))
    
    return sorted_items

@social_memory_registry.register("action_memory")
class ActionMemory(BaseMemory):

    seen_tweets:List[str] = []
    posted_tweets:List[str] = []
    topic_memory:dict = {
        "posted_topics":[],
        "followed_topics":[]
    } 
    action_counts:dict = {}
    searched_keywords:list = []
    action_plan:dict = {
        "tweet":0.2,
        "reply":0.6,
        "retweet":0.2,
        "log_in_days":10,
        "last_update_time":100, # 保证第一次更新
    }
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
       
   
    def add_topics(self,
                   topics:Sequence[str],
                   type = "posted"):
        if type == "posted":
            self.topic_memory["posted_topics"].extend(topics)
       

    def add_tweets(self, 
                    tweet_content:str,
                    type = "seen"):
        tweets = tweet_content.split("\n\n")
        if type == "seen":
            self.seen_tweets.extend(tweets)
        elif type == "posted":
            self.posted_tweets.extend(tweets)

    def add_actions(self,
                    actions:List[str]):
        action_counter = Counter(actions)
        for action, count in action_counter.items():
            if action not in self.action_counts:
                self.action_counts[action.lower()] = 0
            self.action_counts[action.lower()] += count
    
    def retrieve_tweets_memory(self,
                        upper_token = 1e3):
        
        template = """
Here's some tweets that you have seen before:
{tweets}
"""
        tweets = self.seen_tweets[-10:]
        if len(tweets) == 0:
            return ""
        tweets_content = "\n\n".join(tweets)
        tweets_content = select_to_last_period(tweets_content,upper_token)
        memory = template.format(tweets=tweets_content)
        return memory
    

    def get_interested_topics(self):
        topics = []
        for k,v in self.topic_memory.items():
            topics.extend(v)
        if len(topics) > 100:
            topics = random.sample(topics,100)
        topics_ndv = sort_by_frequency_and_value(topics)
        return topics_ndv
    

    def get_action_history(self,
                           action_days = False):
        forum_action_template = """
You have executed Action Tweet for {tweet} times, \
Action Reply for {reply} times, \
Action Retweet for {retweet} times. 
"""
        action_days_template = """
And you log in twitter {log_in_days} a month
And your previous activity frequency on Twitter was: 
{{
"tweet": {tweet},
"retweet": {retweet},
"reply": {reply},
"log_in_days": {log_in_days}
}}

"""

        forum_action_hint = forum_action_template.format(
            tweet=self.action_counts.get("tweet",0),
            reply=self.action_counts.get("reply",0),
            retweet=self.action_counts.get("retweet",0)
        )
        action_days_hint = action_days_template.format_map(self.action_plan)
        if not action_days:
            return forum_action_hint
        return forum_action_hint +"\n" +action_days_hint

    def get_forum_action_hint(self):
        forum_action_hint = self.get_action_history(
            action_days=True
        )
        self.action_plan["last_update_time"] += 1
        hint = "You should {do_action} more tweets, avoid {not_do_action}."
        
        sum_action = sum(self.action_counts.values())
        if sum_action == 0:
            return ""
        action_threshold_map = {
            "tweet": int(self.action_plan.get("tweet",0.2) * sum_action),
            "reply": int(self.action_plan.get("reply",0.2) * sum_action),
            "retweet": int(self.action_plan.get("retweet",0.6) * sum_action)
        }
        do_actions = []
        not_do_actions = []
        for action, count in self.action_counts.items():
            if count >= action_threshold_map[action]:
                do_actions.append(action)
            else:
                not_do_actions.append(action)

        
        do_actions = ",".join(do_actions)
        not_do_actions = ",".join(not_do_actions)
        hint = hint.format(do_action=do_actions, not_do_action=not_do_actions)
        return forum_action_hint + "\n" +hint
    
    def update_action_plan(self,
                           action_plan:dict):
        self.action_plan["last_update_time"] = 0

        type_converter = {
            "tweet":float,
            "reply":float,
            "retweet":float,
            "log_in_days":int,
            "last_update_time":int # 保证第一次更新
        }
        try:
            self.action_plan.update(action_plan)
            self.action_plan = {k:type_converter[k](v) for k,v in self.action_plan.items()}
        except Exception as e:
            pass

    def add_searched_keywords(self,keywords:List[str]):
        self.searched_keywords.extend(keywords)

    def get_searched_keywords(self):
        """to avoid too many keywords"""
        searched_keywords = self.searched_keywords
        if len(searched_keywords) > 100:
            searched_keywords = random.sample(searched_keywords,100)
        return searched_keywords