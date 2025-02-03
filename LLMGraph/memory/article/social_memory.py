from typing import List, Sequence,Union,Dict

from pydantic import Field

from LLMGraph.message import Message

from . import article_memory_registry
from LLMGraph.memory.base import BaseMemory

from pydantic import BaseModel
import re
from typing import Any, Dict, List, Tuple, Type, Union
import random


SUMMARY_PROMPT = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""


from agentscope.message import Msg


from .. import select_to_last_period



@article_memory_registry.register("social_memory")
class SocialMemory(BaseMemory):

    
    reflection:bool = False # 若设置为true,则触发分类reflection  
    
    social_network: dict = {} # id:{name:,view:,dialogues：List[Message],chat_history, dialogues_pt}
    
    # 设置各类消息的buffer大小数
    summary_threshold:int = 5 # 每次总结后，再多5条就触发一次总结 -> 记忆库内
    
    discussion_cur_paper:list = [] # 在每次进行本轮communication之前清空，存储所有自己发出的讨论
    
    def __init__(self,**kwargs):
        social_network = kwargs.pop("social_network")
        # assert len(social_network)>=1
        for id,info in social_network.items():
            info["dialogues_pt"] = -1       
            info["chat_history"] = ""
            info["comment"] = ""
            info["dialogues"] = []

        
        super().__init__(social_network = social_network,
                         **kwargs)
    
    def clear_discussion_cur(self):
        self.discussion_cur_paper = []
    
    def add_message(self, 
                    messages: List[Msg]):
        if not isinstance(messages,list):
            messages =[messages]
        for message in messages:
            if message.name == self.id:
                # receive_ids = list(self.social_network.keys())
                self.discussion_cur_paper.append(message)
                continue
            else:
                receive_ids = [message.name]
            for receive_id in receive_ids:
                if receive_id not in self.social_network.keys():
                    self.social_network[receive_id] = {"name": message.name,
                                                        "relation":"stranger",
                                                        "dialogues":[],
                                                        "dialogues_pt":-1,
                                                        "chat_history":"",
                                                        "comment":""}
                    
                if "dialogues" in self.social_network[receive_id].keys():
                    self.social_network[receive_id]["dialogues"].append(message)
                else:
                    self.social_network[receive_id]["dialogues"] = [message]
                    self.social_network[receive_id]["dialogues_pt"] = -1
                
        
    def topk_message_default(self,
                             messages:List[Msg],
                             k=5)->List[Msg]:
        messages.sort(key=lambda x: x.sort_rate(),reverse=True)
        return messages[:k] if k<len(messages) else messages
    
 

    def to_string(self, 
                  messages:List[Msg],
                  add_sender_prefix: bool = False,
                  ) -> str:
        if add_sender_prefix:
            return "\n".join(
                [
                    message.to_str()
                    for message in messages
                ]
            )
        else:
            return "\n".join([message.content for message in messages])
  

    def reset(self) -> None:
        for id,info in self.social_network.items():
            info["dialogues_pt"] = -1       
            info["chat_history"] = ""
            info["comment"] = ""
            info["dialogues"] = []


    ###############         一系列的 retrive memory rule       ##############
    
    #  调用各类 retrive 方法
        
    def retrieve_recent_chat(self,
                             agent_ids: Union[List,str] = "all",
                             upper_token = 4e3):
        assert agent_ids is not None
        if agent_ids == "all":
            agent_ids = list(self.social_network.keys())
       
        if isinstance(agent_ids,str):
            agent_ids = [agent_ids]
        
        recent_chats = []
        for agent_id in agent_ids:
            # if agent_id not in self.social_network.keys():
            #     continue
            if ("dialogues" in self.social_network[agent_id].keys()):
                dialogues_sn = self.social_network[agent_id].get("dialogues")
                dialogues_sn = reversed(dialogues_sn)
                recent_chats.append(self.to_string(dialogues_sn))

        recent_chats.append(self.to_string(self.discussion_cur_paper))

        chats = "\n".join(recent_chats)
        if len(chats) > upper_token:
            return select_to_last_period(chats, upper_token)
        return chats
    
    def get_researchers_infos(self):
        researchers = []
        if len(self.social_network) >10:
            sn_filtered = random.sample(self.social_network.keys(),10)
            sn_filtered ={
                id:self.social_network[id] for id in sn_filtered
            }
        else:
            sn_filtered = self.social_network
        for id, researcher in sn_filtered.items():
            expertises = researcher["expertises"]
            expertises = expertises[:100]
            topics = researcher["topics"]
            name = researcher["name"]
            expertises = ",".join(expertises)
            topics = ",".join(topics)
            researcher_info = f"{id}. {name}: {expertises}, interested in {topics}"
            researchers.append(researcher_info)
            
        return "\n".join(researchers)
    

            
    
    
    
    
   
     