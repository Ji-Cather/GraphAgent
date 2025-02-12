from pydantic import BaseModel, Field
from typing import List, Tuple, Set


from datetime import datetime
from pydantic import BaseModel
from typing import Union

from agentscope.message import Msg



# 这里导致prompt template返回类型有问题
class Message(Msg):
    def __init__(self,
                message_type: str = "",
                receivers:list = [],
                output_keys: List[str] = [],
                importance_rate: float = 0,
                relation_rate :float = 0,
                tool_response: list = [],
                conver_num:int = 0, #记录对话次数
                continue_dialogue : bool = True, # 记录对话是否继续
                **kwargs
                ) -> None:
        self.receivers = receivers
        self.message_type = message_type
        self.output_keys = output_keys
        self.importance_rate = importance_rate
        self.relation_rate = relation_rate
        self.tool_response = tool_response
        self.conver_num = conver_num
        self.continue_dialogue = continue_dialogue
        super().__init__(**kwargs)

    @classmethod
    def from_msg(cls, msg: Msg,**kwargs):
        return cls(**kwargs,
                   name = msg.name,
                   content = msg.content,
                   role = msg.role,
                   url = msg.url,
                   timestamp = msg.timestamp)
    
    def update_attr(self,**kwargs):
        for key,value in kwargs.items():
            self.__setattr__(key,value)
    
        
    def sort_rate(self):
        return self.timestamp+self.importance_rate+self.relation_rate

    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"
    
    def __str__(self):
        if isinstance(self.content,dict):
            content_str = [f"{key.capitalize()}:{self.content[key]}"for key in self.output_keys]
            return "\n".join(content_str)
        elif isinstance(self.content,str):
            return self.content
        else:
            raise ValueError()
        