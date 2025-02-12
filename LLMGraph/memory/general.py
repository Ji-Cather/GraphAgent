from abc import abstractmethod
from typing import Dict, List

from LLMGraph.message import Message
from . import memory_registry, select_to_last_period
from .base import BaseMemory


@memory_registry.register("general_memory")
class GeneralMemory(BaseMemory):
    memory:List[Message] = [] # action tragectory

    class Config:
        arbitrary_types_allowed = True
    
    def update(self, messages: List[Message]) -> None:
        self.memory.extend(messages)

    def reset(self) -> None:
        self.memory = []

    def to_string(self) -> str:
        return "\n".join([str(message) for message in self.memory])
    
    def retrieve_recent_chat(self,
                             upper_token = 4e3):
        recent_chats = self.to_string()
        if len(recent_chats) > upper_token:
            return select_to_last_period(recent_chats, upper_token)
        return chats
    