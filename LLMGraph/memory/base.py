from abc import abstractmethod
from typing import Dict, List, Any

from pydantic import BaseModel, Field
from typing import Any

from LLMGraph.message import Message
from . import memory_registry


@memory_registry.register("base")
class BaseMemory(BaseModel):
    name: Any # 标记是谁的记忆
    id:Any = None# 标记是谁的记忆
    class Config:
        arbitrary_types_allowed = True

    def add_message(self, messages: List[Message]) -> None:
        pass


    def to_string(self) -> str:
        pass


    def reset(self) -> None:
        pass


