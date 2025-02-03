from typing import Any, Dict, List

from pydantic import BaseModel

from LLMGraph.message import Message
from pydantic import BaseModel

from . import env_registry as EnvironmentRegistry
from agentscope.message import Msg
from agentscope.agents.rpc_agent import RpcAgentServerLauncher
from agentscope.agents import AgentBase

from LLMGraph.wrapper import BaseAgentWrapper

@EnvironmentRegistry.register("base")
class BaseEnvironment(BaseModel):
    to_dist: bool = False
    launcher_args:list = []
    manager_agent: AgentBase
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)

    def call_manager_agent_func(self,
                                func_name:str,
                                kwargs:dict ={}) -> Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name,
                )
        return_msg = self.manager_agent(msg)
        return return_msg
    
    def call_agent_func(self,
                        agent:BaseAgentWrapper,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name
                )
        return_msg = agent(msg)
        return return_msg
    

    # functions to be defined
    def initialize(self):
        pass

    def step(self):
        pass
