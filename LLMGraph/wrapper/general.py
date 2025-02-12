from typing import Dict,Sequence,Union
from agentscope.agents import AgentBase
from .base import BaseAgentWrapper
from agentscope.message import Msg, PlaceholderMessage
import time
from LLMGraph.utils.str_process import remove_before_first_space

from LLMGraph.agent import GeneralAgent,ToolAgent
from agentscope.agents.rpc_agent import RpcAgentServerLauncher
import random


class GeneralAgentWrapper(BaseAgentWrapper):
    
    def __init__(self, 
                 name: str, 
                 agent:AgentBase,
                 manager_agent:AgentBase,
                 max_retrys: int = 3, 
                 max_tool_iters: int = 1, 
                 **kwargs) -> None:
        
        """tools: 需要已经to_dist后的agent"""
        
        
        super().__init__(name, agent, manager_agent, max_retrys, max_tool_iters, **kwargs)


    def reply(self, 
              message:Msg = None) -> Msg:
        
        func_name = message.get("func",False)
        func = getattr(self,func_name)
        kwargs = message.get("kwargs",{})
        func_res = func(**kwargs)
        
        return func_res


    def interact(self):
        """
        interact simulation
        """
        node_info = self.call_agent_func("get_node_info").content
        node_memory = self.call_agent_func("get_node_memory").content
        node_items = self.query(node_info, node_memory)
        actions = self.call_agent_func(func_name="action",
                        kwargs={
                            "node_info":node_info,
                            "node_memory":node_memory,
                            "node_items":node_items,
                        }).content
        return Msg(
            name = self.name,
            content = actions,
            role = "assistant"
        )
    
        
    def query(self,
            node_info,
            node_memory):
        """
        query simulation
        """
        agent_msgs = self.call_agent_get_prompt(func_name="query",
                                        kwargs={
                                            "node_info":node_info,
                                            "node_memory":node_memory
                                        }).content

        response = self.step(agent_msgs=agent_msgs,
            use_tools=True,
            return_intermediate_steps=True,
            return_tool_exec_only=True)

        intermediate_steps = response.get("intermediate_steps",[])
        template = """
Here's some information you get from searching:

{searched_infos}

The end of searched information.       
"""
        searched_infos = ""
        for intermediate_step in intermediate_steps:
            action, observation = intermediate_step
            searched_infos += observation.get("result","") + "\n"
        return template.format(searched_infos = 
                            searched_infos)


    def get_agent_memory_msgs(self):
        return self.call_agent_func(
            "get_short_memory"
        )