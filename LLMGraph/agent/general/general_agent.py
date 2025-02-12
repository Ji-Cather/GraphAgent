# from LLMGraph.manager import ForumManager
# from LLMGraph.involvers import System,Tool,search_forum_topk
from LLMGraph.message import Message
from LLMGraph.prompt.general import general_prompt_registry

from LLMGraph.output_parser import general_output_parser_registry

# from langchain.agents import AgentExecutor
from typing import Any, List, Optional, Tuple, Union,Dict,Sequence

from LLMGraph.memory import GeneralMemory
import random
import copy
from agentscope.message import Msg,PlaceholderMessage
from LLMGraph.message import Message
from .. import agent_registry

from LLMGraph.agent.base_agent import BaseGraphAgent

import json
from LLMGraph import MODEL
from LLMGraph.utils.count import count_prompt_len

@agent_registry.register("general_agent")
class GeneralAgent(BaseGraphAgent):
    
    long_memory:GeneralMemory

    def __init__(self, 
                 name,
                 agent_configs:dict,
                 agent_info:str,
                 agent_memory:str,
                 dataset_name,
                **kwargs):
        
        agent_configs = copy.deepcopy(agent_configs)
        agent_llm_config = agent_configs.get('llm')
        
        init_mode = f"interact_{dataset_name}_action"
        prompt_template = general_prompt_registry.build(init_mode)
        output_parser = general_output_parser_registry.build(init_mode)

        model_config_name = agent_llm_config["config_name"]
        # TBD: memory update
        long_memory_config = agent_configs.get('long_memory')
        
        long_memory = GeneralMemory(name = name)       
        long_memory.update([Message(name=name,content=agent_memory)])     
        self.long_memory = long_memory
        self.mode = init_mode
        self.dataset_name = dataset_name
        self.agent_info = agent_info
        # memory = ActionHistoryMemory(llm=kwargs.get("llm",OpenAI()))
        super().__init__(name = name,
                         prompt_template=prompt_template,
                         output_parser=output_parser,
                         model_config_name=model_config_name,
                         **kwargs)

    
    class Config:
        arbitrary_types_allowed = True
  
    def clear_discussion_cur(self):
        self.social_memory.clear_discussion_cur()
        
    
    def reset_state(self,
                    mode = "default",
                    ):
       
        if self.mode == mode : return
        self.mode = mode
        
        prompt = general_prompt_registry.build(mode)
        output_parser = general_output_parser_registry.build(mode)
        self.prompt_template = prompt
        self.output_parser = output_parser

    def observe(self, messages:List[Message]=None):
        if not isinstance(messages,Sequence):
            messages = [messages]
        for message in messages:
            if isinstance(message,PlaceholderMessage):
                message.update_value()
            # if not isinstance(message,Msg):continue
            if message.content == "":
                continue
            if message.get("long_term"):
                self.long_memory.add_message(messages=messages)
                break
            elif message.name == self.name or message.name == "system":
                self.short_memory.add(messages)
                break
       
    def get_node_info(self):
        return Msg(self.name, content=self.agent_info, role = "assistant")
    
    def get_node_memory(self):
        mem = self.long_memory.retrieve_recent_chat(upper_token=2e3)
        return Msg(self.name,content=mem, role = "assistant")

    def query(self,
             node_info:str,
             node_memory:str):
        mode = f"interact_{self.dataset_name}_query"
        self.reset_state(mode=mode)
        prompt_inputs = {
            "node_info": node_info,
            "node_memory": node_memory,
        }
        return prompt_inputs
    
    def action(self,
               node_info:str,
               node_memory:str,
               node_items:str):
        mode = f"interact_{self.dataset_name}_action"
        self.reset_state(mode=mode)
        prompt_inputs = {
            "node_info": node_info,
            "node_memory": node_memory,
            "node_items": node_items,
        }
        response = self.step(prompt_inputs)

        return response