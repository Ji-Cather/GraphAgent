from LLMGraph.agent.tool_agent import ToolAgent
from LLMGraph.manager import GeneralManager
from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from LLMGraph.tool import create_general_retriever_tool
from typing import List,Dict,Union,Any
from datetime import datetime

from LLMGraph.agent.base_manager import ManagerAgent
import copy
from agentscope.models import ModelWrapperBase

class GeneralManagerAgent(ManagerAgent):
    """
    The manager agent is responsible for managing the whole process of writing
    a paper. It is responsible for the following tasks:
    1. Generating prompts for the tools.
    2. Generating prompts for the authors.
    3. Generating prompts for the paper.
    4. Writing the paper.
    5. Updating the database.
    Args:
        name (str): The name of the agent.
        model_config_name (str): The name of the model config.
        task_path (str): The path of the task.
        config_path (str): The path of the config.
        cur_time (str): The current time.
        article_write_configs (dict): The configs for writing the paper.
    Attributes:
        name (str): The name of the agent.
        model_config_name (str): The name of the model config.
        task_path (str): The path of the task.
        config_path (str): The path of the config.
        cur_time (str): The current time.
        article_write_configs (dict): The configs for writing the paper.
        tools_prompt (str): The prompt for the tools.
        func_name_mapping: func_map_dict 
        article_manager: manage paper DB.
    """
    general_manager:GeneralManager
    
    def __init__(self,
                 name,
                general_manager_configs,
                model_config_name,
                task_path,                
                config_path,
                time_configs,
                 ):
        
        super().__init__(name = name,
                         model_config_name=model_config_name)
        

        

        general_manager = GeneralManager.load_data(
            dataset_name=general_manager_configs.get("dataset_name"),
            task_path=task_path,
            config_path=config_path,
            retriever_kwargs=general_manager_configs.get("retriever_kwargs"),
            time_configs=time_configs,
            llm = self.model,
            graph_structure = general_manager_configs.get("graph_structure"),
            embedding_model_name = general_manager_configs.get("embedding_model_name"),
            general_memory_config = general_manager_configs.get("general_memory_config"),
        )
        
        self.general_manager = general_manager
        self.tools = {}
        
        self.document_prompt = PromptTemplate.from_template("""
Node ID: {node_id}
Node Type: {node_type}
Node Info: {page_content}
""")
        
        self.update()




    def reply(self, message:Msg=None):
        
        func_name_map = {
           "self":["update",
                   "call_tool_func",
                   "get_prompt_tool_msgs",
                   ],
            "manager":[
                "update_interactions",
                "save"
            ]
        }
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        
        if func_name in func_name_map["self"]:
            func = getattr(self,func_name)
        else:
            func = getattr(self.general_manager,func_name)

        return_values = func(**kwargs)
        if return_values is None:
            return_values = ""
        return Msg(self.name,content = return_values,role = "assistant")
     
    def update(self,
               ): 
        # update after one round of simulation
        time_configs = self.general_manager.update_retriver()
        retriever = self.general_manager.retriever
        retriever_tool_func, retriever_tool_dict = create_general_retriever_tool(
            retriever,
            "search",
    "Search for relevant items so as to interact with them! Try to clarify your query step by step.",
            document_prompt = self.document_prompt,
            )
        
        tools = {
        "search":ToolAgent("search",
                    tool_func=retriever_tool_func,
                    func_desc=retriever_tool_dict
                    ),          
        }
        self.update_tools(tools)
    

    
    def call_manager_func(self,func_name,kwargs):
        func = getattr(self.general_manager,func_name)
        func_res = func(**kwargs)
        return func_res

    
    