from LLMGraph.agent.tool_agent import ToolAgent
from LLMGraph.agent.base_manager import ManagerAgent
from LLMGraph.manager import SocialManager
from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from LLMGraph.tool import create_forum_retriever_tool
from typing import List,Dict,Union
from datetime import datetime
from .. import manager_agent_registry


@manager_agent_registry.register("social_manager")
class SocialManagerAgent(ManagerAgent):
    
    social_manager:SocialManager

    def __init__(self,
                 name,
                social_data_dir,
                generated_data_dir,
                social_manager_configs,
                cur_time,
                to_dist = False,
                **kwargs
                 ):
        
        super().__init__(name = name,to_dist = to_dist,**kwargs)
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        social_manager = SocialManager.load_data(
                                         cur_time = cur_time,
                                         social_data_dir = social_data_dir,
                                        generated_data_dir = generated_data_dir,
                                        **social_manager_configs
                                         )
        self.social_manager = social_manager
        
        self.tools = {}
        
        self.document_prompt = PromptTemplate.from_template("""
{tweet_idx}:
    user: {user_name}
    topic: {topic}
    tweet: {page_content}""")
        
        self.update(
            filter_keys = []
        )
        
    def update(self,
                social_follow_map:dict = {
                    "follow_ids": [],
                    "friend_ids": []
                },
                interested_topics:List[str] = [],
                max_retry:int = 5,
               **retriever_kargs_update):

        retriever_tool_func,retriever_tool_dict = \
            self.social_manager.get_forum_retriever_tool(
            document_prompt = self.document_prompt,
            social_follow_map = social_follow_map,
            max_retry = max_retry,
            interested_topics = interested_topics,
            retriever_kargs_update = retriever_kargs_update)
        
        tools = {
        "search_forum":ToolAgent("search_forum",
                      tool_func=retriever_tool_func,
                      func_desc=retriever_tool_dict
                      )}
        self.update_tools(tools)
        

    def reply(self, message:Msg=None):
        
        func_name_map = {
           "self":["update",
                   "get_docs_len",
                   "get_prompt_tool_msgs",
                   "call_tool_func",
                   "save"],
        }
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        
        if func_name in func_name_map["self"]:
            func = getattr(self,func_name)
        else:
            func = getattr(self.social_manager,func_name)

        return_values = func(**kwargs)
        return Msg(self.name,content = return_values,role = "assistant")
     

    def call_tool_func(self, function_call_msg:list):
        execute_results = []
        for i,res_one in enumerate(function_call_msg):
            arg_list = self.func_name_mapping[res_one["tool"]]
        
            func_res = self.tools[res_one["tool"]].execute_func(i,arg_list,res_one["tool_input"])
            execute_results.append((res_one,func_res))
                
        return execute_results
    
    def call_manager_func(self,func_name,kwargs):
        func = getattr(self.social_manager,func_name)
        func_res = func(**kwargs)
        return func_res
