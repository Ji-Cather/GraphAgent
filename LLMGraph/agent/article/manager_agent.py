from LLMGraph.agent.tool_agent import ToolAgent
from LLMGraph.manager import ArticleManager
from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from LLMGraph.tool import (create_article_retriever_tool,
                           create_article_online_retriever_tool)
from typing import List,Dict,Union,Any
from datetime import datetime

from LLMGraph.agent.base_manager import ManagerAgent
import copy
class ArticleManagerAgent(ManagerAgent):
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
    article_manager:ArticleManager
    
    def __init__(self,
                 name,
                article_manager_configs,
                model_config_name,
                task_path,                
                config_path,
                cur_time,
                article_write_configs,
                 ):
        
        super().__init__(name = name,
                         model_config_name=model_config_name)
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        article_manager = ArticleManager.load_data(**article_manager_configs,
                                         task_path = task_path,
                                         config_path = config_path,
                                         llm = self.model,
                                         cur_time = cur_time)
        article_manager.update_big_name_list()
        self.article_manager = article_manager

        # retriever = article_manager.get_retriever()

        
        prompt = PromptTemplate.from_template("""
Title: {title}
Topic: {topic}
Cited: {cited}
Author: {author_name} {institution} {country} cited: {author_cited}
Publish Time: {time}
Content: {page_content}""")
        

       
        self.tools = {}
        self.tools_cache = {}# 暂存信息

        self.document_prompt = prompt

        avg_deg,std_deg = self.article_manager.calculate_avg_degree_citation()
        if self.article_manager.online_retriever is not None:
            
            citations = article_write_configs["citations"]
            article_write_configs.update({
                "min_citations_db":citations,
                "max_citations_db":citations+10,
                "min_citations_online":citations,
                "max_citations_online":citations+10,
            }) # online
            retriever_tool_func,retriever_tool_dict = create_article_online_retriever_tool(
                self.article_manager.online_retriever,
                "Search",
                "Search for relevant papers, so as to refine your paper. \
These papers should be included in your paper's citations if you use them in your paper. \
Your paper should cite at least {min_citations_online} papers!".format_map(article_write_configs),
            )
            self.online_tools = {
        "Search":ToolAgent("Search",
                      tool_func=retriever_tool_func,
                      func_desc=retriever_tool_dict
                      )}
                      
        else:
            if not article_write_configs.get("use_graph_deg",False):
                avg_deg = int(article_write_configs["citations"])
            article_write_configs.update({
                "min_citations_db":int(avg_deg-std_deg),
                "max_citations_db":int(avg_deg+std_deg)
            })
            self.online_tools= None
            
        update_configs = self.article_manager.get_article_write_configs()
        article_write_configs.update(update_configs)
        self.article_write_configs = article_write_configs
        self.article_write_configs["min_citations_db"] = 1 if self.article_write_configs["min_citations_db"] <1 else \
            self.article_write_configs["min_citations_db"]

        self.update()


    def get_article_write_configs(self):
        return self.article_write_configs

    def get_docs_len(self):
        return len(self.article_manager.article_loader.docs)



    def get_llm_author(self,
                       topic:str,
                        author_num:int = 5):
        return self.article_manager.get_llm_author(self.model,
                                            topic=topic,
                                            author_num=author_num)
    def reply(self, message:Msg=None):
        
        func_name_map = {
           "self":["update",
                   "get_docs_len",
                   "get_prompt_tool_msgs",
                   "call_tool_func",
                   "save",
                   "get_article_write_configs",
                   "have_online_tools",
                   "change_online_tools",
                   "change_offline_tools",
                   "get_llm_author"],
        }
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        
        if func_name in func_name_map["self"]:
            func = getattr(self,func_name)
        else:
            func = getattr(self.article_manager,func_name)

        return_values = func(**kwargs)
        if return_values is None:
            return_values = ""
        return Msg(self.name,content = return_values,role = "assistant")
     
    def update(self,
               articles = None,
               interested_topics = [],
               research_topic = "",
               update_retriever = True
               ):
        if articles is not None:
            self.article_manager.write_and_update_db(
            articles)
        
        if update_retriever:
            retriever = self.article_manager.get_retriever()
            retriever_tool_func,retriever_tool_dict = create_article_retriever_tool(
                retriever,
                "search",
        "Search for relevant papers, so as to refine your paper. \
    These papers should be included in your paper's citations if you use them in your paper. \
    Your paper should cite at least {min_citations_db} papers!".format_map(self.article_write_configs),
                
                author_data = self.article_manager.author_data,
                article_meta_data = self.article_manager.article_meta_data,
                experiment = self.article_manager.experiment,
                document_prompt = self.document_prompt,
                filter_keys= self.article_manager.tool_kwargs["filter_keys"],
                big_name_list = self.article_manager.big_name_list,
                interested_topics = interested_topics,
                research_topic = research_topic)
            
            tools = {
            "search":ToolAgent("search",
                        tool_func=retriever_tool_func,
                        func_desc=retriever_tool_dict
                        )}
            self.update_tools(tools)
    

    def change_online_tools(self):
        self.tools_cache = self.tools
        self.tools = self.online_tools

    def change_offline_tools(self):
        self.tools = self.tools_cache

    def have_online_tools(self):
        return self.online_tools is not None

    def save(self,start_time,cur_time,save_encoded_features:int = -1):
        if save_encoded_features <= 0:
            self.article_manager.save_networkx_graph()
            cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
            self.article_manager.write_and_update_db()
            self.article_manager.save_infos(cur_time,
                                            start_time,
                                            self.article_write_configs)
        else:
            self.article_manager.save_encoded_features(save_encoded_features)
        
    def access_article_db(self,func_name,func_kwargs):
        return getattr(self.article_manager,func_name)(**func_kwargs)
    
    
    
    def call_manager_func(self,func_name,kwargs):
        func = getattr(self.article_manager,func_name)
        func_res = func(**kwargs)
        return func_res

