from typing import Dict,Sequence,Union
from agentscope.agents import AgentBase
from .base import BaseAgentWrapper
from agentscope.message import Msg, PlaceholderMessage
import time
from LLMGraph.utils.str_process import remove_before_first_space

from LLMGraph.agent import ArticleAgent,ToolAgent
from agentscope.agents.rpc_agent import RpcAgentServerLauncher
import random


class ArticleAgentWrapper(BaseAgentWrapper):
    
    def __init__(self, 
                 name: str, 
                 agent:AgentBase,
                 manager_agent:AgentBase,
                 max_retrys: int = 3, 
                 max_tool_iters: int = 2, 
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
        

    def add_social_network(self,social_network:dict):
        return self.call_agent_func("add_social_network",
                                    kwargs={"social_network":social_network})
    def clear_discussion_cur(self):
        return self.call_agent_func("clear_discussion_cur")

    def idea_generation(self,
                      role_description:str,
                      research_content
                      ):
        return self.call_agent_func(
            func_name="idea_generation",
            kwargs={
                "role_description":role_description,
                "research_content":research_content,
            }
        )
    
    def choose_researcher(self,
                        research_content,
                        role_description
                        ):
        return_msg = self.call_agent_func(
            "choose_researcher",
            kwargs = {
                "role_description":role_description,
                "research_content":research_content,
            }
        )
        if isinstance(return_msg,PlaceholderMessage):
            return_msg.update_value()
        return return_msg
    

    def write_process(self, 
                      research_content,
                      ):
        article_write_configs = self.call_manager_func("get_article_write_configs").content
        
        online = self.call_manager_func("have_online_tools").content
        if online:
            self.call_manager_func("change_online_tools").content
            research_content = self.write_article(research_content,
                                            article_write_configs["max_refine_round"],
                                            article_write_configs["min_citations_online"],
                                            article_write_configs["max_citations_online"],
                                            online=True
                                            )
            self.call_manager_func("change_offline_tools").content

        research_content = self.write_article(research_content,
                                article_write_configs["max_refine_round"],
                                article_write_configs["min_citations_db"],
                                article_write_configs["max_citations_db"],
                                )
        

        
        return Msg(content=research_content,
                   name=self.agent.name,
                   role="assistant")

    def write_article(self, 
                      research_content,
                      max_refine_round,
                      min_citations,
                      max_citations,
                      online:bool = False
                      ):
        
        refine_round = 0
        searched_keywords = []
        _searched_items_round = [] # 本轮的候选论文
        citation_article_names = research_content.get("citations",[])
        
        role_description = self.call_manager_func(
            "get_author_description",
            kwargs={
               "agent_name":self.agent.name
            }
        ).content
        
        try:
            while(refine_round < max_refine_round):
                refine_round += 1
                agent_msgs = self.call_agent_get_prompt(func_name="write_article",
                                                kwargs={
                                                    "role_description":role_description,
                                                    "research_content":research_content,
                                                    "searched_keywords":searched_keywords,
                                                    "citation_article_names":citation_article_names,
                                                    "min_citations":min_citations,
                                                    "max_citations":max_citations,
                                                }).content
                

                response = self.step(agent_msgs,use_tools=True,return_intermediate_steps=True)
                if not isinstance(response.content,dict):
                    continue
                article_info = response.content.get("return_values",{})
                all_citations = article_info["citations"]
                
                if online:
                    citations = all_citations
                    citation_article_names.extend(citations)
                else:
                    citations = self.call_manager_func(
                        "filter_citations",
                        kwargs={
                            "citations": all_citations
                        }
                    ).content

                citation_article_names.extend(citations)

                citation_article_names = list(set(citation_article_names)) # 去重
                article_info["citations"] = citation_article_names
                article_info["searched_keywords"] = searched_keywords
                research_content.update(article_info)
                steps = response.get("intermediate_steps",[])
                ## 记录搜索过的keywords
                for step in steps:
                    query = step[0].get("tool_input",{})["query"]
                    searched_keywords.append(query)
                    _searched_items_round.append(step[1].get("result",""))
                searched_keywords = list(set(searched_keywords))
                
                research_content["searched_items"] = "\n".join(_searched_items_round)
                research_content["success"] = True
                research_content["cited_num"] = len(citation_article_names)
                research_content["refine_round"] = refine_round
                
            
        except Exception as e:
            print("write_article error", e)
        
        
        return research_content
        
    def get_agent_memory_msgs(self):
        return self.call_agent_func(
            "get_short_memory"
        )

    

    def choose_reason(self,
                      research_content,
                      cur_time_str):
        cited_doc_titles = research_content["citations"]
        docs_str = self.call_manager_func(
            "get_article_infos",
            kwargs = {
                "titles":cited_doc_titles
            }
        ).content

        role_description = self.call_manager_func(
            "get_author_description",
            kwargs={
               "agent_name":self.agent.name
            }
        ).content
        
        research_content = self.call_agent_func(
                                                func_name="choose_reason",
                                                kwargs={
                                                    "docs_str":docs_str,
                                                    "role_description":role_description,
                                                    "research_content":research_content,
                                                    "cur_time_str":cur_time_str
                                                }).content
        return Msg(content=research_content,
                   name=self.agent.name,
                   role="assistant")
    
    def group_discuss(self,
                      author_id,
                      role_description_1,
                      role_description_2,
                      research_content):
        return self.call_agent_func("group_discuss",
                                    kwargs={
                                        "author_id":author_id,
                                        "role_description_1":role_description_1,
                                        "role_description_2":role_description_2,
                                        "research_content":research_content,
                                    })
    
    def clear_discussion_cur(self):
        return self.call_agent_func("clear_discussion_cur")
    
    def return_interested_topics(self):
        return self.call_agent_func("return_interested_topics")
    
    def update_interested_topics(self,topics):
        return self.call_agent_func("update_interested_topics",
                                    kwargs={"topics":topics})