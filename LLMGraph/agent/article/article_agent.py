# from LLMGraph.manager import ForumManager
# from LLMGraph.involvers import System,Tool,search_forum_topk
from LLMGraph.message import Message
from LLMGraph.prompt.article import article_prompt_registry

from LLMGraph.output_parser import article_output_parser_registry

# from langchain.agents import AgentExecutor
from typing import Any, List, Optional, Tuple, Union,Dict,Sequence

from LLMGraph.memory import BaseMemory,article_memory_registry

from LLMGraph.memory.article import SocialMemory,RationalMemory
import random
import copy
from agentscope.message import Msg,PlaceholderMessage
from .. import agent_registry

from LLMGraph.agent.base_agent import BaseGraphAgent

import json
from LLMGraph.prompt import MODEL
from LLMGraph.utils.count import count_prompt_len

@agent_registry.register("article_agent")
class ArticleAgent(BaseGraphAgent):
    
    social_memory:SocialMemory
    write_memory:RationalMemory
    
    def __init__(self, 
                 name,
                infos,
                agent_configs,
                social_network:dict,
                **kwargs):
        
        agent_configs = copy.deepcopy(agent_configs)
        agent_llm_config = agent_configs.get('llm')
        
        init_mode = "group_discuss"
        prompt_template = article_prompt_registry.build(init_mode)
        output_parser = article_output_parser_registry.build(init_mode)

        model_config_name = agent_llm_config["config_name"]
        
        
        social_memory_config = agent_configs.get('social_memory')
        social_memory_config.update({"social_network":copy.deepcopy(social_network)})
    
        social_memory_config["name"] = infos.get("name",name)
        social_memory_config["id"] = name
        social_memory = article_memory_registry.build(**social_memory_config)
        
        
        write_memory_config = agent_configs.get('write_memory')
        write_memory_config["name"] = infos.get("name",name)
        write_memory_config["id"] = name
        write_memory_config["interested_topics"] = infos.get("topics",[])
        write_memory = article_memory_registry.build(**write_memory_config)

        
        self.infos = infos
        self.social_memory = social_memory
        self.write_memory = write_memory
        self.mode = init_mode
        
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
                    mode = "group_discuss",
                    ):
       
        if self.mode == mode : return
        self.mode = mode
        
        prompt = article_prompt_registry.build(mode)
        output_parser = article_output_parser_registry.build(mode)
        self.prompt_template = prompt
        self.output_parser = output_parser

    def observe(self, messages:Union[Msg]=None):
        if not isinstance(messages,Sequence):
            messages = [messages]
        for message in messages:
            if isinstance(message,PlaceholderMessage):
                message.update_value()
            # if not isinstance(message,Msg):continue
            if message.content == "":
                continue
            if message.get("social"):
                self.social_memory.add_message(messages=messages)
                break
            elif message.name == self.name or message.name == "system":
                self.short_memory.add(messages)
                break
       
    

    def add_social_network(self,social_network:dict):
        sn_ori = self.social_memory.social_network
        for id_author in social_network.keys():
            if id_author not in sn_ori.keys():
                self.social_memory.social_network[id_author] = \
                    social_network[id_author]

    
    

    

   
        
    def choose_researcher(self,
                        research_content,
                        role_description
                        ):
        self.reset_state(mode="choose_researcher")
        past_context = self.social_memory.retrieve_recent_chat("all",upper_token = 4e2)
        researcher_infos = self.social_memory.get_researchers_infos()
        
        prompt_inputs={
            "role_description":role_description,
            "research_topic":research_content.get("topic",""),
            "past_context":past_context,
            "researcher":researcher_infos,
            }
        response = self.step(prompt_inputs)
        other_id = None
        try:
            response = response.content.get("return_values",{})
            name = response.get("researcher","")
            for candidate_id, candidate_info in self.social_memory.social_network.items():
                if candidate_info["name"] in name:
                    other_id = candidate_id
                    break
            
        except:
            candidate_id = random.choice(list(self.social_memory.social_network.keys()))
            other_id = candidate_id
        if other_id is None:
            other_id = random.choice(list(self.social_memory.social_network.keys()))
        assert other_id is not None
        return Msg(self.name,content=other_id,role = "assistant")
        
  
    # 这里加一个recent chat
    def group_discuss(self,
                      author_id,
                      role_description_1,
                      role_description_2,
                      research_content):
        self.reset_state(mode="group_discuss")
        past_context = self.social_memory.retrieve_recent_chat("all",upper_token = 5e2)
        cur_context = self.social_memory.retrieve_recent_chat(author_id, upper_token = 5e2)
        current_paper_template = """
The paper you are going to write is about: {topic}
"""
        prompt_inputs = {
            "character_1":role_description_1,
            "character_2":role_description_2,
            "past_context":past_context,
            "cur_context":cur_context,
            "research_content": current_paper_template.format_map(research_content),
        }
        response = self.step(prompt_inputs)
        try:
            response = response.content.get("return_values",{})
            context = response.get("communication","")

        
            # new ver
            msg = Msg(self.name,
                      context,
                      "assistant",
                      social = True)
            self.social_memory.add_message(msg)
            return msg
        
        except Exception as e:
            msg = Msg(self.name,
                      "",
                      "assistant")
            return msg

    
        
                    
    def idea_generation(self,
                      role_description:str,
                      research_content
                      ):
        self.reset_state(mode="get_idea")
        past_context = self.social_memory.retrieve_recent_chat("all",upper_token = 5e2)
        researcher_infos = self.social_memory.get_researchers_infos()
        
        prompt_inputs={
            "role_description": role_description[:500],
            "research_idea":research_content["topic"],
            "past_context":past_context,
            "researcher":researcher_infos[:500],
            }
        
        response = self.step(prompt_inputs)
        if response.get("fail",False):
            return response
        try:
            response = response.content.get("return_values",{})
            action = response.get("action","")
            if action == "discuss":
                research_content["finish"] = False
            else:
                research_content["finish"] = False
                research_content.update(response)
               
        
        except Exception as e:
            research_content["finish"] = False
        
        return Msg(self.name,content=research_content,role = "assistant")
   
    

    
        
    def choose_reason(self,
                      docs_str,
                      role_description:str,
                      research_content:str,
                      cur_time_str:str
                      ):
        
        cited_doc_titles = research_content["citations"]
        if len(cited_doc_titles) == 0:
            print("no citation for", research_content)
            return Msg(self.name,content=research_content,role="assistant")
        
         
        citation_motive_reasons = """
1. Background: The purpose of referencing literature is to provide contextual background and establish the foundation for the current research.

2. Background fundamental idea: Referencing scholarly works serves the fundamental purpose of elucidating the core concepts or hypotheses underlying the research endeavor.

3. Method technical basis: The incorporation of citations within academic writing serves as a technical basis for validating claims, bolstering arguments, and demonstrating the intellectual lineage of ideas.

4. Comparison: Referencing facilitates the comparison of findings, methodologies, and theoretical frameworks across different studies, enabling scholars to situate their work within the broader scholarly discourse.
"""
        citation_part_reasons = """
1. Because the content of this article is similar to what you are studying.
2. Because this article has a high number of citations.
3. Because this article was published recently.
4. Because the author of this article is a highly cited author.
5. Because the author of this article is from the same country/region/institution as mine.
6. Because the article focused on a similar topic of research as mine.
7. Because the author is a author I known.
"""


        current_paper_template = """\
title: {title}
keywords: {keyword}
abstract: {abstract}
citations: {citations}
time: {cur_time_str}
"""

        
        for reason_key,citation_reasons in {
            "motive_reason":citation_motive_reasons,
            "part_reason":citation_part_reasons
            }.items():
        
            prompt_inputs = {
                "role_description":role_description,
                "memory":self.social_memory.retrieve_recent_chat(upper_token=1e3),
                "searched_papers": research_content.get("searched_items",""),
                "paper_content": current_paper_template.format(
                    title = research_content.get("title",""),
                    keyword = ",".join(research_content["keywords"]),
                    abstract = research_content["topic"]+"\n"+research_content["abstract"],
                    citations = "\n".join([f"{idx+1}. \"{name}\"" \
                        for idx,name in enumerate(cited_doc_titles)]),
                    cur_time_str = cur_time_str
                ),
                "citation_articles": docs_str,
                "citation_reasons": citation_reasons,
                "num_citation": len(cited_doc_titles)
            }
            self.reset_state("choose_reason")
            try:
                response = self.step(prompt_inputs)
                citation_reasons = response.content.get("return_values",{}).get("citation_reasons",{})
                try:
                    json.dumps(citation_reasons)
                except:
                    citation_reasons = {}
                research_content[reason_key] = citation_reasons
            except:
                pass
        
        reason_msg = Msg(self.name,content = research_content,role="assistant")
        
        self.write_memory.add_message(reason_msg)

        return reason_msg



    def write_article(self,
                      role_description:str,
                      research_content,
                      searched_keywords :list = [],
                      citation_article_names:list =[],
                      min_citations = 10,
                      max_citations = 30,
                      topics_available:list = []
                      ):
        self.reset_state("write_article")

        current_paper_template = """
The version of your paper now:     

title: {title}
keywords: {keyword}
abstract: {abstract}
citations: {citations}
"""
        search_info_template = """
You have already searched the following keywords, so avoid searching them again! 
{searched_info}

Search other keywords instead!!
"""

        researcher_infos = self.social_memory.get_researchers_infos()
        past_context = self.social_memory.retrieve_recent_chat("all",upper_token=1e3)

        prompt_inputs = {
        "role_description":role_description,
        "researcher":researcher_infos,
        "past_context": past_context,
        "write_memory": self.write_memory.retrieve_article_memory(
            upper_token=5e2,
        ),
        "current_paper": current_paper_template.format(
            title = research_content.get("title",""),
            keyword = ",".join(research_content["keywords"]),
            abstract = research_content["topic"]+"\n"+research_content["abstract"],
            citations = "\n".join([f"{idx+1}. \"{name}\"" \
                for idx,name in enumerate(citation_article_names)])
        ),
            "searched_info":"",
            "max_citations":max_citations,
            "min_citations":min_citations
    }          
            
        if len(searched_keywords)>0:
            searched_info = search_info_template.format(
                searched_info = "\n".join([f"{idx+1} . {query}" \
                    for idx, query in enumerate(searched_keywords)])
            )
            prompt_inputs["searched_info"] = searched_info
            
        return prompt_inputs
           

    def return_interested_topics(self):
        return Msg(self.name,
                   self.write_memory.return_interested_topics(),
                   "assistant")
    
    def update_interested_topics(self,topics):
        self.write_memory.update_interested_topics(topics)