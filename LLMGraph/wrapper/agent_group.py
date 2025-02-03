"""A general dialog agent."""
import random
import time
import re
from typing import List
from loguru import logger

from agentscope.message import Msg
from agentscope.agents import AgentBase
from agentscope.message import Msg,PlaceholderMessage
from LLMGraph.wrapper.article import ArticleAgentWrapper


class GroupDiscussAgent(AgentBase):
    """A Moderator to collect values from participants."""

    def __init__(
        self,
        name: str,
        communication_num:int = 5,
        agents:List[AgentBase] = [],
        manager_agent: AgentBase = None
    ) -> None:
        super().__init__(name)
        self.agents = agents
        self.communication_num = communication_num
        self.manager_agent = manager_agent

    def call_manager_agent_func(self,
                                func_name:str,
                                kwargs:dict ={})->Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name,
                )
        return_msg = self.manager_agent(msg)
        return return_msg
    
    def call_agent_func(self,
                        agent:ArticleAgentWrapper,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name
                )
        return_msg = agent(msg)
        if isinstance(return_msg,PlaceholderMessage):
            return_msg.update_value()
        return return_msg


    def communication(self, research_content) -> Msg:
        for idx in range(self.communication_num):
            agent = self.agents[idx%len(self.agents)]
            role_description = self.call_manager_agent_func("get_author_description",
                                                    kwargs={"agent_name":agent.name}).content

            
            candidate_id_msg = self.call_agent_func(agent, "choose_researcher",
                                    kwargs={"role_description":role_description,
                                            "research_content":research_content})
            
            candidate_id = candidate_id_msg.content
            role_description_2 = self.call_manager_agent_func("get_author_description",
                                            kwargs={"agent_name":candidate_id}).content
            
            group_discussion_msg = self.call_agent_func(agent, "group_discuss",
                                    kwargs={"role_description_1":role_description,
                                            "role_description_2":role_description_2,
                                            "research_content":research_content,
                                            "author_id": candidate_id})
            
            for agent in self.agents:
                agent.observe(group_discussion_msg)

                    # print(role_description)
        research_content_msg = self.call_agent_func(agent,
                                                    "idea_generation",
                                                    kwargs={"role_description":role_description,
                                                "research_content":research_content})
        try:
            research_content_update = research_content_msg.content
            assert isinstance(research_content_update,dict)
        except:
            research_content_update = research_content

        return Msg(
            name=self.name,
            role="assistant",
            content=research_content_update,
        )
    

    def write(self, 
              research_content,
              cur_time_str) -> Msg:
        agent_first_author = self.agents[0]

        interested_topics = self.call_agent_func(agent_first_author, "return_interested_topics").content
        try:
            research_topic = research_content.get("topic", None)
        except:
            print("error", research_content)
        self.call_manager_agent_func("update",
                                     kwargs={"interested_topics":interested_topics,
                                             "research_topic":research_topic})

        
        research_content = self.call_agent_func(agent_first_author, 
                            "write_process",
                        kwargs={"research_content":research_content}).content
        
        
        # research_content = self.call_agent_func(agent_first_author, 
        #                                             "choose_reason",
        #                                         kwargs={"research_content":research_content,
        #                                                 "cur_time_str":cur_time_str}).content
        if research_content.get("topic") is not None:
            self.call_agent_func(agent_first_author,
                                 "update_interested_topics",
                                     kwargs={
                                         "topics": [research_content.get("topic")]
                                     })

        return Msg(
            name=self.name,
            role="assistant",
            content=research_content,
        )
        
    def reply(self, message: Msg) -> Msg:
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        func = getattr(self,func_name)
        res = func(**kwargs)
        assert isinstance(res,Msg)

        return res