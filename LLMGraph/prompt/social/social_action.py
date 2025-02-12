from LLMGraph.message import Message
from LLMGraph.prompt.social import social_prompt_default,social_prompt_registry

from LLMGraph.prompt.base import BaseChatPromptTemplate  

from typing import Any,List
from agentscope.message import Msg

@social_prompt_registry.register("forum_action")
class ForumActionPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             social_prompt_default.get("forum_action",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "memory",
                     "twitter_data",
                     "friend_data",
                     "num_followers",
                     "forum_action_hint",
                     "agent_scratchpad"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
        
    def format_messages(self, **kwargs) -> List[Msg]:
        # Format them in a particular way
        if "agent_scratchpad" not in kwargs.keys():
            kwargs["agent_scratchpad"] = ""
            
        formatted = self.template.format(**kwargs)

        return [Msg(name = "User",
                    content=formatted,
                    role ="user",
                    )]
        

@social_prompt_registry.register("forum_action_bigname")
class ForumActionBigNamePromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             social_prompt_default.get("forum_action_bigname",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "memory",
                     "twitter_data",
                     "friend_data",
                     "num_followers",
                     "forum_action_hint",
                     "agent_scratchpad"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    def format_messages(self, **kwargs) -> List[Msg]:
        # Format them in a particular way
        if "agent_scratchpad" not in kwargs.keys():
            kwargs["agent_scratchpad"] = ""
            
        formatted = self.template.format(**kwargs)

        return [Msg(name = "User",
                    content=formatted,
                    role ="user",
                    )]
