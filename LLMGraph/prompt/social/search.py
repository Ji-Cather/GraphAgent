from LLMGraph.message import Message
from LLMGraph.prompt.social import social_prompt_default,social_prompt_registry

from LLMGraph.prompt.base import BaseChatPromptTemplate  
    
@social_prompt_registry.register("search_forum")
class ForumSearchPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             social_prompt_default.get("search_forum",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "memory",
                     "searched_info",
                     "agent_scratchpad"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
