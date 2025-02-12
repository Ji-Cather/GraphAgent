from LLMGraph.message import Message
from . import movie_prompt_registry,movie_prompt_default

from LLMGraph.prompt.base import BaseChatPromptTemplate
    
    
@movie_prompt_registry.register("watch_plan")
class WatchPlanPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             movie_prompt_default.get("watch_plan",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "task",
                     "agent_scratchpad",
                     "requirement"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
