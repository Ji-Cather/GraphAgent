from LLMGraph.message import Message
from . import movie_prompt_registry,movie_prompt_default

from LLMGraph.prompt.base import BaseChatPromptTemplate
    
    
@movie_prompt_registry.register("watch_movie_batch")
class WatchMovieBatchPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             movie_prompt_default.get("watch_movie_batch",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "memory",
                     "movie_description",
                     "instruction",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)