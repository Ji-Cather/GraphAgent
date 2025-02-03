from LLMGraph.message import Message
from . import movie_prompt_registry,movie_prompt_default

from LLMGraph.prompt.base import BaseChatPromptTemplate
from typing import Any,List
from agentscope.message import Msg

    
@movie_prompt_registry.register("rate_movie")
class RatePromptTemplate(BaseChatPromptTemplate):
    
    movie_rate_instruction:str =""

    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             movie_prompt_default.get("rate_movie",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "memory",
                     "movie_description",
                     "min_rate",
                     "max_rate",
                     "interested_genres",
                     "watched_movie_names",
                     "agent_scratchpad",
                     ])
        
        movie_rate_instruction = movie_prompt_default.get("movie_rate_instruction","")

        super().__init__(template=template,
                         input_variables=input_variables,
                         movie_rate_instruction= movie_rate_instruction,
                         **kwargs)
        

    def format_messages(self, **kwargs) -> List[Msg]:
        # Format them in a particular way
        if "agent_scratchpad" not in kwargs.keys():
            kwargs["agent_scratchpad"] = ""
            
        human_instruction = self.movie_rate_instruction
        formatted = self.template.format(**kwargs,
                                         human_instruction = human_instruction)

        return [Msg(name = "User",
                    content=formatted,
                    role ="user",
                    )]
