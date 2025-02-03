from LLMGraph.message import Message
from . import movie_prompt_registry,movie_prompt_default, MODEL

from LLMGraph.prompt.base import BaseChatPromptTemplate
from typing import Any,List
from agentscope.message import Msg


    
@movie_prompt_registry.register("choose_genre")
class ChooseGenrePromptTemplate(BaseChatPromptTemplate):
    movie_genre_job_instruction:str = ""
    movie_genre_age_instruction:str = ""
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             movie_prompt_default.get("choose_genre",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "memory",
                     "movie_description",
                     "agent_scratchpad",
                     ])
        
        movie_genre_job_instruction = movie_prompt_default.get(
            "movie_genre_job_instruction","")
        movie_genre_age_instruction = movie_prompt_default.get(
            "movie_genre_age_instruction","")


        super().__init__(template=template,
                         input_variables=input_variables,
                         movie_genre_age_instruction = movie_genre_age_instruction,
                         movie_genre_job_instruction = movie_genre_job_instruction,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> List[Msg]:
        # Format them in a particular way
        if "agent_scratchpad" not in kwargs.keys():
            kwargs["agent_scratchpad"] = ""

        len_prompt_inputs = sum([len(v) for v in kwargs.values()])
        
        if MODEL =="llama":
            if len_prompt_inputs <5e3:
                instruction = self.movie_genre_age_instruction + self.movie_genre_job_instruction
            elif len_prompt_inputs < 3e3:
                instruction = self.movie_genre_job_instruction
            else:
                instruction = ""
        else:
            instruction = self.movie_genre_age_instruction + self.movie_genre_job_instruction

        formatted = self.template.format(**kwargs,
                                         human_instruction = instruction)
        
        
        return [Msg(name = "User",
                    content=formatted,
                    role ="user",
                    )]