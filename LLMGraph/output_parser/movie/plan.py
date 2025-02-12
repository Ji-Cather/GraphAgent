from .. import movie_output_parser_registry
from typing import  Union
from ..base_parser import AgentOutputParser, find_and_load_json

import re
import ast

@movie_output_parser_registry.register("watch_plan")
class WatchPlanMovierParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        
        try:
            number_movies = find_and_load_json(llm_output,"list")
            return {"return_values":{"plan":number_movies}}
        
        except Exception as e:
            return {"fail":True}