
from typing import  Union
from .. import movie_output_parser_registry
from ..base_parser import AgentOutputParser

import re



@movie_output_parser_registry.register("choose_genre")
class ChooseGenreParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        # Parse out the action and action input
        regex = r"Thought\s*\d*\s*:(.*?)Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*?)\n"
        llm_output +="\n"
        
        match = re.search(regex, llm_output, re.DOTALL|re.IGNORECASE)
        
        
        if not match:
            return {"fail":True}
        
        
        thought = match.group(1).strip().strip(" ").strip('"')
        action = match.group(2).strip()
        action_input = match.group(3).strip().strip(" ").strip('"')

        
        if action.lower() == "choose":
            return {"return_values":
                    {"output":action_input,
                        "thought":thought}}
        elif action.lower()=="giveup":
            return {"return_values":
                    {"output":"I choose none of these options.",
                        "thought":thought}}
        
        # Return the action and action input
        return {"fail":True}