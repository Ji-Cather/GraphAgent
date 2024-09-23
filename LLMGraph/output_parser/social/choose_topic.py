from .. import social_output_parser_registry
from ..base_parser import AgentOutputParser, find_and_load_json
from typing import  Union
import json

@social_output_parser_registry.register("choose_topic")
class ChooseTopicParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        topics =[]
        try:
            topics = find_and_load_json(llm_output.strip(),
                                        "list")
        except Exception as e:
            return {"fail":True}
        
        return {"return_values":{"topics":topics}}