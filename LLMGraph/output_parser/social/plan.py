from .. import social_output_parser_registry
from ..base_parser import AgentOutputParser, find_and_load_json
from typing import  Union
import json

@social_output_parser_registry.register("forum_action_plan")
class ForumActionPlanParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        action_plan = {}
        try:
            action_plan = find_and_load_json(llm_output.strip(),
                                            "dict")
        except Exception as e:
            return {"fail":True}
        
        return {"return_values":{"action_plan":action_plan}}