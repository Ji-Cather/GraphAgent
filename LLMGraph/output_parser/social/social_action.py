import json
from .. import social_output_parser_registry
from ..base_parser import AgentOutputParser,find_and_load_json


@social_output_parser_registry.register("forum_action")
class ForumActionParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        actions = []
        try:
            outputs = llm_output.strip()
            actions = find_and_load_json(outputs,outer_type="list")
        except Exception as e:
            return {"fail":True}
        return {"return_values":{"actions":actions}}
    
@social_output_parser_registry.register("forum_action_bigname")
class ForumActionBigNameParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        actions = []
        try:
            outputs = llm_output.strip()
            actions = find_and_load_json(outputs,outer_type="list")
        except Exception as e:
            return {"fail":True}
        return {"return_values":{"actions":actions}}