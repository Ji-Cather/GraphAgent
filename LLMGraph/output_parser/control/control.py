import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal

import re
from agentscope.models import ModelResponse,ModelWrapperBase
from agentscope.message import Msg,MessageBase
from ..base_parser import AgentOutputParser, find_and_load_json
from .. import control_output_parser_registry


@control_output_parser_registry.register("article")
class ArticleControlParser(AgentOutputParser):

    def parse(self, llm_output: str):
        configs = {}
        
        try:
            configs = find_and_load_json(llm_output.strip(),
                                        "dict")
        except Exception as e:
            return {"fail":True}
        
        return {"return_values":{"configs":configs}}
    
@control_output_parser_registry.register("movie")
class MovieControlParser(AgentOutputParser):

    def parse(self, llm_output: str):
        configs = {}
        
        try:
            configs = find_and_load_json(llm_output.strip(),
                                        "dict")
        except Exception as e:
            return {"fail":True}
        
        return {"return_values":{"configs":configs}}

@control_output_parser_registry.register("social")
class SocialControlParser(AgentOutputParser):

    def parse(self, llm_output: str):
        configs = {}
        
        try:
            configs = find_and_load_json(llm_output.strip(),
                                        "dict")
        except Exception as e:
            return {"fail":True}
        
        return {"return_values":{"configs":configs}}