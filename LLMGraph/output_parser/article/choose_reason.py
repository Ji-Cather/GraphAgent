from __future__ import annotations

import re
from typing import Union
from ..base_parser import AgentOutputParser, find_and_load_json
from .. import article_output_parser_registry
from agentscope.models import ModelResponse

import json
@article_output_parser_registry.register("choose_reason")
class ChooseReasonParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        try:
            citation_reasons = find_and_load_json(llm_output,"list")
            assert isinstance(citation_reasons, list)
            json_filed = {"return_values":
                    {"citation_reasons": citation_reasons}}
            return json_filed
        except Exception as e:
            return {"fail":True}