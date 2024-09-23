import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal
from .. import social_output_parser_registry

from agentscope.models import ModelResponse,ModelWrapperBase
from ..tool_parser import ToolParser


@social_output_parser_registry.register("search_forum")
class ForumSearchParser(ToolParser):
    """return a batch of movie search results"""
    @property
    def _type(self) -> str:
        return "search_forum"