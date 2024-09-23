import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal
from .. import movie_output_parser_registry
from agentscope.models import ModelResponse
from ..tool_parser import ToolParser

@movie_output_parser_registry.register("watch_movie_batch")
class WatchMovieBatchParser(ToolParser):
    """return a batch of movie search results"""
    
    @property
    def _type(self) -> str:
        return "watch_movie_parser"
    