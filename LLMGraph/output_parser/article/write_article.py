import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal

import re
from agentscope.models import ModelResponse,ModelWrapperBase
from agentscope.message import Msg,MessageBase

from ..tool_parser import ToolParser, find_and_load_json
from .. import article_output_parser_registry


@article_output_parser_registry.register("write_article")
class WriteArticleParser(ToolParser):

    
    @property
    def _type(self) -> str:
        return "write_article_parser"
    
    @staticmethod
    def parse_ai_message_to_openai_tool_action(
        raw_output: Union[dict,str],
    ):
        """Parse an AI message potentially containing tool_calls."""
        res = raw_output
        if isinstance(res,str) or (isinstance(res,dict) and len(res.get("function",[]))==0):
            try:
                if isinstance(res,str):
                    regex = r"title\s*\d*\s*:(.*?)\nkeywords.*?:(.*?)\nabstract.*?:(.*?)\ncitations.*?:(.*)"    
                    output = re.search(regex, res, re.DOTALL|re.IGNORECASE)
                    # finish = output.group(5)
                    # if "true" in finish.lower():
                    #     finish = True
                    # else:
                    #     finish = False
                    article_infos = {"title":output.group(1).strip(),
                                    "keywords":output.group(2).strip().split(","),
                                    "abstract":output.group(3).strip(),
                                    "citations":output.group(4).strip().split("\n")}
                else:
                    article_infos = res
                    assert isinstance(article_infos,dict)
                    ks = ["title","keywords","abstract","citations"]
                    for k in ks: assert k in article_infos.keys()
                return {"return_values":article_infos}
            except Exception as e:
                return {"fail":True}
        
        
        actions: List = []
        if isinstance(res,list):
            ks =["tool","tool_input","log"]
            for action in res:
                try:
                    for k in ks: assert k in action.keys()
                    actions.append(action)
                except:
                    continue
            return {"actions":actions}
        try:
            actions_raw = res["function"]
        except:
            try:
                actions_raw = find_and_load_json(res,"list")
            except Exception as e:
                return {"fail":True}
            
        for i, func in enumerate(actions_raw):
            func_name = func["name"]
            func_args = func["arguments"]
            if not isinstance(func_args,dict):
                try:
                    _tool_input = json.loads(func_args or "{}")
                except JSONDecodeError:
                    return {"fail":True}
        
                if "__arg1" in _tool_input:
                    tool_input = _tool_input["__arg1"]
                else:
                    tool_input = _tool_input
            else: tool_input = func_args
            
            content_dict = list(filter(lambda item: item[0] not in ["function"], res.items()))
            content = "\n".join([f"{item[0]}:{item[1]}" for item in content_dict])
            log = f"\nInvoking: `{func_name}` with `{tool_input}`\n {content}\n"
            actions.append(
                {
                    "tool": func_name,
                    "tool_input": tool_input,
                    "log": log,
                }
            )
        return {"actions":actions}