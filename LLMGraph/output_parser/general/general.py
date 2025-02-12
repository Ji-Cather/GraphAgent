import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal

import re
from agentscope.models import ModelResponse,ModelWrapperBase
from agentscope.message import Msg,MessageBase
from ..base_parser import AgentOutputParser, find_and_load_json
from ..tool_parser import ToolParser
from .. import general_output_parser_registry
import yaml
from LLMGraph import MODEL


prompt_one = yaml.safe_load(open(f"LLMGraph/prompt/general/{MODEL}/interact.yaml"))
dataset_names = ["".join(k.split("_")[:-1]) for k in prompt_one.keys()]
dataset_names = list(set(dataset_names))

def _parse_func(self, llm_output: str):
    try:
        output_json = find_and_load_json(llm_output, "dict")
        json_filed = {"return_values": output_json}
        return json_filed
    except Exception as e:
        return {"fail": True}

def parse_ai_message_to_openai_tool_action(
        raw_output: Union[dict,str],
    ):
    """Parse an AI message potentially containing tool_calls."""
    res = raw_output
    if isinstance(res,str) or (isinstance(res,dict) and len(res.get("function",[]))==0):
        try:
            output_json = find_and_load_json(res, "dict")
            json_filed = {"return_values": output_json}
            return json_filed
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

for dataset_name in dataset_names:
    node_mode = f"interact_{dataset_name}_query"
    class_name = node_mode + 'Parser'
    query_parser_class = type(class_name, (ToolParser,),{})
    general_output_parser_registry.register(node_mode)(query_parser_class)

    node_mode = f"interact_{dataset_name}_action"
    class_name = node_mode + 'Parser'
    action_parser_class = type(class_name, (AgentOutputParser,), {"parse": _parse_func})
    general_output_parser_registry.register(node_mode)(action_parser_class)
