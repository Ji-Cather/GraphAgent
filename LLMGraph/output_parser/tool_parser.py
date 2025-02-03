import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal

from agentscope.models import ModelResponse
from .base_parser import (AgentOutputParser,
                          find_and_load_json)


class ToolParser(AgentOutputParser):
    """return a batch of movie search results"""
    
  
    @staticmethod
    def parse_ai_message_to_openai_tool_action(
        raw_output: Union[dict,str],
    ):
        """Parse an AI message potentially containing tool_calls."""
        res = raw_output
        if isinstance(res,str) or (isinstance(res,dict) and len(res.get("function",[]))==0):
            if isinstance(res,dict):
                llm_output = res.get("speak","")
            else:
                llm_output = raw_output
            llm_output += "\n"
            return{"return_values":{"output": llm_output}}
            
        
        actions: List = []
        if isinstance(res,list):
            ks =["tool","tool_input","log"]
            for action in res:
                try:
                    for k in ks: assert k in action.keys()
                    actions.append(action)
                except: continue
            return {"actions":actions}
        try:
            actions_raw = res["function"]
        except:
            try:
                actions_raw = find_and_load_json(res,"list")
            except Exception as e:
                return {"fail":True}
            
        for i, func in enumerate(actions_raw):
            try:
                ks = ["tool","tool_input","log"]
                for k in ks: assert k in func.keys()
                actions.append(func)
            except:
                try:
                    func_name = func["name"]
                    func_args = func["arguments"]
                except:continue
                
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
            
    def parse_func(self, output:ModelResponse):
        raw_output = find_and_load_json(output.text,outer_type="dict")
        try:
            json_filed = self.parse_ai_message_to_openai_tool_action(raw_output)       
            output.json = json_filed
            # if "return_values" in json_filed.keys():
            #     output.finish = True
            return output
        except Exception as e:
            output.json = {"fail":True}
            return output
