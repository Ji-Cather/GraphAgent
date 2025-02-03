from pydantic import BaseModel

from typing import Any, Optional, Union, Sequence, Literal
from abc import abstractmethod

from agentscope.message import MessageBase
from agentscope.models import ModelResponse
import json


class AgentOutputParser(BaseModel):
    
    # @abstractmethod
    def parse(self, text: str) -> Union[dict,list,str]:
        """Parse text into json_field."""

    def parse_func(self, output:ModelResponse) -> ModelResponse:
        raw_data = output.text
        try:
            json_field = self.parse(raw_data)
            
        except Exception as e:
            json_field = {"fail":True}

        output.json = json_field
        return output
    
def find_and_load_json(s, outer_type = "dict"):
    try:
        # 尝试直接解析整个字符串
        return json.loads(s)
    except json.JSONDecodeError as e:
        # 解析错误，可能是字符串不是一个完整或正确的 JSON
        if outer_type == "dict":
            
            start_pos = s.find('{')
            end_pos = s.rfind('}')
            if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
                # 找到了可能的 JSON 字典，尝试再次解析
                try:
                    return json.loads(s[start_pos:end_pos+1])
                except json.JSONDecodeError:
                    try: 
                        return json.loads(s[start_pos:end_pos+1]+"}")
                    except: return s
        if outer_type == "list":
            start_pos = s.find('[')
            end_pos = s.rfind(']')
            if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
                try:
                    return eval(s[start_pos:end_pos+1])
                except Exception as e:
                    try: 
                        return eval(s[start_pos:end_pos+1]+"]")
                    except: 
                        try:
                            return eval("["+s[start_pos:end_pos+1]+"]")
                        except:
                            return s
        else:
            return s