from typing import List, Tuple
from agentscope.agents import ReActAgent,AgentBase
from pydantic import BaseModel

from LLMGraph.prompt.base import BaseChatPromptTemplate
from agentscope.message import Msg
from agentscope.models import _get_model_wrapper, load_model_by_config_name, ModelWrapperBase
from typing import Any, List, Optional, Tuple, Union,Dict,Sequence
from agentscope.memory import TemporaryMemory
from LLMGraph.output_parser.base_parser import AgentOutputParser
import json
from LLMGraph.prompt import MODEL
from LLMGraph.utils.count import count_prompt_len, parse_prompt



class BaseGraphAgent(AgentBase):
    def __init__(self, 
                 name: str,
                 prompt_template,
                 output_parser:AgentOutputParser,
                 max_retrys:int = 2,
                 sys_prompt: str= None, 
                 model_config_name: str = None, 
                 use_memory: bool = True, 
                 memory_config: dict = None,
                 to_dist = False,
                 short_memory_config: Optional[dict] = None,
                 ) -> None:
        
        super().__init__(name, 
                         sys_prompt, 
                         model_config_name, 
                         use_memory, 
                         memory_config,
                         to_dist)

        self.prompt_template = prompt_template
        self.max_retrys = max_retrys  
        self.short_memory = TemporaryMemory(short_memory_config)
        self.output_parser = output_parser

    def step(self,
             prompt_inputs) -> Msg:
        
        desearlize_prompt_msgs = [ 
            *self.prompt_template.format_messages(**prompt_inputs),
            *self.short_memory.get_memory(recent_n=3)]
        for msg in desearlize_prompt_msgs:
            if isinstance(msg.name,int):
                msg.name = str(msg.name)
        
        prompt = self.model.format(
                desearlize_prompt_msgs
            )
        
        # Step 1: Thought
        # Generate LLM response
        for i in range(self.max_retrys):
            prompt = parse_prompt(prompt)

            res = self.model(
                prompt,
                parse_func=self.output_parser.parse_func,
                max_retries=self.max_retrys,
            )

            res_json = res.json
            if res_json and not res_json.get("fail",False):
                msg_finish = Msg(self.name, res.json, role="assistant")
                self.speak(
                        Msg(self.name, json.dumps(res.raw, indent=4), role="assistant"),
                    )
                return msg_finish
            
        return Msg("user",content= f"FAIL TO PARSE",role = "user", fail = True)

    def get_short_memory(self) -> Msg:
        memory = self.short_memory.get_memory(recent_n=3)
        return Msg(self.name,content=memory,role="assistant")


    def clear_short_memory(self):
        self.short_memory.clear()

    def reply(self, message:Msg=None):
        """对于三种message calling进行回复:
        1. get_prompt msg(content = "get_prompt", func = func_name, kwargs = {})
        2. call_function msg(content = "call_function", func = func_name, kwargs = {})
        3. reply prompt: msg(content = "reply_prompt", prompt = [prompt_msgs])

        Args:
            message (Msg, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if  message.content == "get_prompt":
            kwargs = message.get("kwargs",{})
            func_name = message.get("func","")
            func = getattr(self,func_name)
            prompt_inputs = func(**kwargs)
            return_msgs = self.prompt_template.format_messages(**prompt_inputs)
            return Msg("user",content=return_msgs,role = "user")
        
        if  message.content == "call_function":
            func_name = message.get("func","")
            kwargs = message.get("kwargs",{})
            func = getattr(self,func_name)
            res = func(**kwargs)
            if res is None:
                res = Msg(self.name,content = None,role = "assistant")
            assert isinstance(res,Msg), res
            return res

        if  message.content == "reply_prompt":
            prompt_msgs = message.get("prompt",[])
            desearlize_prompt_msgs = []
            for prompt_msg in prompt_msgs:
                if isinstance(prompt_msg,Msg):
                    desearlize_prompt_msgs.append(prompt_msg)
                elif isinstance(prompt_msg,dict):
                    desearlize_prompt_msgs.append(Msg(**prompt_msg))
            
            # parse name int
            for msg in desearlize_prompt_msgs:
                if isinstance(msg.name, int):
                    msg.name = str(msg.name)
                
            prompt = self.model.format(
                        *desearlize_prompt_msgs
                    )
            prompt = parse_prompt(prompt)
            
            try:
                res = self.model(
                    prompt,
                    parse_func=self.output_parser.parse_func,
                    max_retries=3,
                )
                output = res.json
                if "fail" in output.keys():
                    return Msg("user",content= f"FAIL TO PARSE",role = "user",fail = True)
                if "actions" in output.keys():
                    actions = output["actions"]
                  
                    if len(actions) > 5:
                        actions = actions[:5] # 每次function call 最多同时执行五个
                        
                    call_tool_msg = Msg(self.name, 
                        role="assistant",
                        func = "call_tool_func",
                        content = actions,
                        finish = False)
                    return call_tool_msg
                   
                elif "return_values" in output.keys():
                    # Record the response in memory
                    msg_finish = Msg(self.name, 
                                    output, 
                                    role="assistant",
                                    finish = True)
                    # To better display the response, we reformat it by json.dumps here
                    self.speak(
                        Msg(self.name, json.dumps(res.raw, indent=4), role="assistant"),
                    )
                    return msg_finish
                else:
                    return Msg("user",content=f"FAIL TO PARSE",role = "user",fail =True)
            except Exception as e:
                print(e)
                return Msg("user",content=f"FAIL TO PARSE",role = "user",fail =True)
            
        return Msg("user",content=f"FAIL TO PARSE",role = "user",fail =True)
