from agentscope.message import Msg
from typing import Union
from LLMGraph.prompt import MODEL

def count_prompt_len(prompt: Union[str,dict,list]):
    # if isinstance(prompt,str):
    #     return len(prompt)
    # elif isinstance(prompt,Msg):
    #     return len(prompt.content)
    # elif isinstance(prompt,dict):
    #     return sum([len(prompt["content"]) for key in prompt])
    # elif isinstance(prompt,list): # list
    #     if len(prompt) == 0:
    #         return 0
    #     return sum([count_prompt_len(prompt_msg) for prompt_msg in prompt])
    return sum([len(prompt_msg.get("content","")) for prompt_msg in prompt])

def select_to_last_period(s, upper_token = 4e3):
    upper_token = int(upper_token)
    s = s[-upper_token:]
    # 查找最后一个句号的位置
    last_period_index = s.rfind('.')
    # 如果找到句号，返回从开始到最后一个句号之前的部分
    if last_period_index != -1:
        return s[:last_period_index]
    else:
        # 如果没有找到句号，返回整个字符串
        return s

def parse_prompt(prompt, upper_token=7e3):
    if MODEL == "llama":
        return prompt
    #     if count_prompt_len(prompt) <upper_token:
    #         return prompt
    #     upper_token = int(upper_token)
    #     assert isinstance(prompt,list)
    #     reduce_token = count_prompt_len(prompt) - upper_token
    #     try:
    #         # prompt = [prompt_msg.get("content","") for prompt_msg in prompt]
    #         prompt_last = prompt[-1]
    #         assert isinstance(prompt_last,dict)
    #         content = prompt_last.get("content","")[-len(prompt_last)+reduce_token:]
    #         prompt_last["content"] = select_to_last_period(content)
    #         if len(prompt)>1:
    #             prompt_left = prompt[:-1]
    #             return [*prompt_left,prompt_last]
    #         else:
    #             return [prompt_last]
    #     except Exception as e:
    #         print(e)
    #         return prompt
    return prompt