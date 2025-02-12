# from agentscope.agents import AgentBase
from typing import List, Tuple
from agentscope.message import Msg
from agentscope.service import ServiceResponse, ServiceExecStatus
from typing import Callable

from LLMGraph.tool.tool_warpper import GraphServiceFactory

DEFAULT_TOOL_PROMPT = """
The following tool functions are available in the format of
```
{{index}}. {{function name}}: {{function description}}
    {{argument name}} ({{argument type}}): {{argument description}}
    ...
```

Tool Functions:
{function_prompt}

Notice:
1. Fully understand the tool function and its arguments before using it.
2. Only use the tool function when it's necessary.
3. Check if the arguments you provided to the tool function is correct in type and value.
4. You can't take some problems for granted. For example, where you are, what's the time now, etc. But you can try to use the tool function to solve the problem.
5. If the function execution fails, you should analyze the error and try to solve it.

"""  # noqa

TOOL_HINT_PROMPT = """
Generate a response in the following format:

Response Format:
You should respond in the following format, which can be loaded by `json.loads` in Python:
{{
    "thought": "what you thought",
    "speak": "what you said",
    "function": [{{"name": "{{function name}}", "arguments": {{"{{argument name}}": {{argument_value}}, ...}}}}, ...]
}}

Taking using web_search function as an example, the response should be like this:
{{
    "thought": "xxx",
    "speak": "xxx",
    "function": [{{"name": "web_search", "arguments": {{"query": "what's the weather today?"}}}}]
}}
"""  # noqa

FUNCTION_RESULT_TITLE_PROMPT = """Execution Results:
"""

FUNCTION_RESULT_PROMPT = """{index}. {function_name}:
    [EXECUTE STATUS]: {status}
    [EXECUTE RESULT]: {result}
"""


class ToolAgent():
    
    def __init__(self, 
                 name: str, 
                 tool_func: Callable,
                 func_desc:dict) -> None:
        self.name = name
        self.tool_func = tool_func
        self.func_desc = func_desc

    
    def reply(self, message:Msg = None):

        if  func_name:= message.get("func",False):
            kwargs = message.get("kwargs",{})
            func = getattr(self,func_name)
            tool_res = func(**kwargs)
            
            return Msg(self.name,
                       content = tool_res,
                       role = "assistant")



    def execute_func(self, 
                     index: int, 
                     arg_list, 
                     func_args:dict ) -> dict:
        """Execute the tool function and return the result.

        Args:
            index (`int`):
                The index of the tool function.
            func_call (`dict`):
                The function call dictionary with keys 'name' and 'arguments'.

        Returns:
            `ServiceResponse`: The execution results.
        """

       
        try:
            func_args = dict(filter(lambda item: item[0] in arg_list, func_args.items()))
            func_res = self.tool_func(**func_args)
        except Exception as e:
            func_res = ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=str(e),
            )

        status = (
            "SUCCESS"
            if func_res.status == ServiceExecStatus.SUCCESS
            else "FAILED"
        )
        if status == "FAILED":
            func_res.content = str(func_res.content)
        # return the result of the function
        return {
            "index": index + 1,
            "function_name": self.name,
            "status": status,
            "result": func_res.content,
        }
    
    def describe(self):
        return self.func_desc