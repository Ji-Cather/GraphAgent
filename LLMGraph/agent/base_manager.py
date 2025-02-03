from LLMGraph.agent.tool_agent import ToolAgent
from agentscope.agents import AgentBase

from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from LLMGraph.tool import create_article_retriever_tool
from typing import List,Dict,Union
from datetime import datetime

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
{
    "thought": "what you thought",
    "speak": "what you said",
    "function": [{"name": "{function name}", "arguments": {"{argument name}": {argument_value}, ...}}, ...]
}

Taking using web_search function as an example, the response should be like this:
{
    "thought": "xxx",
    "speak": "xxx",
    "function": [{"name": "web_search", "arguments": {"query": "what's the weather today?"}}]
}
"""  # noqa




class ManagerAgent(AgentBase):
    """
    The manager agent is responsible for managing the whole process of writing
    a paper. It is responsible for the following tasks:
    1. Generating prompts for the tools.
    2. Generating prompts for the authors.
    3. Generating prompts for the paper.
    4. Writing the paper.
    5. Updating the database.
    Args:
        name (str): The name of the agent.
        model_config_name (str): The name of the model config.
        task_path (str): The path of the task.
        config_path (str): The path of the config.
        cur_time (str): The current time.
        article_write_configs (dict): The configs for writing the paper.
    Attributes:
        name (str): The name of the agent.
        model_config_name (str): The name of the model config.
        task_path (str): The path of the task.
        config_path (str): The path of the config.
        cur_time (str): The current time.
        article_write_configs (dict): The configs for writing the paper.
        tools_prompt (str): The prompt for the tools.
        func_name_mapping: func_map_dict 
        article_manager: manage paper DB.
    """
    
    def __init__(self,
                 name,
                 to_dist = False,
                 **kwargs):
        
        super().__init__(name = name,
                         to_dist = to_dist, **kwargs)
      
        self.tools = {}
        self.tools_prompt = None
        self.func_name_mapping = None


    def update_tools(self,tools:dict = {}):
        self.tools = tools
   
    def prepare_funcs_prompt(self):
        """Convert function descriptions from json schema format to
        string prompt format.

        Args:
            tools (`List[Tuple]`):
                The list of tool functions and their descriptions in JSON
                schema format.

        Returns:
            `Tuple[str, dict]`:
                The string prompt for the tool functions and a function name
                mapping dict.

            .. code-block:: python

                {index}. {function name}: {function description}
                    {argument name} ({argument type}): {argument description}
                    ...

        """
        tools_prompt = []
        func_name_mapping = {}
        tools_descriptions = [tool.describe()
        for tool_name,tool in self.tools.items()
        ]

        for i, desc in enumerate(tools_descriptions):
            func_name = desc["function"]["name"]
            

            func_desc = desc["function"]["description"]
            args_desc = desc["function"]["parameters"]["properties"]

            args_list = [f"{i + 1}. {func_name}: {func_desc}"]
            for args_name, args_info in args_desc.items():
                if "type" in args_info:
                    args_line = (
                        f'\t{args_name} ({args_info["type"]}): '
                        f'{args_info.get("description", "")}'
                    )
                else:
                    args_line = (
                        f'\t{args_name}: {args_info.get("description", "")}'
                    )
                args_list.append(args_line)

            func_name_mapping[func_name] = list(args_desc.keys())
            func_prompt = "\n".join(args_list)
            tools_prompt.append(func_prompt)

        return "\n".join(tools_prompt), func_name_mapping
    

    def get_prompt_tool_msgs(self):
        func_prompt, func_name_mapping = self.prepare_funcs_prompt()
        tools_prompt = DEFAULT_TOOL_PROMPT.format(function_prompt=func_prompt)
        self.tools_prompt = tools_prompt
        self.func_name_mapping = func_name_mapping

        tool_msgs =  [
                    Msg("system", self.tools_prompt, role="system"),
                    Msg("system", TOOL_HINT_PROMPT, role="system"),
                    ]
        return tool_msgs
    


    def call_tool_func(self, function_call_msg:list):
        execute_results = []
        for i,res_one in enumerate(function_call_msg):
            arg_list = self.func_name_mapping[res_one["tool"]]
        
            func_res = self.tools[res_one["tool"]].execute_func(i,arg_list,res_one["tool_input"])
            execute_results.append((res_one,func_res))
                
        return execute_results
    

