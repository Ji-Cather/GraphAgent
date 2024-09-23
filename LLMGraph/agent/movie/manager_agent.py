from LLMGraph.agent.tool_agent import ToolAgent
from agentscope.agents import AgentBase
from LLMGraph.agent.base_manager import ManagerAgent
from LLMGraph.manager import MovieManager
from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from LLMGraph.tool import create_movie_retriever_tool
from typing import List,Dict,Union
from datetime import datetime,timedelta,date



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




class MovieManagerAgent(ManagerAgent):

    movie_manager:MovieManager

    def __init__(self,
                 name,
                 movie_data_dir,
                 link_movie_path,
                 generated_data_dir,
                 model_config_name,
                 movie_manager_configs:dict,
                 start_time:str,
                 cur_time:str,
                 movie_time_delta:int,
                 **kwargs
                 ):
        super().__init__(name = name, model_config_name = model_config_name, **kwargs)
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        start_time = datetime.strptime(start_time,"%Y-%m-%d").date()
        movie_time_delta = timedelta(days = movie_time_delta)
        movie_manager = MovieManager.load_data(
            movie_data_dir = movie_data_dir,
            link_movie_path = link_movie_path,
            generated_data_dir = generated_data_dir,
            start_time = start_time,
            cur_time = cur_time,
            movie_time_delta = movie_time_delta,
            **movie_manager_configs
        )
        self.movie_manager = movie_manager
        # retriever = article_manager.get_retriever()
        prompt = PromptTemplate.from_template("""
Title: {Title}
Genres: {Genres}
Content: {page_content}""")
        
        self.tools = {}
        self.document_prompt = prompt
        self.update()



    def update(self,
                online = True,
                interested_genres:list = [],
                watched_movie_ids:list = [],
                **retriever_kargs_update):
     
        retriever_tool_func, retriever_tool_dict = self.movie_manager.get_movie_retriever_tool(
                                                            online = online,
                                                            interested_genres=interested_genres,
                                                            watched_movie_ids=watched_movie_ids,
                                                             **retriever_kargs_update)
        # html_tool_func,html_tool_dict = self.movie_manager.get_movie_html_tool(online = online,
        #                                            **retriever_kargs_update)
        
        tools = {"SearchMovie":ToolAgent( 
                        "SearchMovie",
                        retriever_tool_func,
                        retriever_tool_dict
                        ),
                #  "GetOneMovie":ToolAgent(
                #         "GetOneMovie",
                #         html_tool_func,
                #         html_tool_dict
                #  )
        } # debug

        self.update_tools(tools)
 

    
    def reply(self, message:Msg=None):
        
        func_name_map = {
           "self":["update",
                   "get_docs_len",
                   "get_prompt_tool_msgs",
                   "call_tool_func",
                   "save"],
        }
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        
        if func_name in func_name_map["self"]:
            func = getattr(self,func_name)
        else:
            func = getattr(self.movie_manager,func_name)

        return_values = func(**kwargs)
        return Msg(self.name,content = return_values,role = "assistant")
       
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
        tools_descriptions = [tool.describe() for tool_name,tool in self.tools.items()]

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
    

    def get_docs_len(self):
        return self.movie_manager.get_docs_len()
    
    def save(self,
             cur_time:str,
             start_time:float):
        
        cur_time = datetime.strptime(cur_time, "%Y-%m-%d").date()
        self.movie_manager.save_networkx_graph()
        self.movie_manager.save_infos(cur_time, start_time)

    