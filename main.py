import argparse
import os
# huggingface tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
from LLMGraph.executor import Executor

import time
parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器
import agentscope


parser.add_argument('--config', 
                    type=str, 
                    default="small", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="citeseer", 
                    help='The task setting for the LLMGraph')  # 添加参数

parser.add_argument("--api_path",
                    type=str,
                    default="LLMGraph/llms/api.json",
                    help="The default path of apis json.")

parser.add_argument("--build",
                    action='store_true',
                    default=False,
                    help="start the building process")

parser.add_argument("--save",
                    action='store_true',
                    default=False,
                    help="save the networkx graph")

parser.add_argument("--user_input",
                    type=str, 
                    default="")


parser.add_argument("--test",
                    action='store_true',
                    default=False,
                    help="test program")

parser.add_argument("--eval",
                    action='store_true',
                    default=False,
                    help="evaluate program")

parser.add_argument('--launcher_save_path', 
                    type=str, 
                    default="LLMGraph/llms/launcher_info.json", 
                    help="The path to save launcher info")

args = parser.parse_args()  # 解析参数



if __name__ == "__main__":
    
    args = {**vars(args)}
    import os

    agentscope.init(
        project="llmgraph",
        name="main",
        model_configs="LLMGraph/llms/default_model_configs.json",
        use_monitor=False,
        save_code=False,
        save_api_invoke=False,
    )
    
    if args["user_input"] != "":
        from LLMGraph.initialization import get_arg_config
        args = get_arg_config(args)
    
    if args["build"]:
        executor = Executor.from_task(args)
        executor.run()
        
    if args["save"]:
        args["launcher_save_path"] = "LLMGraph/llms/launcher_info_none.json"
        executor = Executor.from_task(args)
        executor.save()
    
    if args["test"]:
        args["launcher_save_path"] = "LLMGraph/llms/launcher_info_none.json"
        executor = Executor.from_task(args)
        executor.test()

    if args["eval"]:
        args["launcher_save_path"] = "LLMGraph/llms/launcher_info_none.json"
        executor = Executor.from_task(args)
        executor.eval()