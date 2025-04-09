import argparse
import os
from agentscope.agents.rpc_agent import RpcAgentServerLauncher
import agentscope
from LLMGraph.wrapper.agent_group import GroupDiscussAgent
from LLMGraph.agent.article import ArticleAgent, ArticleManagerAgent
from LLMGraph.agent.movie import MovieAgent, MovieManagerAgent
from LLMGraph.agent.social import SocialAgent,SocialManagerAgent
from LLMGraph.agent.general import GeneralAgent, GeneralManagerAgent

from LLMGraph.wrapper import (ArticleAgentWrapper,
                              MovieAgentWrapper,
                              SocialAgentWrapper,
                              GeneralAgentWrapper)
from LLMGraph.utils.io import writeinfo
import random
import multiprocessing

# huggingface tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(description='launcher_builder')  # 创建解析器

parser.add_argument('--launcher_num', 
                    type=int, 
                    default=48, 
                    help="The number of launchers")  # 添加参数

parser.add_argument('--to_dist', 
                    type=bool, 
                    default=True, 
                    help="whether to start the launchers or not")  # 添加参数

parser.add_argument('--launcher_save_path', 
                    type=str, 
                    default="LLMGraph/llms/launcher_info.json", 
                    help="The path to save launcher info")

parser.add_argument('--hosts', 
                    type=list, 
                    default=["localhost"], 
                    help="host list names")




def setup_server(server_launcher:RpcAgentServerLauncher) -> None:
    """Setup rpc server for participant agent"""
    server_launcher.launch(in_subprocess=False)
    server_launcher.wait_until_terminate()


def main(args):
    """
    启动多个RPC代理服务器的主函数。

    参数:
        args (Namespace): 命令行参数，包括启动器数量、是否分布式、保存路径和主机列表。

    功能:
        1. 根据提供的主机列表和启动器数量，创建多个RpcAgentServerLauncher实例。
        2. 将每个启动器的主机和端口信息保存到指定的文件路径。
        3. 启动多个进程，每个进程运行一个RPC服务器，并等待所有进程结束。
    """
    launcher_args = []
    launchers = []
    for i in range(args.launcher_num):
        host = random.choice(args.hosts)
        
        launcher = RpcAgentServerLauncher(
            host=host,
            # choose port automatically
            custom_agents=[ArticleAgent, 
                            ArticleManagerAgent,
                            GroupDiscussAgent,
                            ArticleAgentWrapper,
                            MovieAgent,
                            MovieManagerAgent,
                            MovieAgentWrapper,
                            SocialManagerAgent,
                            SocialAgent,
                            SocialAgentWrapper,
                            GeneralAgent,
                            GeneralManagerAgent,
                            GeneralAgentWrapper],
            local_mode=False,
        )
        launchers.append(launcher)
        launcher_arg = {
            "host":launcher.host,
            "port":launcher.port
        }
        launcher_args.append(launcher_arg)

    writeinfo(args.launcher_save_path,
            launcher_args)

    # 创建多个进程，每个进程执行函数一次，并传入不同的参数
    processes = []
    for i in range(args.launcher_num):
        p = multiprocessing.Process(target=setup_server, 
                                    args=(launchers[i],))
        processes.append(p)
        p.start()

    # 等待所有进程执行结束
    for p in processes:
        p.join()

if __name__ == "__main__":
    args = parser.parse_args()  # 解析参数
    # setup 所有的server都要init一遍吗？
    agentscope.init(
        project="llmgraph",
        name="server",
        model_configs="LLMGraph/llms/default_model_configs.json",
        use_monitor=False,
        save_code=False,
        save_api_invoke=False,
    )
    main(args)