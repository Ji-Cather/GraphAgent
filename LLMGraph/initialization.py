import os
from typing import Dict

import yaml

from LLMGraph.manager import manager_registry
from LLMGraph.environments import env_registry
from LLMGraph.memory import memory_registry
import agentscope
from agentscope.models import load_model_by_config_name
from agentscope.message import Msg

def get_arg_config(args):
    
    llm = load_model_by_config_name("default")
    template = """
Here's three kind of human activitys:

1. article writing: related to the generation of citation, authors, co-author networks..

2. movie reviewing: related to the generation of movie reviews..

3. online socialization: related to the generation of social network..

Now here's some information about the network generation task:
{instruction}

Now respond the simulation scenrio you should be simulating. which should be an interger from 1 to 3 (article writing, movie reviewing, online socialization).
Respond only the number:
"""
    prompt = template.format(instruction = args["user_input"])
    prompt_msg = llm.format(Msg("user",prompt,"user"))
    response = llm(prompt_msg)
    content = response.text
    import re
    int_match = re.search(r'\d+', content)
    simulation_number = int(int_match.group())
    simulation_task_map = {
        1:"citeseer",
        2:"movielens",
        3:"tweets"
    }
    try:
        args["task"] = simulation_task_map[simulation_number]
    except:
        print("Invalid simulation for input")
        exit()
    args["config"] = "user"
    args["build"] = True
    print("task: {task}, config: {config}".format(task=  args["task"], config=  args["config"]))
    return args

def load_memory(memory_config: Dict,
                llm,
                llm_loader):
    memory_config["llm"] =llm
    memory_config["llm_loader"] =llm_loader
    return memory_registry.build(memory_config)


def load_environment(env_config: Dict) :
    env_type = env_config.pop('env_type', 'article')
    return env_registry.build(env_type, **env_config)



def load_manager(manager_config: Dict,manager_type) :
    return manager_registry.load_data(manager_type, ** manager_config)
    

    
def prepare_task_config(task,
                        data):
    """Read the yaml config of the given task in `tasks` directory."""
    all_data_dir = os.path.join(os.path.dirname(__file__), 'tasks')
    task_path = os.path.join(all_data_dir,data)
    if not os.path.exists(task_path):
        all_datas = []
        for data_name in os.listdir(all_data_dir):
            if os.path.isdir(os.path.join(all_data_dir, task)) \
                and data_name != "__pycache__":
                all_datas.append(data_name)
        raise ValueError(f"Task {data} not found. Available tasks: {all_datas}")
    
    all_task_dir = os.path.join(task_path, "configs")
    config_path = os.path.join(all_task_dir, task)
    config_path = os.path.join(config_path, 'config.yaml')
    
    if not os.path.exists(config_path):
        all_tasks = []
        for task in os.listdir(all_task_dir):
            if os.path.isdir(os.path.join(all_task_dir, task)) \
                and task != "__pycache__":
                all_tasks.append(task)
        raise ValueError(f"Config {task} not found. Available configs: {all_tasks}")
    
    if not os.path.exists(config_path):
        raise ValueError("You should include the config.yaml file in the task directory")
    task_config = yaml.safe_load(open(config_path))

    return task_config,config_path,task_path


from datetime import datetime, timedelta, date
def update_env_config(env_config,
                      agent_config,
                      env_type:str = "article"):
    try:
        if env_type == "article":
            env_config["article_write_configs"]["min_citations"] = agent_config["min_citations"]
            env_config["article_write_configs"]["max_citations"] = agent_config["max_citations"]
            env_config["article_write_configs"]["use_graph_deg"] = False
            start_time = env_config["time_configs"]["cur_time"]
            simulation_time = timedelta(days=agent_config["time"])
            end_time = start_time + simulation_time
            env_config["time_configs"]["end_time"] = end_time
            env_config["time_configs"]["round_time_delta"] = agent_config["article_time_delta"]
            env_config["time_configs"]["author_time_delta"] = agent_config["author_time_delta"]

            filter_keys = ["big_name","topic", "write_topic"]
            filter_keys = filter_keys[:agent_config["filter_keys_num"]]
            env_config["managers"]["article"]["tool_kwargs"]["filter_keys"] = filter_keys
            env_config["managers"]["article"]["control_profile"]["hub_rate"] = agent_config["hub_rate"]

        elif env_type == "movie":
            env_config["managers"]["movie"]["retriever_kwargs"]["search_kwargs"]["k"] = agent_config["k"]
            env_config["time_configs"]["watcher_num"] = agent_config["watcher_num"]
            start_time = env_config["time_configs"]["cur_time"]
            simulation_time = timedelta(days=agent_config["time"])
            end_time = start_time + simulation_time
            env_config["time_configs"]["end_time"] = end_time
            filter_keys = [ "watched_movie_ids", "interested_genres"]
            filter_keys = filter_keys[:agent_config["filter_keys_num"]]
            env_config["managers"]["movie"]["tool_kwargs"]["filter_keys"] = filter_keys
        
        elif env_type == "social":
            env_config["managers"]["social"]["retriever_kwargs"]["search_kwargs"]["k"] = agent_config["k"]
            start_time = env_config["time_configs"]["cur_time"]
            simulation_time = timedelta(days=agent_config["time"])
            end_time = start_time + simulation_time
            env_config["time_configs"]["end_time"] = end_time
            env_config["time_configs"]["social_time_delta"] = agent_config["social_time_delta"]
            env_config["time_configs"]["people_add_delta"] = agent_config["people_add_delta"]
            env_config["social_configs"]["max_people"] = agent_config["max_people"]
            env_config["social_configs"]["add_people_rate"] = agent_config["add_people_rate"]
            env_config["social_configs"]["delete_people_rate"] = agent_config["delete_people_rate"]
            filter_keys = [ "follow",  "topic", "big_name", "friend"]
            filter_keys = filter_keys[:agent_config["filter_keys_num"]]
            env_config["managers"]["social"]["tool_kwargs"]["filter_keys"] = filter_keys
            env_config["managers"]["social"]["control_profile"]["hub_rate"] = agent_config["hub_rate"]
    except:pass

    return env_config