from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import random
from LLMGraph.wrapper.agent_group import GroupDiscussAgent
from LLMGraph.wrapper import GeneralAgentWrapper
from LLMGraph.agent.general import GeneralAgent, GeneralManagerAgent

from datetime import datetime, timedelta, date
from agentscope.message import Msg, PlaceholderMessage

from agentscope.agents.rpc_agent import RpcAgentServerLauncher,RpcAgent
from typing import Any
from tqdm import tqdm

END_TIME = datetime.max
START_TIME = datetime.min
END_EDGE = float('inf')
START_EDGE = 0




@EnvironmentRegistry.register("general")
class GeneralEnvironment(BaseEnvironment):
    """
    A environment implementing the logic of conversation.
    """
    created_agents:dict = {}
    active_agent_ids:list = [] # 可重复，按照queue的顺序来，每次interact one step
    agent_configs:dict = {} # 存储agent的config
    save_encoded_features:bool = False # save encoded features for GNN evaluation
    
    time_configs:dict ={
       "start_time": START_TIME,
       "start_edge": START_EDGE,
       "cur_time": START_TIME,
       "cur_edge": START_EDGE,
       "time_delta": timedelta(days=10),
       "edge_delta": 10,
       "end_edge": END_EDGE,
       "end_time": END_TIME,
       "update_method":"time",# time or node
    }
    
  
    class Config:
        arbitrary_types_allowed = True
    
    

    def __init__(self, 
                 launcher_args:list = [],
                 **kwargs):
        to_dist = len(launcher_args) > 0
        general_manager_configs = kwargs.pop("managers").pop("general")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")
        time_configs = kwargs.pop("time_configs",{})
        
        if to_dist:
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_args[0]["host"],
                    "port":launcher_args[0]["port"]
                }
            }
        else:
            to_dist_kwargs = {}

        model_config_name = general_manager_configs.pop("model_config_name",
                                                        "llama3")
        manager_agent = GeneralManagerAgent(name = "general_manager",
                                     general_manager_configs = general_manager_configs,
                                    model_config_name = model_config_name,
                                    task_path = task_path,
                                    config_path = config_path,
                                    time_configs = time_configs,
                                    **copy.deepcopy(to_dist_kwargs)
                                    )
        
        

        super().__init__(
                        manager_agent = manager_agent,
                        agent_configs = agent_configs,
                        time_configs = time_configs,
                        launcher_args = launcher_args,
                        to_dist = to_dist,
                        **kwargs
                        )
        

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        complete_status = self.call_manager_agent_func(func_name="get_complete_status").content
        if complete_status:
            self.call_manager_agent_func(func_name="save_run_info").content
        return complete_status
        
        

    def init_agent(self, 
                   agent_id,
                    agent_info,
                    agent_memory, launcher_id = 0) -> GeneralAgentWrapper:
       
        if self.to_dist:
            launcher_arg = self.launcher_args[launcher_id]
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_arg["host"],
                    "port":launcher_arg["port"]
                }
            }
        else:
            to_dist_kwargs = {}
        
        dataset_name = self.call_manager_agent_func("return_dataset_name").content
        agent = GeneralAgent(name = agent_id,
                            agent_configs=self.agent_configs,
                            agent_info = agent_info,
                            agent_memory = agent_memory,
                            dataset_name = dataset_name,
                            **copy.deepcopy(to_dist_kwargs))
        
        wrapper_agent = GeneralAgentWrapper(
                            name = agent_id,
                            agent = agent,
                            manager_agent = self.manager_agent,
                            max_tool_iters = 1,
                            **copy.deepcopy(to_dist_kwargs))
        
        return wrapper_agent

    def init_active_agents(self):
        """_summary_

        Args:
            general_number (int, optional): the number of generals to be generated. Defaults to 10.
            author_number (int, optional): the number of authors for topic discussion (per general). Defaults to 5.
        """
        for i, agent_id in enumerate(self.active_agent_ids):
            """initialize agent_wrapper(tool(to_dist), agent(to_dist))"""
            if self.to_dist:
                launcher_id = i% len(self.launcher_args)
            else:
                launcher_id = 0
           
            if agent_id in self.created_agents.keys():
                continue
           
            # get agent base info
            # get aggregation of neighbor agent messages
            agent_info, agent_memory = self.call_manager_agent_func(func_name="get_actor_node_info",
                                                       kwargs={"node_id":agent_id}).content
            

            wrapper_agent = self.init_agent(agent_id,
                                           agent_info,
                                           agent_memory,
                                           launcher_id = launcher_id) 
            self.created_agents[agent_id] = wrapper_agent
        

    def update_graph(self):
        time_configs = self.call_manager_agent_func(func_name="update_retriver").content
        self.time_configs = time_configs
        active_agent_ids = self.call_manager_agent_func(func_name="get_active_agent_ids").content # actor ids only
        self.active_agent_ids = active_agent_ids
        
        
    def interact(self):
        msg_butters = []
        # 这边应该按照 manager的edge指定interact queue
        # TBD
        active_agent_id_queue = self.call_manager_agent_func(
            "get_active_agent_id_queue"
        ).content

        for active_agent_id, active_timestamp in active_agent_id_queue:
            agent = self.created_agents[active_agent_id]
            msg_butters.append((active_agent_id, active_timestamp, self.call_agent_func(agent, "interact")))

        batch_size = 20 # Batch update message
        return_msgs_content = []
        for i in tqdm(range(0, len(msg_butters), batch_size),f"Simulation Round {self.simulation_round}"):
            sub_msg_butters = msg_butters[i:i+batch_size]
            for msg_butter in sub_msg_butters:
                if isinstance(msg_butter[2], PlaceholderMessage):
                    msg_butter[2].update_value()
                return_msgs_content.append((msg_butter[0], msg_butter[1], msg_butter[2].content))

        return return_msgs_content
    
    def update_interactions(self, interaction_msgs):
        self.call_manager_agent_func(func_name="update_interactions",
                                     kwargs = {
                                         "interaction_msgs":interaction_msgs
                            }).content # 初始化,在to_dist之后初始化)

    def step(self):       
        self.update_graph() # update agent ids
        self.init_active_agents()
        msgs = self.interact()
        self.update_interactions(msgs) # update environment retriver
        
            

    def save(self,
             start_time):
        self.call_manager_agent_func(func_name="save",
                                     kwargs = {
                                        "simulation_round":self.simulation_round,
                                        "save_encoded_features":self.save_encoded_features
                                }).content
    