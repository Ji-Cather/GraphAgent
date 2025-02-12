""" load the basic infos of authors"""

import os
from LLMGraph.manager.base import BaseManager
from agentscope.message import Msg
from . import manager_registry as ManagerRgistry
from typing import List,Union,Any
from copy import deepcopy
from langchain_community.document_loaders import TextLoader
from LLMGraph.loader.general import GeneralLoader

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import random
from LLMGraph import select_to_last_period
import networkx as nx
from LLMGraph.utils.io import readinfo,writeinfo
from LLMGraph.retriever import retriever_registry

from LLMGraph.output_parser import find_and_load_json

from datetime import datetime,date,timedelta
import copy
from agentscope.models import  ModelWrapperBase
import time
import numpy as np
import wandb

@ManagerRgistry.register("general")
class GeneralManager(BaseManager):
    """
        manage infos of different community.
    """
    loader: Union[GeneralLoader,None] # item environment
    db: Any
    simulation_time: int = 0 # execution time
    llm: ModelWrapperBase
    embeddings: Any
    general_memory_config: dict = {
        "reflect_memory": False,
        "memory_retrieval_method": "random_walk",
        "memory_retrieval_params": {
            "walk_length": 3,
            "num_walks": 10
        }
    } # the memory config of general memory
    retriever : Any
    generated_data_dir:str
    run:Any

    class Config:
        arbitrary_types_allowed = True
    
    
    
    @classmethod
    def load_data(cls,
                  dataset_name,
                  task_path,
                  config_path,
                  retriever_kwargs,
                  llm,
                  time_configs,
                  graph_structure,
                  embedding_model_name,
                  general_memory_config
                  ):
        
        retriever_kwargs = copy.deepcopy(retriever_kwargs)
        generated_data_dir = os.path.join(os.path.dirname(config_path),
                                          "generated_data") 
        loader = GeneralLoader(
                        dataset_name=dataset_name,
                        time_configs=time_configs,
                        graph_structure=graph_structure,
                        memory_retrieval_method=general_memory_config["memory_retrieval_method"],
                        memory_retrieval_params=general_memory_config["memory_retrieval_params"],
                        )
        
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        if not os.path.exists(generated_data_dir):
            os.makedirs(generated_data_dir)
        run = wandb.init(project="LLMGraph_{dn}".format(
            dn = dataset_name),
            dir = generated_data_dir,
            mode="offline"
            )

        return cls(
           retriever = None,
           retriever_kwargs = retriever_kwargs, 
           db = None,
           loader = loader,
           embeddings = embeddings,
           llm = llm,
           general_memory_config = general_memory_config,
           generated_data_dir = generated_data_dir,
           run = run
           )
    
    def update_memory(self, msgs_content):
        self.loader.update_interactions(msgs_content)

    def update_interactions(self, interaction_msgs):
        self.loader.update_interactions(interaction_msgs)

    def update_retriver(self):
        if self.db is None:
            item_documents = self.loader.get_item_documents()
            self.db = FAISS.from_documents(item_documents, self.embeddings)
        else:
            add_item_documents = self.loader.update_delta_graph()
            if len(add_item_documents) > 0:
                db_update = FAISS.from_documents(add_item_documents, self.embeddings)
                self.db.merge_from(db_update)
            
        self.retriever_kwargs["vectorstore"] = self.db
        self.retriever = retriever_registry.from_db(**self.retriever_kwargs)
        return self.loader.ref_graph.time_configs

    def get_active_agent_ids(self):
        active_agent_ids = self.loader.get_active_agent_ids()
        return active_agent_ids

    def get_actor_node_info(self, node_id):
        actor_node_info = self.loader.get_node_info(node_id)
        actor_node_memory = self.loader.get_src_node_memory(node_id, upper_token = 4e3)
        
        if self.general_memory_config["reflect_memory"]:
            template = """  
        You are a helpful assistant.
        You are given the following information:
        {actor_node_info}
        You are also given the following memory:
        {actor_node_memory}
        Please reflect on the information and memory, and provide a summary of the actor's information and memory.
        """
            prompt_inputs = {
                "actor_node_info": actor_node_info,
                "actor_node_memory": actor_node_memory
            }
            prompt = template.format_map(prompt_inputs)
            prompt = self.llm.format(
                Msg("system","You're a helpful assistant","system"),
                Msg("user",prompt,"user"))
            actor_node_memory = self.llm(prompt).content
            actor_node_memory = select_to_last_period(actor_node_memory, upper_token = 4e3)

        return actor_node_info, actor_node_memory

    def get_complete_status(self):
        time_configs = self.loader.ref_graph.time_configs
        if time_configs["update_method"] == "time":
            return time_configs["cur_time"] >= time_configs["end_time"]
        elif time_configs["update_method"] == "edge":
            return time_configs["cur_edge"] >= time_configs["end_edge"]
        else:
            raise ValueError(f"Unknown update method: {time_configs['update_method']}")

    def return_dataset_name(self):
        return self.loader.ref_graph.dataset_name

    def get_active_agent_id_queue(self):
        return self.loader.get_active_agent_id_queue()

    def save(self, 
            simulation_round:int,
            save_encoded_features:bool = False):
        generated_edges = self.loader.simulated_graph.edges
        generated_edges.to_csv(os.path.join(self.generated_data_dir,
        "edges.csv"))
        generated_edge_len = generated_edges[generated_edges["seed"] == False].shape[0]
        self.run.log({
            "simulation_round": simulation_round,
            "simulation_edge_len": generated_edge_len,
        })

    def save_run_info(self):
        self.run.finish()