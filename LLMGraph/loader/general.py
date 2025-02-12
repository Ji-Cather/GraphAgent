from typing import Sequence
from langchain_community.document_loaders.directory import DirectoryLoader

from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, Union
from LLMGraph.utils.dataset import dataset_retrieve_registry
from LLMGraph.memory.manager_memory import manager_memory_registry
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
import os
from LLMGraph import select_to_last_period
import concurrent
import logging
import random
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, Union

from langchain_core.documents import Document
from collections import deque
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

from datetime import datetime, date
import copy 

from LLMGraph.prompt.general import general_prompt_registry
import pandas as pd
import numpy as np
from agentscope.message import Msg






class GeneralGraph:
    def __init__(self, 
                 nodes: pd.DataFrame,
                 edges: pd.DataFrame,
                 add_actor_ids: List[dict] = [],
                 ):
        self.nodes = nodes
        self.edges = edges
        self.add_actor_ids = add_actor_ids
        
    def update_graph(self, edges:List[dict]):
        edges_df = pd.DataFrame(edges)
        self.edges = pd.concat([self.edges, edges_df], ignore_index=True)

class ReferenceGraph:
    def __init__(self, 
                 dataset_name:str = "sephora",
                 time_configs: dict = {},
                 
                 ):
        # TBD for other datasets
        self.dataset_name = dataset_name
        self.node_df, self.edge_df = dataset_retrieve_registry.build(dataset_name) # reference graph original data
        self.time_configs = time_configs
        self.added_node_ids = []
        

    def update_graph(self, 
                     seed_graph: bool = False): # only for seed graph
        time_configs = self.time_configs
        if time_configs["update_method"] == "time":
            if not seed_graph:
                update_edges = self.edge_df[self.edge_df["timestamp"]>self.time_configs["cur_time"] & self.edge_df["timestamp"]<=self.time_configs["cur_time"]+self.time_configs["time_delta"]]
                self.time_configs["cur_time"] = self.time_configs["cur_time"] + self.time_configs["time_delta"]
            else:
                update_edges = self.edge_df[self.edge_df["timestamp"]<=self.time_configs["start_time"]]
                self.time_configs["cur_time"] = self.time_configs["start_time"]
                
        elif time_configs["update_method"] == "edge":
            if not seed_graph:
                update_edges = self.edge_df.iloc[self.time_configs["cur_edge"]:self.time_configs["cur_edge"]+self.time_configs["edge_delta"]]
                self.time_configs["cur_edge"] = self.time_configs["cur_edge"] + self.time_configs["edge_delta"]
            else:
                update_edges = self.edge_df.iloc[:self.time_configs["start_edge"]]
                self.time_configs["cur_edge"] = self.time_configs["start_edge"]
        else:
            raise ValueError(f"Invalid update method: {time_configs['update_method']}")
        
        added_edges = update_edges
        add_actor_ids = update_edges["actor_id"].unique()
        add_item_ids = update_edges["item_id"].unique()
        added_node_ids = list(set(add_actor_ids).union(set(add_item_ids))) 
        added_node_ids = list(filter(lambda x:x not in self.added_node_ids, added_node_ids))   
        added_nodes = self.node_df.loc[self.node_df["node_id"].isin(added_node_ids)]
        

        # for seed graph, general graph is edges before cur time
        # for simulation update, general graph is edges > cur time and < cur time + time gap
        if seed_graph:
            added_edges.loc[:,"seed"] = True
        graph = GeneralGraph(nodes = added_nodes,
                             edges = added_edges,
                             add_actor_ids = add_actor_ids)
        if not seed_graph:
            self.added_node_ids.extend(added_node_ids)
        return graph
    
    def update_delta_graph(self):
        delta_graph = self.update_graph(self.time_configs,
                                        seed_graph = False)
        return delta_graph
        
    

    


class GeneralLoader:
    def __init__(self,
                 dataset_name: str,
                 time_configs: dict,
                 graph_structure,
                 memory_retrieval_method: str = "random_walk",
                 memory_retrieval_params: dict = {},
                 *args,
                 **kargs):
        super().__init__(*args,**kargs)
        # graph structure
            # node:
               # [product, user] # node_types
            # edge:
                # [review] # edge_types
            # item_nodes: [product]
            # actor_nodes: [user]
    
        self.graph_structure = graph_structure
        self.prompt_templates = self.update_prompt_templates()
        self.ref_graph = ReferenceGraph(dataset_name = dataset_name,
                                        time_configs = time_configs)

        # start from seed graph
        self.simulated_graph: GeneralGraph = \
            self.ref_graph.update_graph(seed_graph = True)

        # searlization of ref graph
        self.cur_graph_sealize = self.add_graph(graph = self.simulated_graph)
        self.active_edges = []
        
        self.memory_retrieval_method = manager_memory_registry.build(memory_retrieval_method)
        self.memory_retrieval_params = memory_retrieval_params

    def calculate_avg_degree(self):
        # Get total number of nodes
        num_nodes = len(self.cur_graph_sealize[0])
        if num_nodes == 0:
            raise ValueError("Unavailable environment")
            
        # Calculate degree for each node
        degrees = []
        for node_id in list(self.cur_graph_sealize[0].keys()):
            degree = 0
            # Count edges where node is source or destination
            for edge in self.cur_graph_sealize[1]:
                actor_id = edge["actor_id"]
                item_id = edge["item_id"]
                if node_id == actor_id or node_id == item_id:
                    degree += 1
            degrees.append(degree)
            
        # Calculate average degree
        avg_degree = sum(degrees) / num_nodes
        
        # Calculate variance
        variance = sum((d - avg_degree) ** 2 for d in degrees) / num_nodes
        
        return avg_degree, variance
    
    def update_prompt_templates(self):
        prompt_templates = {}
        for label_type in self.graph_structure["node"]:
            prompt_template = general_prompt_registry.build(f"node_{label_type}")
            prompt_templates[f"node_{label_type}"] = prompt_template
        for label_type in self.graph_structure["edge"]:
            prompt_template = general_prompt_registry.build(f"edge_{label_type}")
            prompt_templates[f"edge_{label_type}"] = prompt_template
        return prompt_templates

    def update_delta_graph(self):
        # add_nodes 一定是先前没有记录的, add_active_node_ids 则不一定
        graph = self.ref_graph.update_graph(seed_graph = False)
        add_nodes, add_adjs, add_active_node_ids = self.add_graph(graph) 
        self.cur_graph_sealize[0].update(add_nodes)
        self.cur_graph_sealize[1].extend(add_adjs)
        self.active_edges = add_adjs
        self.cur_graph_sealize[2] = add_active_node_ids
        add_item_nodes = list(filter(lambda x:x[1]["node_type"] in self.graph_structure["item_nodes"], add_nodes.items()))
        add_item_documents = [Document(page_content=node[1]["node_text"],
                                        metadata={"node_id":node[0],
                                                  "node_type":node[1]["node_type"]}) 
                                        for node in add_item_nodes]
        return add_item_documents
    
    
    

    

    def get_actor_texts(self):
        nodes = self.cur_graph_sealize[0]
        actor_nodes = list(filter(lambda x:x[1]["node_type"] in self.graph_structure["actor_nodes"], nodes.items()))
        actor_texts = [node[1]["node_text"] for node in actor_nodes]
        return actor_texts

    def get_active_agent_ids(self):
        active_agent_ids = self.cur_graph_sealize[2]
        if isinstance(active_agent_ids, np.ndarray):
            active_agent_ids = active_agent_ids.tolist()
        
        return active_agent_ids

    def get_active_agent_id_queue(self):
        edges = self.active_edges
        actor_ids = [(edge["actor_id"], edge["timestamp"]) for edge in edges]
        return actor_ids
    
    def get_item_documents(self):
        nodes = self.cur_graph_sealize[0]
        item_nodes = list(filter(lambda x:x[1]["node_type"] in self.graph_structure["item_nodes"], nodes.items()))
        item_node_documents = [Document(page_content=node[1]["node_text"],
                                        metadata={"node_id":node[0],
                                                  "node_type":node[1]["node_type"]}) 
                                        for node in item_nodes]
        return item_node_documents
    
    def get_actor_edge_adjs(self, actor_id):
        edges = self.cur_graph_sealize[1]
        actor_edge_adjs = edges[actor_id]
        return actor_edge_adjs
    
    def update_interactions(self, interaction_msgs):
        # timestamp, interaction_msg 
        # generate edges
        edge_all = []
        for actor_id, timestamp, interaction_msg in interaction_msgs:
            edges = interaction_msg["return_values"]
            try:
                for action_name, edge_info in edges.items():
                    edge_dict = {
                        "actor_id":actor_id,
                        "item_id":edge_info.pop("node_id"),
                        "edge_type":f"{self.ref_graph.dataset_name}_{action_name}".lower(),
                        "timestamp":timestamp,
                        "seed":False,
                        **edge_info
                    }
                    edge_all.append(edge_dict)
            except:
                # raise ValueError(f"Invalid interaction message: {interaction_msg}")
                continue
        self.simulated_graph.update_graph(edge_all)
            
    

    def add_graph(self, graph):
        nodes = {} # node_id: {"node_text":node_txt, "node_type":node_type}
        
        for row_idx, node_info in graph.nodes.iterrows(): 
            node_info = node_info.to_dict()
            node_text = self.prompt_templates[f"node_{node_info['node_type']}"].format_messages(**node_info)[0].content
            nodes[node_info["node_id"]] = {"node_text":node_text, "node_type":node_info["node_type"]}

        edges = []
        # Process edges from graph
        for row_idx, edge_info in graph.edges.iterrows():
            edge_type = edge_info["edge_type"]
            edge_text = self.prompt_templates[f"edge_{edge_type}"].format_messages(**edge_info)[0].content
            edges.append(
                {"edge_text":edge_text,
                "edge_type":edge_type,
                "actor_id": edge_info["actor_id"],
                "item_id":edge_info["item_id"],
                "timestamp":edge_info["timestamp"]
                }
            )
            
        return [nodes, edges, graph.add_actor_ids]
       
    
    def get_src_node_memory(self, src_node_id, upper_token = 4e3):
        edges = self.cur_graph_sealize[1]
        nodes = self.cur_graph_sealize[0]
        neighbour_node_texts, neighbour_edge_texts = self.memory_retrieval_method(edges, nodes, src_node_id, **self.memory_retrieval_params)
        node_memory_template = PromptTemplate.from_template("""
neighbour_node_texts:
{neighbour_node_texts}

neighbour_edge_texts:
{neighbour_edge_texts}""")
        
        neighbour_node_texts = select_to_last_period("\n".join(neighbour_node_texts), upper_token = upper_token//2)
        neighbour_edge_texts = select_to_last_period("\n".join(neighbour_edge_texts), upper_token = upper_token//2)

        node_memory = node_memory_template.format(neighbour_node_texts = neighbour_node_texts,
                                                  neighbour_edge_texts = neighbour_edge_texts)
        return node_memory

    def get_node_info(self, node_id):
        node_info_text = self.cur_graph_sealize[0][node_id]["node_text"]
        return node_info_text

    
    
    