import networkx as nx
import argparse
import os
import pandas as pd
from LLMGraph.utils.dataset import dataset_retrieve_registry
from LLMGraph.prompt.general import general_prompt_registry
import yaml

from LLMGraph.baselines.analysis.mmd import evaluate_sampled_graphs
parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器
parser.add_argument('--config', 
                    type=str, 
                    default="sephora", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="general", 
                    help='The task setting for the LLMGraph')  # 添加参数



class GraphBuilder:
    def __init__(self, graph_structure):
        self.graph_structure = graph_structure
        self.prompt_templates = self.update_prompt_templates()

    def update_prompt_templates(self):
        prompt_templates = {}
        for label_type in self.graph_structure["node"]:
            prompt_template = general_prompt_registry.build(f"node_{label_type}")
            prompt_templates[f"node_{label_type}"] = prompt_template
        for label_type in self.graph_structure["edge"]:
            prompt_template = general_prompt_registry.build(f"edge_{label_type}")
            prompt_templates[f"edge_{label_type}"] = prompt_template
        return prompt_templates

    def build_actor_item_graph(self, edge_df, node_df):
        """
            build a graph from a dataframe with 
            Nodes:
                actor_id, actor_type, actor_text
                item_id, item_type, item_text
            Edges:
                actor_id, item_id, edge_type, edge_text
            
            Return:
                G: a networkx graph
        """

        G = nx.MultiDiGraph()
        actor_node_ids = edge_df["actor_id"].unique()
        item_node_ids = edge_df["item_id"].unique()
        node_ids = [*actor_node_ids, *item_node_ids]
        node_df = node_df[node_df["node_id"].isin(node_ids)]

        for row_idx, node_info in node_df.iterrows():
            node_info = node_info.to_dict()
            node_text = self.prompt_templates[f"node_{node_info['node_type']}"].format_messages(**node_info)[0].content
            G.add_node(node_info["node_id"], node_type=node_info["node_type"], node_text=node_text)
        
        for row_idx, edge_info in edge_df.iterrows():
            edge_info = edge_info.to_dict()
            edge_text = self.prompt_templates[f"edge_{edge_info['edge_type']}"].format_messages(**edge_info)[0].content
            G.add_edge(edge_info["actor_id"], edge_info["item_id"], edge_type=edge_info["edge_type"], edge_text=edge_text)
    
        return G

def evaluate_graph(args):
    config_path = "LLMGraph/tasks/{task}/configs/{config}/config.yaml".format(
        task = args.task,
        config = args.config
    )
    config = yaml.safe_load(open(config_path))
    dataset_name = config["environment"]["managers"]["general"]["dataset_name"]
    df_save_root = "evaluate/general/results"
    os.makedirs(df_save_root,exist_ok=True)
    df_path = os.path.join(df_save_root,f"eval_{dataset_name}.csv")

    node_df, edge_df = dataset_retrieve_registry.build(dataset_name)
    generated_edge_path = os.path.join(os.path.dirname(config_path),
                                          "generated_data",
                                          "edges.csv") 
    generated_edge_df = pd.read_csv(generated_edge_path)
    generated_edge_df_len = len(generated_edge_df)
    ref_edge_df = edge_df.iloc[:generated_edge_df_len]

    graph_builder = GraphBuilder(config["environment"]["managers"]["general"]["graph_structure"])

    ref_graph = graph_builder.build_actor_item_graph(ref_edge_df, node_df)
    generated_graph = graph_builder.build_actor_item_graph(generated_edge_df, node_df)

    # Convert to undirected graph without multi-edges for evaluation
    out = evaluate_sampled_graphs([generated_graph], [ref_graph])
    experiment = "{llm}_{memory}".format(llm = config["environment"]["managers"]["general"]["model_config_name"],
                                          memory = config["environment"]["managers"]["general"]["general_memory_config"]["memory_retrieval_method"])
    if not os.path.exists(df_path):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(df_path)

    for k,v in out.items():
        df.loc[experiment, f"{k}"] = v
    df.loc[experiment, "edges"] = generated_edge_df_len
    df.to_csv(df_path)
    



if __name__ == "__main__":
    args = parser.parse_args()  # 解析参数
    evaluate_graph(args)
    

    