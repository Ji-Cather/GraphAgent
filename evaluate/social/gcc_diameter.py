import pandas as pd
import networkx as nx
from evaluate.matrix.base_info import calculate_effective_diameter

def calculate_lcc_proportion(DG:nx.DiGraph,
                             graph_name):
    df = pd.DataFrame()
    if nx.is_directed(DG):
            largest_cc = max(nx.strongly_connected_components(DG), key=len)
            relative_size = len(largest_cc) / DG.number_of_nodes()
    else:
        largest_cc = max(nx.connected_components(DG), key=len)
        relative_size = len(largest_cc) / DG.number_of_nodes()
    largest_cc = DG.subgraph(largest_cc)
    diameter = calculate_effective_diameter(largest_cc)
    df.loc[graph_name, "diameter"] = diameter
    df.loc[graph_name, "relative_size"] = relative_size
    return df