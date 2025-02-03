import pandas as pd
from .community import get_graph_matrix
from .control import calculate_control_matrix
from .mmd import calculate_mmd_matrix
from .base_info import calculate_DG_base_indicators
import os
import networkx as nx






def calculate_directed_graph_matrix(
                      graph_generated,
                      graph_name:str,
                      type="article",
                      calculate_matrix = [
                          "base_info",
                          "control",
                          "community",
                      ]) -> pd.DataFrame:
    
    dfs = []

    if "community" in calculate_matrix:
        try:
            df = get_graph_matrix(graph_generated, 
                                graph_name)
            dfs.append(df)
        except:pass
    
    if "control" in calculate_matrix:
        pass # to be done
        # if isinstance(graph_generated, nx.DiGraph):
        #     graph_reverse = graph_generated.reverse()
        # else:
        #     graph_reverse = graph_generated
        # df = calculate_control_matrix(graph_reverse,
        #                               graph_name)
        # dfs.append(df)

    if "base_info" in calculate_matrix:
        df = calculate_DG_base_indicators(graph_generated,
                                          graph_name,
                                          type)
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    
    return df
