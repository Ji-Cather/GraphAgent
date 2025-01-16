from citation_llm_graphs import generate_citation_graphs
from pathlib import Path
import pickle
import numpy as np
import networkx as nx
graph_generator = (
            generate_citation_graphs,
            
        )
key = "llmcitationciteseer"
data_path = Path("./data")

train_size = 160
val_size = 32
test_size = 20
min_size = 64
max_size = 512


train = graph_generator(
            num_graphs=train_size,
            min_size=min_size,
            max_size=max_size,
            dataset="train",
            seed=0,
        )
train = list(filter(lambda G:len(G.subgraph(max(nx.connected_components(G), key=len)).nodes())> 2, train))
        
validation = graph_generator(
    num_graphs=val_size,
    min_size=min_size,
    max_size=max_size,
    dataset="val",
    seed=1,
)
test = graph_generator(
    num_graphs=test_size,
    min_size=min_size,
    max_size=max_size,
    dataset="test",
    seed=2,
)
dataset = {
    "train": train,
    "val": validation,
    "test": test,
}

# from scipy.sparse import coo_array, csr_array, eye
# adjs = [nx.to_scipy_sparse_array(G, dtype=np.float64) for G in train]
# adj_len = [csr_array(adj, dtype=np.float64).shape[0] for adj
#            in adjs]
# save the dataset
# with open(data_path / "llmcitation.pkl", "wb") as f:
#     pickle.dump(dataset, f)
with open(data_path / f"llm{key}.pkl", "wb") as f:
    pickle.dump(dataset, f)