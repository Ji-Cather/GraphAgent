# import os
# import pandas as pd
# from LLMGraph.registry import Registry
# from datetime import datetime
# dataset_retrieve_registry = Registry(name="DatasetRetrieveRegistry")

# @dataset_retrieve_registry.register("sephora")
# def load_sephora():
#     # node_df, edge_df = load_cache("sephora")
#     # if node_df is not None:
#     #     return node_df, edge_df
#     root_dir = os.path.join("LLMGraph/tasks/general", "data", "sephora")
#     node_df = pd.read_csv(os.path.join(root_dir, "node_sephora.csv"),index_col=0)
#     edge_df = pd.read_csv(os.path.join(root_dir, "edge_sephora.csv"),index_col=0)
    
#     edge_df["actor_id"] = edge_df["author_node_id"]
#     edge_df["item_id"] = edge_df["product_node_id"]
#     edge_df["original_edge_idx"] = edge_df.index
    
#     str_cols = ["node_id", "actor_id", "item_id"]
#     for col in str_cols:
#         if col in node_df.columns:
#             node_df[col] = node_df[col].astype(str)
#         if col in edge_df.columns:
#             edge_df[col] = edge_df[col].astype(str)
    
#     save_cache("sephora", node_df, edge_df)
#     return node_df, edge_df

# @dataset_retrieve_registry.register("dianping")
# def load_dianping():
#     # node_df, edge_df = load_cache("dianping")
#     # if node_df is not None:
#     #     return node_df, edge_df
#     root_dir = os.path.join("LLMGraph/tasks/general", "data", "Dianping")
#     node_df = pd.read_csv(os.path.join(root_dir, "node_dianping.csv"),index_col=0)
#     edge_df = pd.read_csv(os.path.join(root_dir, "edge_dianping.csv"),index_col=0)

#     edge_df["actor_id"] = edge_df['user_node_id']
#     edge_df["item_id"] = edge_df['business_node_id']
#     edge_df["original_edge_idx"] = edge_df.index
    
#     str_cols = ["node_id", "actor_id", "item_id"]
#     for col in str_cols:
#         if col in node_df.columns:
#             node_df[col] = node_df[col].astype(str)
#         if col in edge_df.columns:
#             edge_df[col] = edge_df[col].astype(str)
    
#     save_cache("dianping", node_df, edge_df)
#     return node_df, edge_df

# @dataset_retrieve_registry.register("imdb")
# def load_imdb():
#     # node_df, edge_df = load_cache("imdb")
#     # if node_df is not None:
#     #     return node_df, edge_df
#     root_dir = os.path.join("LLMGraph/tasks/general", "data", "IMDB")
#     node_df = pd.read_csv(os.path.join(root_dir, "node_imdb.csv"),index_col=0)
#     edge_df = pd.read_csv(os.path.join(root_dir, "edge_imdb.csv"),index_col=0)
    
#     edge_df["actor_id"] = edge_df["actor_a_id"]
#     edge_df["item_id"] = edge_df["actor_b_id"]
#     edge_df["original_edge_idx"] = edge_df.index
    
#     str_cols = ["node_id", "actor_id", "item_id"]
#     for col in str_cols:
#         if col in node_df.columns:
#             node_df[col] = node_df[col].astype(str)
#         if col in edge_df.columns:
#             edge_df[col] = edge_df[col].astype(str)
#     save_cache("imdb",node_df, edge_df)
#     return node_df, edge_df

# def _load_sagraph(sub_dataset_name):
#     # node_df, edge_df = load_cache("sagraph_{}".format(sub_dataset_name))
#     # if node_df is not None:
#     #     return node_df, edge_df
#     root_dir = os.path.join("LLMGraph/tasks/general", "data", "SAGraph_{}".format(sub_dataset_name))
#     node_df = pd.read_csv(os.path.join(root_dir, "node_sagraph.csv"),index_col=0)
#     edge_df = pd.read_csv(os.path.join(root_dir, "edge_sagraph.csv"),index_col=0)
    
#     edge_df["actor_id"] = edge_df["source_user_id"]
#     edge_df["item_id"] = edge_df["destination_user_id"]
#     edge_df["original_edge_idx"] = edge_df.index
    
#     str_cols = ["node_id", "actor_id", "item_id"]
#     for col in str_cols:
#         if col in node_df.columns:
#             node_df[col] = node_df[col].astype(str)
#         if col in edge_df.columns:
#             edge_df[col] = edge_df[col].astype(str)

#     save_cache("sagraph_{}".format(sub_dataset_name),node_df, edge_df)
#     return node_df, edge_df

# @dataset_retrieve_registry.register("sagraph_sbs")
# def load_sagraph_et():
#     return _load_sagraph("ABC")

# @dataset_retrieve_registry.register("sagraph_et")
# def load_sagraph_et():
#     return _load_sagraph("ET")

# @dataset_retrieve_registry.register("sagraph_ifs")
# def load_sagraph_et():
#     return _load_sagraph("IFS")

# @dataset_retrieve_registry.register("sagraph_rfc")
# def load_sagraph_et():
#     return _load_sagraph("RFC")

# @dataset_retrieve_registry.register("sagraph_sbs")
# def load_sagraph_et():
#     return _load_sagraph("SBS")

# @dataset_retrieve_registry.register("sagraph_sbs")
# def load_sagraph_et():
#     return _load_sagraph("ST")
    

# @dataset_retrieve_registry.register("wikirevision")
# def load_wikirevision():
#     # node_df, edge_df = load_cache("wikirevision")
#     # if node_df is not None:
#     #     return node_df, edge_df
#     root_dir = os.path.join("LLMGraph/tasks/general", "data", "WikiRevision")
#     node_df = pd.read_csv(os.path.join(root_dir, "node_wikirevision.csv"),index_col=0)
#     edge_df = pd.read_csv(os.path.join(root_dir, "edge_wikirevision.csv"),index_col=0)
    
#     edge_df["actor_id"] = edge_df["user_node_id"]
#     edge_df["item_id"] = edge_df["wikipage_node_id"]
#     edge_df["original_edge_idx"] = edge_df.index
    
#     str_cols = ["node_id", "actor_id", "item_id"]
#     for col in str_cols:
#         if col in node_df.columns:
#             node_df[col] = node_df[col].astype(str)
#         if col in edge_df.columns:
#             edge_df[col] = edge_df[col].astype(str)
    
#     save_cache("wikirevision", node_df, edge_df)
#     return node_df, edge_df

# @dataset_retrieve_registry.register("wikilife")
# def load_wikilife():
#     # node_df, edge_df = load_cache("wikilife")
#     # if node_df is not None:
#     #     return node_df, edge_df
#     root_dir = os.path.join("LLMGraph/tasks/general", "data", "WikiLifeTrajectory")
#     node_df = pd.read_csv(os.path.join(root_dir, "node_wikilife.csv"),index_col=0)
#     edge_df = pd.read_csv(os.path.join(root_dir, "edge_wikilife.csv"),index_col=0)
    
#     edge_df["actor_id"] = edge_df["person_node_id"]
#     edge_df["item_id"] = edge_df["location_node_id"]
#     edge_df["original_edge_idx"] = edge_df.index

#     str_cols = ["node_id", "actor_id", "item_id"]
#     for col in str_cols:
#         if col in node_df.columns:
#             node_df[col] = node_df[col].astype(str)
#         if col in edge_df.columns:
#             edge_df[col] = edge_df[col].astype(str)
    
#     save_cache("wikilife", node_df, edge_df)
#     return node_df, edge_df

# def save_cache(dataset_name, node_df, edge_df):
#     save_root = os.path.join("LLMGraph/tasks/general", "data", "cache")
#     os.makedirs(save_root, exist_ok=True)
#     node_df.to_csv(os.path.join("LLMGraph/tasks/general", "data", "cache", f"{dataset_name}_node.csv"), index=False)
#     edge_df.to_csv(os.path.join("LLMGraph/tasks/general", "data", "cache", f"{dataset_name}_edge.csv"), index=False)

# def load_cache(dataset_name):
#     if os.path.exists(os.path.join("LLMGraph/tasks/general", "data", "cache", f"{dataset_name}_node.csv")):
#         node_df = pd.read_csv(os.path.join("LLMGraph/tasks/general", "data", "cache", f"{dataset_name}_node.csv"))
#         edge_df = pd.read_csv(os.path.join("LLMGraph/tasks/general", "data", "cache", f"{dataset_name}_edge.csv"))
#         str_cols = ["node_id", "actor_id", "item_id"]
#         for col in str_cols:
#             if col in node_df.columns:
#                 node_df[col] = node_df[col].astype(str)
#             if col in edge_df.columns:
#                 edge_df[col] = edge_df[col].astype(str)
#         return node_df, edge_df
#     else:
#         return None, None

# if __name__ == "__main__":
    
#     node_df, edge_df = dataset_retrieve_registry.build("wikilife")
#     print(node_df.head(1))
#     print(edge_df.head(1))


import os
import pandas as pd
from LLMGraph.registry import Registry
dataset_retrieve_registry = Registry(name="DatasetRetrieveRegistry")

@dataset_retrieve_registry.register("sephora")
def load_sephora():
    root_dir = os.path.join("LLMGraph/tasks/general", "data", "sephora")
    product_df = pd.read_csv(os.path.join(root_dir, "Product.csv"),index_col=0)
    review_df = pd.read_csv(os.path.join(root_dir, "Review.csv"))
    review_df.rename(columns={"submission_time": "timestamp"}, inplace=True)
    review_df["timestamp"] = pd.to_datetime(review_df["timestamp"])
    review_df.sort_values(by="timestamp",inplace=True)
    
    user_df = pd.read_csv(os.path.join(root_dir, "User.csv"),index_col=0)
    product_df["node_type"] = "sephora_product"
    user_df["node_type"] = "sephora_author"
    user_df["node_id"] = list(map(str, range(product_df.shape[0], product_df.shape[0] + user_df.shape[0])))
    product_df["node_id"] = list(map(str, range(0, product_df.shape[0])))
    node_df = pd.concat([user_df, product_df], axis=0)
    

    review_df["actor_id"] = review_df["author_id"].map(dict(zip(user_df.index, user_df["node_id"])))
    review_df["item_id"] = review_df["product_id"].map(dict(zip(product_df.index, product_df["node_id"])))
    review_df["edge_type"] = "sephora_review"
    # filter 不存在于node_df/product_df的review: actor_id and item_id 不是none
    review_df = review_df[review_df["actor_id"].notna() & review_df["item_id"].notna()]
    return node_df, review_df

@dataset_retrieve_registry.register("dianping")
def load_dianping():
    root_dir = os.path.join("LLMGraph/tasks/general", "data", "Dianping")
    product_df = pd.read_csv(os.path.join(root_dir, "Businesses_3core.csv"),index_col=0)
    review_df = pd.read_csv(os.path.join(root_dir, "Reviews_filtered_rest_3core.csv"))
    review_df.rename(columns={"time": "timestamp"}, inplace=True)
    review_df.sort_values(by="timestamp",inplace=True)
    
    user_df = pd.read_csv(os.path.join(root_dir, "User_3core.csv"),index_col=0)
    product_df["node_type"] = "dianping_business"
    user_df["node_type"] = "dianping_user"
    user_df["node_id"] = list(map(str, range(product_df.shape[0], product_df.shape[0] + user_df.shape[0])))
    product_df["node_id"] = list(map(str, range(0, product_df.shape[0])))
    node_df = pd.concat([user_df, product_df], axis=0)
    

    review_df["actor_id"] = review_df['userId'].map(dict(zip(user_df.index, user_df["node_id"])))
    review_df["item_id"] = review_df['restId'].map(dict(zip(product_df.index, product_df["node_id"])))
    review_df["edge_type"] = "dianping_review"
    # filter 不存在于node_df/product_df的review: actor_id and item_id 不是none
    review_df = review_df[review_df["actor_id"].notna() & review_df["item_id"].notna()]
    return node_df, review_df

if __name__ == "__main__":
    node_df, review_df = dataset_retrieve_registry.build("sephora")
    print(node_df)
    print(review_df)

    node_df, review_df = dataset_retrieve_registry.build("sephora")
    print(node_df)
    print(review_df)