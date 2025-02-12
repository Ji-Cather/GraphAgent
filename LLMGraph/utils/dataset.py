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
