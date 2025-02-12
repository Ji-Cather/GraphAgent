from . import loader_registry


import pandas as pd
from langchain_community.document_loaders.base import BaseLoader
from typing import (Iterator,List)
from langchain_core.documents import Document
import os
import json

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)
            
@loader_registry.register("social_loader")
class SocialLoader(BaseLoader):
    """Load a `Dat` file into a list of Documents.
    """

    def __init__(
        self,
        social_data:pd.DataFrame = None,
        data_path :str = None,
        content_cols:list = ["text"],
        meta_data_cols:list = ["user_index",
                               "owner_user_index", # origin tweet owner
                               "user_name",
                               "topic",
                               "tweet_idx",
                               "action",
                               "origin_tweet_idx"]
    ):
        self.content_cols = content_cols
        self.meta_data_cols = meta_data_cols
        docs = []
        if data_path is not None:
            forum_docs = readinfo(data_path)
            for doc in forum_docs:
                doc = Document(page_content = doc["page_content"],
                               metadata = {k:v for k,v in doc.items() if k != "page_content"})
                docs.append(doc)
        else:
            for i, row in social_data.iterrows():
                try:
                    contents = [row[content_col] for content_col in self.content_cols]
                    content =  "\n".join(contents)
                    metadata = {}
                    for k in row.keys():
                        if k in self.meta_data_cols:
                            metadata[k] = row[k]
                    
                    if "tweet_idx" not in metadata.keys():
                        metadata["tweet_idx"] = i
                        append_metadata = {
                            "tweet_idx":i,
                            "action": "tweet",
                            "origin_tweet_idx": -1,
                        }
                        metadata.update(append_metadata)
                            
                    doc = Document(page_content=content, metadata=metadata)
                    docs.append(doc)
                except Exception as e:
                    pass
        

        self.docs = docs

        


    def load(self) -> Iterator[Document]:
        return self.docs
        
            
    def add_social(self,
                   social_data:pd.DataFrame):
        docs =[]
        idx_prefix = len(self.docs)
        for i, row in social_data.iterrows():
            try:
                contents = [row[content_col] for content_col in self.content_cols]
                content =  "\n".join(contents)
                metadata = {}
                for k in self.meta_data_cols:
                    if k in row.keys():
                        metadata[k] = row[k]
               
                metadata["tweet_idx"] = idx_prefix+i 
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
            except Exception as e:
                pass
        self.docs.extend(docs)
        return docs
        
    def save(self,data_path):
        forum_data = []
        for doc in self.docs:
            forum_data.append({"page_content":doc.page_content,
                               **doc.metadata})
        writeinfo(data_path,forum_data)