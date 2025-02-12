from . import loader_registry

import pandas as pd
from langchain_community.document_loaders.base import BaseLoader
from typing import (Iterator)
from langchain_core.documents import Document
import os

@loader_registry.register("tweet_loader")
class TweetLoader(BaseLoader):
    """Load a `Dat` file into a list of Documents.
    """

    def __init__(
        self,
        file_path: str,
    ):
        self.file_path = file_path
        self.tweets = None
        self.content_cols = ["text"]
        self.meta_data_cols = ["user_name","user_location","user_description","topic","is_retweet"]
        
    def lazy_load(self) -> Iterator[Document]:
        try:
            yield from self.__read_file()
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e
        
        

    def __read_file(self) -> Iterator[Document]:
        
        assert os.path.exists(self.file_path)
        if self.tweets is None:
            data = pd.read_csv(self.file_path)
            self.tweets = data
        else:data = self.tweets

        for i, row in data.iterrows():
            try:
                contents = [row[content_col] for content_col in self.content_cols]
                content =  "\n".join(contents)
                
                metadata = {
                    k: row[k]
                    for k in self.meta_data_cols
                }
            
                yield Document(page_content=content, metadata=metadata)
            except:
                continue
            
    def add_tweet(self,tweet:dict):
        tweet = pd.DataFrame(tweet)
        if self.tweets == None:
            self.tweets = tweet
        else:
            self.tweets = pd.concat([self.tweets,tweet])