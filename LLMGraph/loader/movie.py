from . import loader_registry

import pandas as pd
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_community.document_loaders.base import BaseLoader
from typing import (List, Optional,Dict,Iterator,Tuple)
from langchain_core.documents import Document

from LLMGraph.tool.movie_html import AsyncMovieHtmlLoader
import numpy as np
from datetime import datetime,timedelta



    
    
@loader_registry.register("movie1d_loader")
class Movie1MDatLoader(BaseLoader):
    """Load a `Dat` file into a list of Documents.
    """

    def __init__(
        self,
        movie_data_array,
        link_movie_path: str,
        cur_time: datetime,
        load_movie_html: bool = True,
    ):

        self.link_data = pd.read_csv(link_movie_path,index_col=0)
        self.mivie_url_base ={
            "movieId":"https://movielens.org/movies/",
            "imdbId":"http://www.imdb.com/title/tt",
            "tmdbId":"https://www.themoviedb.org/movie/"
        }
        self.load_movie_html = load_movie_html
        
        
        self.cur_time = cur_time # 这里记录movie数据库的时间 放到哪里了
        self.data_ptr = 0
        self.movie_data_array = movie_data_array # store 完整的movie array信息 
        self.movies_genre_map = {} # genre: number
        self.docs = [] # 线上movie doc
        self.cur_movie_docs =[] # 如今正在放映的movie doc
        
        
    def load(self) -> List[Document]:
        if len(self.docs) == 0:
            return super().load()
        else:
            if self.data_ptr == len(self.docs): return self.docs
            return self.docs[:self.data_ptr]
        
    def lazy_load(self) -> Iterator[Document]:
        yield from self.__read_file()
        
    def update(self,
               cur_time:datetime):
        if self.data_ptr != len(self.docs):
            assert len(self.cur_movie_docs) + len(self.docs) == self.data_ptr
            self.docs.extend(self.cur_movie_docs)
            self.cur_movie_docs =[]
        
        if self.data_ptr == len(self.movie_data_array):return
        upper_idx = np.argmax(self.movie_data_array[:, -1] > cur_time)
        
        data = self.movie_data_array[self.data_ptr:upper_idx]
        docs = list(self.load_movie_docs_(data))
        
       
        self.cur_movie_docs = docs
        self.data_ptr = upper_idx
        self.cur_time = cur_time
        
    def init(self,
             cur_time:datetime,
            movie_time_delta:timedelta):
        init_time_upper = cur_time - movie_time_delta
        upper_idx = np.argmax(self.movie_data_array[:,-1] > init_time_upper)
        data = self.movie_data_array[self.data_ptr:upper_idx]
        docs = list(self.load_movie_docs_(data))
        self.docs = docs
        self.data_ptr = upper_idx
        self.cur_time = init_time_upper
        
        cnt_upper_idx = np.argmax(self.movie_data_array[:,-1] > cur_time)
        data_cur = self.movie_data_array[upper_idx:cnt_upper_idx]
        cur_movie_docs = list(self.load_movie_docs_(data_cur))
        self.cur_movie_docs = cur_movie_docs
        self.data_ptr = cnt_upper_idx
        self.cur_time = cur_time
        
        # ### debug 
        # if len(self.docs)>5:
        #     self.docs = self.docs[:5]
        # if len(self.cur_movie_docs)>5:
        #     self.cur_movie_docs = self.cur_movie_docs[:5]
        # ###
        
        
    def load_movie_urls_data(self, urls:List[str] =[]):
        docs = AsyncMovieHtmlLoader(urls).load()
        return docs
        
        
    def load_movie_docs_(self,data):
        for i, row in enumerate(data):
            try:
                # assert isinstance(row[-1],datetime)
                metadata = {"Title":row[1].strip(),
                            "Genres":row[2].split("|"),
                            "MovieId":row[0],
                            "Timestamp":row[-1]}
                
                metadata[f"movieId_url"] = self.mivie_url_base["movieId"]+\
                        str(int(row[0]))  
                content = f"{row[1].strip()}, {row[2]}"
                movie_urls = [self.mivie_url_base["movieId"]+\
                        str(int(row[0]))]
                
                for col_name in self.link_data.columns:
                    try:
                        links = self.link_data.loc[row[0]]
                        url = self.mivie_url_base[col_name]+\
                            str(int(links[col_name]))  
                        metadata[f"{col_name}_url"] = url        
                        movie_urls.append(url)
                    except:
                        metadata[f"{col_name}_url"]  = None
                if self.load_movie_html:    
                    docs = self.load_movie_urls_data(movie_urls)
                    docs_content = "\n\n".join(
                        [doc.page_content for doc in docs]
                    )
                    content += "\n\n" + docs_content
                    
            except Exception as e:
                continue
            
            for genre in metadata["Genres"]:
                if genre not in self.movies_genre_map.keys():
                    self.movies_genre_map[genre] =1
                else:
                    self.movies_genre_map[genre] +=1
            
            yield Document(page_content=content, metadata=metadata)

    def __read_file(self) -> Iterator[Document]:
        
        data = self.movie_data_array
        if self.data_ptr != -1 and self.data_ptr < len(data):
            data = data[:self.data_ptr]
        self.data_ptr = len(data)
        return self.load_movie_docs_(data)
        
        
        
            
    def get_movie_description(self) ->str:
        """ return the description of movie 
        and the number of movies available"""
        
        prompt_template = """
There exists {movie_type_num} types of movies: {movie_types}

The number of movie types are listed as follows: {movie_count}
"""
        movie_types = ",".join(list(self.movies_genre_map.keys()))
        
        movie_count = "\n".join(
            [
                f"{k}: {v} movies available"
                for k,v in self.movies_genre_map.items()
            ]
        )
        return prompt_template.format(
            movie_type_num = len(self.movies_genre_map),
            movie_types = movie_types,
            movie_count = movie_count
        )
    
    def get_movie_types(self):
        return list(self.movies_genre_map.keys())
    
    
    