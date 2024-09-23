from abc import abstractmethod
from typing import Dict, List,Union

from pydantic import BaseModel, Field

from LLMGraph.message import Message
from . import movie_memory_registry
import pandas as pd
from datetime import datetime

from LLMGraph.utils.process_time import transfer_time
from .. import select_to_last_period
import copy
@movie_memory_registry.register("movie_memory")
class MovieMemory(BaseModel):

    name: str # 标记是谁的记忆
    
    ratings:dict = {
        
    } # movie_id: {name, thought, rating, timestamp} 
    
    rating_counts:dict = {} # genre: [rating]
    
    
    def __init__(self,
                 rating_counts:dict,
                 ratings:dict,
                 **kwargs):
        for movie_id,rating in ratings.items():
            rating['timestamp'] = transfer_time(rating['timestamp'])
        super().__init__(rating_counts = rating_counts,
                         ratings = ratings,
                         **kwargs)
    
    """store the thought and rating scores each agent give to movies"""
    def add_message(self, messages: List[Message]) -> None:
        pass


    def to_string(self) -> str:
        pass


    def reset(self) -> None:
        pass
    
    def update_rating(self,
                      movie_rating:dict):
        movie_rating = copy.deepcopy(movie_rating)
        movie_id = movie_rating["movie_id"]
        if isinstance(movie_rating['timestamp'],str):
            movie_rating['timestamp'] = transfer_time(movie_rating['timestamp']) 
        self.ratings[movie_id] = movie_rating
        
    
    def get_watched_movie_ids(self) -> list:
        return list(self.ratings.keys())
    
    def get_watched_movie_names(self) -> list:
        return [movie_info["movie"] for movie_info in self.ratings.values()]
    
    def retrieve_rating_counts_memory(self,
                                    upper_token = 1e3) -> str:
        
        template = """
You have watched {num_genre} genres of movies. \
Here's the movie genere and the corresbonding rating score you give to each movie:
{movie_ratings}
"""
        num_genre = len(self.rating_counts)
        movie_ratings = "\n".join([
           f"{genre}: {int(sum(score)/len(score))}" for genre,score in self.rating_counts.items()
        ])
        memory = template.format(
            num_genre = num_genre,
            movie_ratings = movie_ratings
        )

        if len(memory)> upper_token:
            return select_to_last_period(memory, upper_token)
        return memory

    def retrieve_movie_ratings_memory(self,
                                      topk = 10, # 选择最邻近的10条
                                      upper_token = 1e3)->str:
        movie_template = """
You watched movie <{movie}>. And you thought "{thought}"\
You give a rating score of {score} for <{movie}>.
"""     
        if len(self.ratings)==0:return "You haven't watched any movies yet"
        movie_rating_list = []
        rating_items = list(self.ratings.items())
        if topk < len(rating_items):
            rating_items = rating_items[:topk]
        rating_items = sorted(rating_items,key = lambda item: item[-1]['timestamp'],reverse= True)
        
            
        for movie_id, rating_movie in rating_items:
            movie_rating_str = movie_template.format(
                movie = rating_movie["movie"],
                score = rating_movie["rating"],
                thought = rating_movie["thought"]
            )
            movie_rating_list.append(movie_rating_str)
        rating_movie = "\n".join(movie_rating_list)

        if len(rating_movie)> upper_token:
            rating_movie = select_to_last_period(rating_movie, upper_token)
        
        return """
Here's your memory of rating movies: 
{rating_movie}


You should watch some other movies, and give the rating score in the following process.
""".format(rating_movie = rating_movie)

    def retrieve_movie_memory(self):
        rating_counts = self.retrieve_rating_counts_memory()
        rating_movies = self.retrieve_movie_ratings_memory()
        return rating_counts +"\n\n"+rating_movies