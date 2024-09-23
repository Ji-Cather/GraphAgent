from langchain_core.prompts import BasePromptTemplate, PromptTemplate

from langchain_core.retrievers import BaseRetriever


from functools import partial
from typing import Optional
import random
from . import RetrieverInput
from LLMGraph.tool.movie_html import AsyncMovieHtmlLoader

from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.pydantic_v1 import BaseModel, Field
import functools
from langchain_openai import ChatOpenAI
import re
from .tool_warpper import GraphServiceFactory
from agentscope.service import ServiceResponse, ServiceExecStatus
def get_llm_vedio_summary(llm,
                          movie_name,
                          urls:list =[]):
    """The urls are vedio links for movie <movie_name>
We refer to the llm for detailed explaination of the vedio content"""
    
    template = """
Here's the vedio url for movie <{movie_name}>: {url}.
Please explain and summarize the content of this page. \
And you should answer the following questions:

1. What is the fun of the internal video? 
2. What is the plot of this movie trailer?

If you are unable to answer this question, provide your response in the following format:
Understand: False

If you can answer this question, provide your answer in the following format:
Understand: True
Response: (Your answer to the questions, try your best to answer this question）    
"""
    prompt_template =  PromptTemplate.from_template(template= template)
    for url in urls:
        prompt_inputs = {
            "movie_name":movie_name,
            "url":url
        }
        prompt = prompt_template.format(**prompt_inputs)
        response = llm.invoke(prompt).content
        response += "\n"
        regex = r"Understand\s*\d*\s*:(.*?)\n"
        
        try:
            output = re.search(regex, response, re.DOTALL|re.IGNORECASE)
            understand = output.group(1).strip()
            if not "true" in understand.lower():
                continue
            else:
                regex = r"Response\s*\d*\s*:(.*)"
                output = re.search(response,regex, re.DOTALL|re.IGNORECASE).group(1)
                template = """
Here's something you get from the movie vedio about {movie_name}:
{output}            
"""
                if output is not None: return template.format(movie_name = movie_name,
                                                            output = output)
        except: continue
    return ""



def get_llm_summary(llm,
                    movie_name):
    """The urls are vedio links for movie <movie_name>
We refer to the llm for detailed explaination of the vedio content"""
    
    template = """
ell me something about the movie <{movie_name}> based on your knowledge database. And please explain and summarize the content of the movie and you should answer the following questions:

1. The overall introduction of this movie 
2. What is the plot of this movie trailer?
3. Other unique features of this movie

If you are unable to answer this question, provide your response in the following format:
Understand: False

If you can answer this question, provide your answer in the following format:
Understand: True
Response: (Your answer to the questions, try your best to answer this question）   
"""
    prompt_template =  PromptTemplate.from_template(template= template)
    
    prompt_inputs = {
        "movie_name":movie_name,
    }
    prompt = prompt_template.format(**prompt_inputs)
    response = llm.invoke(prompt).content
    response += "\n"
    regex = r"Understand\s*\d*\s*:(.*?)\n"
    
    try:
        output = re.search(regex, response, re.DOTALL|re.IGNORECASE)
        understand = output.group(1).strip()
        if not "true" in understand.lower():
            return ""
        else:
            regex = r"Response\s*\d*\s*:(.*)"
            output = re.search(regex,response, re.DOTALL|re.IGNORECASE).group(1)
            template = """
Here's something you know about {movie_name}:
{output}            
"""
            if output is not None: return template.format(movie_name = movie_name,
                                                          output = output)
    except: return ""


def get_movie_html(query,
          movie,
          movie_scores,
          url_keys:list = [
                            # "movieId_url", 这个有反爬
                           "imdbId_url",
                           "tmdbId_url"],
          upper_token = 1000):
    response = ""
    try:
        urls = [movie.metadata.get(k) for k in url_keys]
        urls = list(filter(lambda url:url is not None, urls))
        docs = AsyncMovieHtmlLoader(urls).load()
        docs_content = ""
        
        for doc in docs:
            doc_content = str(doc.metadata)
            docs_content += doc_content
            if len(docs_content) > upper_token:
                break
            
        template = """
Here's the html of the movie {query}: 
{docs_content}
"""
        
        response += template.format(
            docs_content = docs_content[:upper_token],
            query = query)
        
        movie_rating = movie_scores.get(movie.metadata["MovieId"])
        if isinstance(movie_rating,float):
            response += """
Here is the historical rating for movie {movie_name}, ranging from 0 to 5 points:
{movie_rating:.3f}
""".format(movie_name = movie.metadata['Title'],
           movie_rating = movie_rating)
        return response
        
    except Exception as e: 
        return response 
        
def func_get_movie_html(query,
          retriever,
          movie_scores,
          llm_kdb_summary:bool = False,
          llm_url_summary:bool = False,
          url_keys:list = [
                            # "movieId_url", 这个有反爬
                           "imdbId_url",
                           "tmdbId_url"],
          upper_token = 1000):
    """You can get the movie html information of one movie you want to watch using this tool.[!Important!] You should always give your rating after using this tool!! 

    Args:
        query (string): The name of a certain movie

    Returns:
        str: information about the movie online
    """
    response =""
    try:
        movie = retriever.invoke(query)[0]
        response += get_movie_html(query,movie,movie_scores,url_keys,upper_token)
        if llm_kdb_summary:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            response += get_llm_summary(llm,movie_name = query)
        if llm_url_summary:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            response += get_llm_vedio_summary(llm, movie_name = query, urls=[movie.metadata[k] for k in url_keys])
    
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,
                           content=response)
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR,
                           content=e)

def get_movie_url(query,
          retriever):
    try:
        movie = retriever.invoke(query)[0]
        urls =[]
        for k in movie.metadata.keys():
            if "url" in k: urls.append(movie.metadata[k])
        template ="""Here's the movie urls:{urls}"""
        
        response = template.format(urls = "\n".join(urls))
        return response
    
    except Exception as e:
        return ""
    
def create_get_movie_html_tool(
    retriever: BaseRetriever,
    movie_scores: dict,
    name: str,
    description: str,
    upper_token:int = 200,
    url_keys:list = [
                    # "movieId_url", 这个有反爬
                    "imdbId_url",
                    "tmdbId_url"],
    llm_kdb_summary:bool = False,
    llm_url_summary:bool = False,
) :
    """Create the html content of certain movie

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    
    return GraphServiceFactory.get(
        func_get_movie_html,
        name = name,
        description = description,
        retriever = retriever,
        movie_scores = movie_scores,
        llm_kdb_summary = llm_kdb_summary,
        llm_url_summary = llm_url_summary,
        upper_token = upper_token,
        url_keys = url_keys
    )
    
    
def create_movie_url_tool(
    retriever: BaseRetriever,
    name: str,
    description: str):
    """return the urls of certain movie"""
    
    return GraphServiceFactory.get(
        get_movie_url,
        name = name,
        description = description,
        retriever = retriever
    )
    

def _get_movie_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    filter_keys: list = [
        "interested_genres",  "watched_movie_ids"
    ],
    max_search:int = 5,
    interested_genres:list = [],
    watched_movie_ids:list = []
) -> str:
    """Search for information about relevant movies, you can use this to get the movies you care about the most.

    Args:
        query (str): seperated by comma, you should provide some keywords for the movie you want to watch (like genres, plots and directors...)

    Returns:
        str: the information about some relevant movies.
    """
    
    try:
        query_list = query.split(",")
        filtered_docs = []
        k = retriever.search_kwargs["k"]

        filter_pipeline = []
        filter_keys_set_map = {
            "interested_genres":generate_rank_movie_docs(interested_genres=interested_genres),
            "watched_movie_ids":generate_not_watched(watched_movie_ids=watched_movie_ids)
            }
        for filter_key,filter_function in filter_keys_set_map.items():
            if filter_key in filter_keys:
                filter_pipeline.append(
                    filter_keys_set_map[filter_key]
                )
                
        for query in query_list:
            docs = retriever.get_relevant_documents(query)
            filtered_docs.extend(docs)

        for filter_function in filter_pipeline:
            key_func = functools.cmp_to_key(filter_function)
            filtered_docs = list(
            sorted(filtered_docs,
                    key=key_func))
            
        if len(filtered_docs) > k:
            filtered_docs = filtered_docs[:k]
            
        output = document_separator.join(
            format_document(doc, document_prompt) for doc in filtered_docs
        )
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,
                           content=output)
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR,
                           content=e)
    

"""retriever filter function"""
def generate_rank_movie_docs(interested_genres:list =[]):
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        if len(interested_genres) == 0:
            return 0
        item1_weight = 1 if item1.metadata.get("Genres") in interested_genres\
              else 0
        item2_weight = 1 if item2.metadata.get("Genres") in interested_genres \
            else 0
        return item2_weight - item1_weight
    return compare

def generate_not_watched(watched_movie_ids:list = []):
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        if len(watched_movie_ids) == 0:
            return 0
        item1_weight = 1 if item1.metadata.get("MovieId") not in watched_movie_ids\
              else 0
        item2_weight = 1 if item2.metadata.get("MovieId") not in watched_movie_ids \
            else 0
        return item2_weight - item1_weight
    return compare

def create_movie_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    filter_keys: list = [
        "interested_genres",  "watched_movie_ids"
    ],
    max_search:int = 5,
    interested_genres:list = [],
    watched_movie_ids:list = []
) :
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    
    
   
    return GraphServiceFactory.get(
        _get_movie_relevant_documents,
        name=name,
        description=description,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        filter_keys=filter_keys,
        max_search=max_search,
        interested_genres=interested_genres,
        watched_movie_ids=watched_movie_ids
    )
