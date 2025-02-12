
from functools import partial
from typing import Optional,Type, List
import functools

from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.retrievers import BaseRetriever
from agentscope.service import ServiceResponse, ServiceExecStatus
from .tool_warpper import GraphServiceFactory
from . import RetrieverInput
import time
import random
"""
compare docs functions
    follow: based on follow_ids and follow_me_ids
    big_name: based on big_name tag
    topic: based on topic type 
"""


def generate_compare_function_topic(interested_topic):
    # 在interested topic中的排前面
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("topic") in interested_topic else 0
        item2_weight = 1 if item2.metadata.get("topic") in interested_topic else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_big_name(big_name_list):
    # 在interested topic中的排前面
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("owner_user_index") in big_name_list else 0
        item2_weight = 1 if item2.metadata.get("owner_user_index") in big_name_list else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_follow(follow_ids:list):
    # 在interested topic中的排前面
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("owner_user_index") in follow_ids else 0
        item2_weight = 1 if item2.metadata.get("owner_user_index") in follow_ids else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_friend(friend_ids:list):
    # 在interested topic中的排前面
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("owner_user_index") in friend_ids else 0
        item2_weight = 1 if item2.metadata.get("owner_user_index") in friend_ids else 0
        return item2_weight - item1_weight
    return compare


def generate_judge_function_follow(follow_ids:list):
    def judge(item):
        # 根据interested_topic的长度来决定项目的排序权重
        item_weight = item.metadata.get("owner_user_index") in follow_ids
        return item_weight
    return judge

#########################################
def _get_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    big_name_list:list = [],
    filter_keys: list = [
        "follow",  "topic", "big_name", "friend"
    ],
    social_follow_map:dict = {
        "follow_ids": [],
        "friend_ids": []
    },
    interested_topics:List[str] = [],
    max_search:int = 5,
    hub_connect:bool = False # prefer connect to unseen hubs
) -> str:
    """You can search for anything you are interested on this platform.

    Args:
        query (str): The keywords you want to search for, seperated by comma; And the informations about the twitters you want to watch.

    Returns:
        str: the twitter information.

    """
    try:
        start_time = time.time()
        k = retriever.search_kwargs["k"]
        filter_pipeline = []
        filter_keys_set_map = {
            "follow":generate_compare_function_follow(social_follow_map.get("follow_ids",[])), 
            "big_name":generate_compare_function_big_name(big_name_list), 
            "topic":generate_compare_function_topic(interested_topics),
            "friend":generate_compare_function_friend(social_follow_map.get("friend_ids",[])), 
            }
        judge_follow_func = generate_judge_function_follow(social_follow_map.get("follow_ids",[]))
                                                           
        for filter_key,filter_function in filter_keys_set_map.items():
            if filter_key in filter_keys:
                filter_pipeline.append(
                    filter_keys_set_map[filter_key]
                )
        filtered_docs = []
        query_list = query.split(",")
        i = 0
        
        for query in query_list[:10]:
            # docs = retriever.get_relevant_documents(query)
            docs = retriever.get_relevant_documents(query)
            
            filtered_docs.extend(docs)
            if i > max_search:
                break
            i += 1

        for filter_function in filter_pipeline:
            key_func = functools.cmp_to_key(filter_function)
            filtered_docs = list(
            sorted(filtered_docs,
                    key=key_func))
        

       
        """for extentensive size hub"""
        if hub_connect:
            not_follow_docs = list(filter(judge_follow_func,filtered_docs))
            fixed_size = int(k*0.3)
            not_follow_size = int(k*0.3) 
            
            fixed_docs = filtered_docs[:fixed_size]

            if not_follow_size < len(not_follow_docs):
                not_follow_docs = random.sample(not_follow_docs,not_follow_size)
            else:
                not_follow_docs = []

            try:
                random_size = k - fixed_size - len(not_follow_docs)
                append_docs = random.sample(filtered_docs[fixed_size:],random_size)
            except:
                append_docs = []
            filtered_docs = not_follow_docs + append_docs + fixed_docs
        
        if len(filtered_docs)> k:
            filtered_docs = filtered_docs[:k]

        output = document_separator.join(
            format_document(doc, document_prompt) for doc in filtered_docs
        )


        print("get_relevant_documents tool_search", time.time()-start_time)
        return ServiceResponse(status = ServiceExecStatus.SUCCESS,
                           content = 
                           {"output":output,
                            })
    except Exception as e:
        return ServiceResponse(status = ServiceExecStatus.ERROR,
                           content = e)



def create_forum_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    big_name_list:list = [],
    filter_keys: list = [
        "follow",  "topic", "big_name", "friend"
    ],
    social_follow_map:dict = {
                    "follow_ids": [],
                    "friend_ids": []
                },
    interested_topics:List[str] = [],
    max_search:int = 5,
    hub_connect:bool = False # prefer connect to unseen hubs
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
        _get_relevant_documents,
        name=name,
        description=description,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        big_name_list = big_name_list,
        filter_keys = filter_keys,
        social_follow_map = social_follow_map,
        interested_topics = interested_topics,
        max_search = max_search,
        hub_connect = hub_connect  
    )