from functools import partial
from typing import Optional, Type, List


from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.retrievers import BaseRetriever
from .tool_warpper import GraphServiceFactory
from . import RetrieverInput
from agentscope.service import ServiceResponse, ServiceExecStatus
from langchain_core.documents import Document
import functools
import random

class ArticleInfoInput(BaseModel):
    """Input to the retriever."""

    title: str = Field(description="title of your paper")
    keyword: str = Field(description="the keywords of your paper")
    abstract: str = Field(description="the abstract of your paper")
    citation: str = Field(description="the citations")


"""sort 精排"""
def generate_compare_function_big_name(big_name_list):
    # 在interested topic中的排前面
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("author_name") in big_name_list else 0
        item2_weight = 1 if item2.metadata.get("author_name") in big_name_list else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_topic(interested_topic):
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("topic") in interested_topic else 0
        item2_weight = 1 if item2.metadata.get("topic") in interested_topic else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_cite():
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        try:
            return item2[0].metadata["cited"] - item1[0].metadata["cited"]
        except:
            return -1

    return compare


def format_document(doc: Document, 
                    article_meta_data:dict,
                    author_data:dict,
                    prompt: BasePromptTemplate[str],
                    experiment:list = [], # default/shuffle/false cite
                    ) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. `page_content`:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.prompts import PromptTemplate

            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
    """
    choose_key = "cited"
    
    try:
        title = doc.metadata["title"]
        author_ids = article_meta_data[title]["author_ids"]
        author_ids = list(filter(lambda x:x in author_data.keys(), author_ids))

        best_author_idx = author_ids[0]
        for author_id in author_ids:
            author_info = author_data[author_id]
            if author_data[best_author_idx].get(choose_key,0) < author_info.get(choose_key,0):
                best_author_idx = author_id
        
        best_author_info ={
            "author_cited": author_data[best_author_idx].get("cited",0),
            "country": author_data[best_author_idx].get("country",0),
            "institution": author_data[best_author_idx].get("institution",0),
            "author_name": author_data[best_author_idx].get("name",0),
        }
    except:
        best_author_info ={
                    "author_cited": "Unknown",
                    "country":"",
                    "institution": "",
                    "author_name": "Unknown",
                }
    base_info ={
        **doc.metadata,
        "page_content":doc.page_content,
        **best_author_info
    }

    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        raise ValueError(
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
    document_info = {k: base_info[k] for k in prompt.input_variables}
    if "false_data" in experiment:
        document_info["cited"] = random.randint(0, 2000)
    if "no_cite" in experiment:
        document_info["cited"] = "Unknown"
    if "no_content" in experiment:
        document_info["page_content"] = "Unknown"
    if "no_paper_time" in experiment:
        document_info["time"] = "Unknown"
    if "no_author" in experiment:
        document_info["author_name"] = "Unknown"
        document_info["author_cited"] = "Unknown"
    if "no_country" in experiment:
        document_info["country"] = "Unknown"
        document_info["institution"] = "Unknown"
    if "no_topic" in experiment:
        document_info["topic"] = "Unknown"
    if "anonymous" in experiment:
        document_info["author_name"] = "Unknown"
        document_info["country"] = "Unknown"
        document_info["institution"] = "Unknown"
        document_info["author_cited"] = "Unknown"
    
    return prompt.format(**document_info)



# def _get_article_relevant_documents(
#     query: str,
#     retriever: BaseRetriever,
#     article_meta_data:dict,
#     author_data:dict,
#     document_prompt: BasePromptTemplate,
#     document_separator: str,
#     experiment:list = [], # default/shuffle/false cite
#     filter_keys: list = [
#         "topic", "big_name", "write_topic"
#     ],
#     max_search:int = 5,
#     big_name_list:list = [],
#     interested_topics:List[str] = [],
#     research_topic:str =""
# ) -> str:
#     """Search for relevant papers, so as to refine your paper. \
# These papers should be included in your paper's citations if you use them in your paper. 

#     Args:
#         query (str): keywords split by commas. The informations about the papers you want to cite, you can enter some keywords or other info.

#     Returns:
#         str: information about some searched papers.
#     """
#     try:
#         k = retriever.search_kwargs["k"]
#         filtered_docs = []
#         query_list = query.split(",")
#         query_list.append(research_topic)
#         for query in query_list[:max_search]:
#             # docs = retriever.get_relevant_documents(query)
#             docs = retriever.get_relevant_documents(query)
#             filtered_docs.extend(docs)
        
#         filter_pipeline = []
#         filter_keys_set_map = {
#             "big_name":generate_compare_function_big_name(big_name_list), 
#             "topic":generate_compare_function_topic(interested_topics),
#             "write_topic":generate_compare_function_topic([research_topic]),
#             # "cite":generate_compare_function_cite(),
#             }
#         for filter_key,filter_function in filter_keys_set_map.items():
#             if filter_key in filter_keys:
#                 filter_pipeline.append(
#                     filter_keys_set_map[filter_key]
#                 )
#         if "nofilter" in experiment:
#             filter_pipeline = []
#         for filter_function in filter_pipeline:
#             key_func = functools.cmp_to_key(filter_function)
#             filtered_docs = list(
#             sorted(filtered_docs,
#                     key=key_func))
            
#         if len(filtered_docs)> k:
#             filtered_docs = filtered_docs[:k]

#         if  "shuffle" in experiment:
#             random.shuffle(filtered_docs)
            
#         output = document_separator.join(
#             format_document(doc, article_meta_data, author_data, document_prompt,
#                             experiment = experiment) for doc in filtered_docs
#         )
#         return ServiceResponse(status=ServiceExecStatus.SUCCESS,
#                            content=output)
#     except Exception as e:
#         return ServiceResponse(status=ServiceExecStatus.ERROR,
#                            content=e)

def _get_article_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    article_meta_data:dict,
    author_data:dict,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    experiment:list = [], # default/shuffle/false cite
    filter_keys: list = [
        "topic", "big_name", "write_topic"
    ],
    max_search:int = 5,
    big_name_list:list = [],
    interested_topics:List[str] = [],
    research_topic:str =""
) -> str:
    """Search for relevant papers, so as to refine your paper. \
These papers should be included in your paper's citations if you use them in your paper. 

    Args:
        query (str): keywords split by commas. The informations about the papers you want to cite, you can enter some keywords or other info.

    Returns:
        str: information about some searched papers.
    """
    try:
        k = retriever.search_kwargs["k"]
        filtered_docs = []
        query_list = query.split(",")
        query_list.append(research_topic)
        for query in query_list[:max_search]:
            # docs = retriever.get_relevant_documents(query)
            docs = retriever.get_relevant_documents(query)
            filtered_docs.extend(docs)
        try:
            filter_pipeline = []
            filter_keys_set_map = {
                "big_name":generate_compare_function_big_name(big_name_list), 
                "topic":generate_compare_function_topic(interested_topics),
                "write_topic":generate_compare_function_topic([research_topic]),
                "cite":generate_compare_function_cite(),
                }
            for filter_key,filter_function in filter_keys_set_map.items():
                if filter_key in filter_keys:
                    filter_pipeline.append(
                        filter_keys_set_map[filter_key]
                    )

            for filter_function in filter_pipeline:
                key_func = functools.cmp_to_key(filter_function)
                filtered_docs = list(
                sorted(filtered_docs,
                        key=key_func))
        except:
            pass
            
        if len(filtered_docs)> k:
            filtered_docs = filtered_docs[:k]

        if  "shuffle" in experiment:
            random.shuffle(filtered_docs)
            
        output = document_separator.join(
            format_document(doc, article_meta_data, author_data, document_prompt,
                            experiment = experiment) for doc in filtered_docs
        )
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,
                           content=output)
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR,
                           content=e)


def create_article_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    article_meta_data:dict,
    author_data:dict,
    *,
    experiment:list = [], # default/shuffle/false cite
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    filter_keys: list = [
        "topic", "big_name", "write_topic"
    ],
    max_search:int = 5,
    big_name_list:list = [],
    interested_topics:List[str] = [],
    research_topic:str =""
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
        _get_article_relevant_documents,
        name=name,
        description=description,
        retriever=retriever,
        article_meta_data=article_meta_data,
        author_data=author_data,
        document_prompt=document_prompt,
        document_separator=document_separator,
        experiment = experiment,
        filter_keys = filter_keys,
        max_search = max_search,
        big_name_list = big_name_list,
        interested_topics = interested_topics,
        research_topic = research_topic
    )
