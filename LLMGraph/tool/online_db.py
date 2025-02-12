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
from langchain_core.prompts import PromptTemplate


def format_document(doc: Document, 
                    ) -> str:
    
    prompt = PromptTemplate.from_template("""
Title: {Title}
Publish Time: {time}
Authors:{Authors}                                       
Content: {page_content}""")

    try:
        time = doc.metadata["Published"].strftime("%Y-%m-%d")
    except:
        time = "Unknown"

    base_info ={
        **doc.metadata,
        "time":time,
        "page_content":doc.page_content
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
    return prompt.format(**document_info)

def _get_article_online_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt:BasePromptTemplate,
    document_separator: str,
    max_search:int = 5,
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
        
        if len(filtered_docs)> k:
            filtered_docs = filtered_docs[:k]
        output = document_separator.join(
            format_document(doc) for doc in filtered_docs
        )
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,
                           content=output)
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR,
                           content=e)


def create_article_online_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    max_search:int = 5,
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
        _get_article_online_relevant_documents,
        name=name,
        description=description,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        max_search = max_search,
        research_topic = research_topic
    )
