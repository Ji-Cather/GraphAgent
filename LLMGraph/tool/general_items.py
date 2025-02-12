from functools import partial
from typing import Optional, Type, List, Union, Callable
import os

from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.retrievers import BaseRetriever
from .tool_warpper import GraphServiceFactory
from . import RetrieverInput
from agentscope.service import ServiceResponse, ServiceExecStatus
from langchain_core.documents import Document
import random



def format_document(doc: Document, 
                    prompt: BasePromptTemplate[str]
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
    
    base_info ={
        **doc.metadata,
        "page_content":doc.page_content
    }

    return prompt.format(**base_info)

def load_rec_model(self, rec_model_path):
    import torch
    print('Loading Rec Model')
    rec_model = torch.load(rec_model_path, map_location="cpu")
    rec_model.eval()
    for name, param in rec_model.named_parameters():
        param.requires_grad = False
    print('Loding Rec model Done')
    return rec_model


def _get_general_relevent_items(
    query:str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    max_search:int = 5,
    rec_model:Union[Callable,str] = "default"
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
        for query in query_list[:max_search]:
            docs = retriever.get_relevant_documents(query)
            filtered_docs.extend(docs)
        random.shuffle(filtered_docs)
        if rec_model == "default":  
            filtered_docs = filtered_docs[:k]
        else:
            pass
            # TBD
            # for filtered_doc in filtered_docs:
            #     filtered_doc.metadata["score"] = rec_model(node_info, filtered_doc.page_content)
            # filtered_docs = sorted(filtered_docs, key=lambda x: x.metadata["score"], reverse=True)

        output = document_separator.join(
            format_document(doc, document_prompt) for doc in filtered_docs
        )
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,
                           content=output)
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR,
                           content=e)


def create_general_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    max_search:int = 5,
    filter_method: str = "default",
    rec_model_path: str = "LLMGraph/recs"
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
    if filter_method in ["SASrec", "GNN"]:
        pretrained_model_path = os.path.join(os.path.dirname(__file__), rec_model_path)
        rec_model = load_rec_model(pretrained_model_path)
    elif filter_method == "default":
        rec_model = filter_method 
    else:
        print("filter method not supported")
    return GraphServiceFactory.get(
        _get_general_relevent_items,
        name=name,
        description=description,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        max_search = max_search,
        rec_model = rec_model
    )
