from LLMGraph.retriever import retriever_registry
from langchain_community.retrievers import ArxivRetriever
from langchain_core.vectorstores import VectorStore
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar)
from langchain_core.pydantic_v1 import Field
from langchain_core.pydantic_v1 import root_validator
from functools import cmp_to_key

from LLMGraph.retriever.docs import GraphVectorRetriever
    
@retriever_registry.register("graph_arxiv_retriever")
class GraphArxivRetriever(ArxivRetriever):
    search_kwargs: dict = Field(default_factory=dict)
    def _get_relevant_arxiv_documents(
        self, query: str, *, run_manager
        ):
        
        if self.get_full_documents:
            docs = self.load(query=query)
        else:
            docs = self.get_summaries_as_docs(query)
            
        for doc in docs:
            

            doc.page_content = doc.page_content.replace("\n"," ")
            if len(doc.page_content) >500:
                doc.page_content = doc.page_content[:500] +"..."
        return docs
            
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ) :
        arxiv_docs = self._get_relevant_arxiv_documents(query,
                                                        run_manager=run_manager)
        
        
        return arxiv_docs

    @classmethod
    def from_db(cls, **kwargs):

        return cls(**kwargs)
        