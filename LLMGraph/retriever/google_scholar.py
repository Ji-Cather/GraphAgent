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

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

class GoogleScholarAPIWrapper(BaseModel):

    top_k_results: int = 10
    hl: str = "en"
    lr: str = "lang_en"
    serp_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serp_api_key = get_from_dict_or_env(values, "serp_api_key", "SERP_API_KEY")
        values["SERP_API_KEY"] = serp_api_key

        try:
            from serpapi import GoogleScholarSearch

        except ImportError:
            raise ImportError(
                "google-search-results is not installed. "
                "Please install it with `pip install google-search-results"
                ">=2.4.2`"
            )
        GoogleScholarSearch.SERP_API_KEY = serp_api_key
        values["google_scholar_engine"] = GoogleScholarSearch

        return values

    def run(self, query: str) -> List[Document]:
        """Run query through GoogleSearchScholar and parse result"""
        total_results = []
        page = 0
        while page < max((self.top_k_results - 20), 1):
           
            results = (
                self.google_scholar_engine(  # type: ignore
                    {
                        "q": query,
                        "start": page,
                        "hl": self.hl,
                        "num": min(
                            self.top_k_results, 20
                        ),  # if top_k_result is less than 20.
                        "lr": self.lr,
                    }
                )
                .get_dict()
                .get("organic_results", [])
            )
            total_results.extend(results)
            if not results:  # No need to search for more pages if current page
                # has returned no results
                break
            page += 20
        if (
            self.top_k_results % 20 != 0 and page > 20 and total_results
        ):  # From the last page we would only need top_k_results%20 results
            # if k is not divisible by 20.
            results = (
                self.google_scholar_engine(  # type: ignore
                    {
                        "q": query,
                        "start": page,
                        "num": self.top_k_results % 20,
                        "hl": self.hl,
                        "lr": self.lr,
                    }
                )
                .get_dict()
                .get("organic_results", [])
            )
            total_results.extend(results)
        if not total_results:
            return "No good Google Scholar Result was found"
        
        docs =[]
        for result in total_results:
            meta_data ={
                "title": f"{result.get('title','')}",
                "cited": f"{result.get('inline_links',{}).get('cited_by',{}).get('total','Unknown')}",
                "authors": f"{','.join([author.get('name') for author in result.get('publication_info',{}).get('authors',[])])}",
            }
            page_content = f"{result.get('snippet','')}"
            
            if len(page_content) > 500:
                page_content = page_content[:500]+"..."
                
            doc = Document(
                page_content = page_content, metadata=meta_data
            )
            docs.append(doc)
        
        return docs

    
@retriever_registry.register("graph_google_scholar_retriever")
class GraphGoogleScholarRetriever(ArxivRetriever):
    
    api_wrapper: GoogleScholarAPIWrapper
    graph_vector_retriever: GraphVectorRetriever
    
    def _get_relevant_google_scholar_documents(
        self, query: str
        ):
        
        docs = self.api_wrapper.run(query=query)
        
            
        return docs
            
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ) :
        google_scholar_docs = self._get_relevant_google_scholar_documents(query)
        
        vector_docs = self.graph_vector_retriever._get_relevant_documents(query,
                                                                          run_manager=run_manager)
        
        return [*vector_docs,*google_scholar_docs]



    @classmethod
    def from_db(cls, 
                vectorstore:VectorStore,
                graph_vector_kargs:dict,
                **kwargs):
        tags = graph_vector_kargs.pop("tags", None) or []
        tags.extend(vectorstore._get_retriever_tags())
        graph_vector_retriever = GraphVectorRetriever(vectorstore = vectorstore,
                   **graph_vector_kargs,
                   tags = tags
                   )
        api_wrapper = GoogleScholarAPIWrapper()
        return cls(graph_vector_retriever = graph_vector_retriever,
                   api_wrapper = api_wrapper,
                   **kwargs)
        