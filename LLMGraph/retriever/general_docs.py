from LLMGraph.retriever import retriever_registry
from langchain_core.vectorstores import VectorStoreRetriever
from typing import (
    Dict,
    Callable)
from langchain_core.pydantic_v1 import root_validator
from functools import cmp_to_key




@retriever_registry.register("graph_general_vector_retriever")
class GraphGeneralVectorRetriever(VectorStoreRetriever):
    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        search_type = values["search_type"]
        if search_type not in cls.allowed_search_types:
            raise ValueError(
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
        if search_type == "similarity_score_threshold":
            score_threshold = values["search_kwargs"].get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                raise ValueError(
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        
        if values["search_kwargs"].get("score_cite") is not None:
            score_cite =  values["search_kwargs"].get("score_cite")
            if score_cite and search_type == "mmr":
                raise NotImplementedError("'score_cite == True' is not supported for mmr searching")
        
        return values
    
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ) :
        
        if "similarity" in self.search_type:
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc,_ in docs_and_similarities]
            
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


    @classmethod
    def from_db(cls, 
                vectorstore,
                 **kwargs):
        tags = kwargs.pop("tags", None) or []
        tags.extend(vectorstore._get_retriever_tags())
        return cls(vectorstore = vectorstore,
                   **kwargs,
                   tags = tags
                   )
        
