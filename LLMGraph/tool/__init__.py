
from langchain_core.pydantic_v1 import BaseModel, Field
class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")
    

from .docs import create_article_retriever_tool
from .online_db import create_article_online_retriever_tool
from .movie import create_get_movie_html_tool,create_movie_retriever_tool
from .forum import create_forum_retriever_tool
from .tool_warpper import GraphServiceFactory