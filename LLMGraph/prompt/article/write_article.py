from LLMGraph.message import Message
from LLMGraph.prompt.article import article_prompt_default,article_prompt_registry

from pydantic import BaseModel 
from LLMGraph.prompt.base import BaseChatPromptTemplate
    
@article_prompt_registry.register("write_article")
class WriteArticlePromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             article_prompt_default.get("write_article",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "researcher",
                     "past_context",
                     "current_paper",
                     "searched_info",
                     "agent_scratchpad",
                     "write_memory",
                     "min_citations",
                     "max_citations",
                     "topics_available"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    