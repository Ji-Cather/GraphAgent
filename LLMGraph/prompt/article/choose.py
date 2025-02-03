from LLMGraph.message import Message
from LLMGraph.prompt.article import article_prompt_default,article_prompt_registry

from LLMGraph.prompt.base import BaseChatPromptTemplate
    
    
@article_prompt_registry.register("choose_reason")
class ChooseReasonPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             article_prompt_default.get("choose_reason",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "searched_papers",
                     "paper_content",
                     "citation_articles",
                     "citation_reasons",
                     "num_citation",
                     "agent_scratchpad",
                     "memory"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
