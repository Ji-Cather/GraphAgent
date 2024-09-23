from LLMGraph.message import Message
from LLMGraph.prompt.article import cora_prompt_default,cora_prompt_registry

from LLMGraph.prompt.base import BaseChatPromptTemplate
    
    
@cora_prompt_registry.register("get_author")
class GetAuthorPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             cora_prompt_default.get("get_author",""))

        input_variables = kwargs.pop("input_variables",
                    ["expertises_list",
                     "author_num",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
