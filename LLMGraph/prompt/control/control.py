from LLMGraph.message import Message
from LLMGraph.prompt.control import control_prompt_default, control_prompt_registry

from LLMGraph.prompt.base import BaseChatPromptTemplate
    
    
@control_prompt_registry.register("article")
class ArticleControlPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             control_prompt_default.get(
                                 "article_control_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["requirement",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
        
@control_prompt_registry.register("movie")
class MovieControlPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             control_prompt_default.get(
                                 "movie_control_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["requirement",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    

    
@control_prompt_registry.register("social")
class SocialControlPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             control_prompt_default.get(
                                 "social_control_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["requirement",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
