from __future__ import annotations

import re
from typing import Union



from .. import article_output_parser_registry
from ..base_parser import AgentOutputParser

    
@article_output_parser_registry.register("group_discuss")
class GroupDiscussParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        
        
        try:
            last_period_index = llm_output.rfind('.')
            if last_period_index != -1:
                llm_output = llm_output[:last_period_index + 1]
               
            return {"return_values":{"communication":llm_output}}
        except Exception as e:
            raise {"fail":True}
    

    
@article_output_parser_registry.register("choose_researcher")
class ChooseResearcherParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        try:
            output = llm_output.strip().split("\n")[0]
            return {"return_values":{"researcher":output}}
        except Exception as e:
            return {"fail":True}
         
@article_output_parser_registry.register("get_idea")
class GetIdeaParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        llm_output +="\n"

        try:
            regex = r"Thought\s*\d*\s*:(.*?)\nIdea.*?:(.*?)\nKeywords.*?:(.*?)\nAbstract.*?:(.*?)\nFinish.*?:(.*?)\n"
            output = re.search(regex,llm_output,re.DOTALL|re.IGNORECASE)
            finish = output.group(5)
            if "true" in finish.lower():
                finish = True
            else:
                finish = False
            return {"return_values":{"thought":output.group(1),
                                    "action":"writetopic",
                                    "idea":output.group(2),
                                    "keywords":output.group(3).split(","),
                                    "finish":finish,
                                    "abstract":output.group(4)}}

                
        except Exception as e:
            return {"fail":True}
        
   