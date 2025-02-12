
# from LLMGraph.prompt.chat_prompt.prompt_value import PromptValue
from typing import Any,List
from pydantic import BaseModel
from abc import abstractmethod

from agentscope.message import Msg

# Set up a prompt template

class BaseChatPromptTemplate(BaseModel):
    # The template to use
    template: str
    input_variables: list = []
    

    
    def format_messages(self, **kwargs) -> List[Msg]:
        # Format them in a particular way
        if "agent_scratchpad" not in kwargs.keys():
            kwargs["agent_scratchpad"] = ""
            
        formatted = self.template.format(**kwargs)

        return [Msg(name = "User",
                    content=formatted,
                    role ="user",
                    )]