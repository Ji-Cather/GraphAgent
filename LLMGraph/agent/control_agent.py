from . import agent_registry
from LLMGraph.agent.base_agent import BaseGraphAgent
from LLMGraph.prompt.control import control_prompt_registry
from LLMGraph.output_parser import control_output_parser_registry
from agentscope.message import Msg



@agent_registry.register("control_agent")
class ControlAgent(BaseGraphAgent):
   

    mode : str = "article" # 控制llm_chain 状态（reset_state中改）
   
    
    def __init__(self, 
                 graph_type:str,
                 llm_config_name="vllm",
                 **kwargs):
        
        init_mode = graph_type
        prompt = control_prompt_registry.build(init_mode)
        output_parser = control_output_parser_registry.build(init_mode)
        name = f"{graph_type}_agent"

        super().__init__(name,
                          prompt_template=prompt,
                         output_parser=output_parser,
                         model_config_name=llm_config_name,
                         **kwargs)
        
    
    def get_environment_config(self,
                               requirement:str = "generate a random citation network",
                               graph_type:str ="article"):
        
        self.reset_state(graph_type)
        prompt_inputs = {
            "requirement": requirement
        }
        response = self.step(prompt_inputs = prompt_inputs).content
        try:
            configs = response.get("return_values",{}).get("configs",{})
        except:
            configs = {}
        return Msg(self.name,
                   configs,
                   role="assistant")

    def reset_state(self,
                    mode = "forum_action"):
       
        if self.mode == mode : return
        self.mode = mode
        
        prompt = control_prompt_registry.build(mode)
        output_parser = control_output_parser_registry.build(mode)
        self.prompt_template = prompt
        self.output_parser = output_parser