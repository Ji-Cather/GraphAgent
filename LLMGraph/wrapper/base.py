import time
from agentscope.agents import AgentBase
from agentscope.agents.rpc_agent import RpcAgentServerLauncher
from agentscope.message import Msg, PlaceholderMessage

from openai import RateLimitError,AuthenticationError
from LLMGraph.message import Message
from typing import List,Dict
import json
from abc import abstractmethod

FUNCTION_RESULT_TITLE_PROMPT = """Execution Results:
"""

FUNCTION_RESULT_PROMPT = """{index}. {function_name}:
    [EXECUTE STATUS]: {status}
    [EXECUTE RESULT]: {result}
"""


class BaseAgentWrapper(AgentBase):
    """A demo agent to gather value"""

    def __init__(self, 
                 name: str, 
                 agent: AgentBase,
                 manager_agent: AgentBase,
                 max_retrys:int = 3,
                 max_tool_iters:int = 2,
                 to_dist = False) -> None:
        super().__init__(name, to_dist = to_dist)

        # Prepare system prompt
        
        self.agent = agent
        self.manager_agent = manager_agent
        
        self.max_retrys = max_retrys
        self.max_tool_iters = max_tool_iters
        

        
        

    def reply(self, message:Msg=None) -> dict:
        """call agent(function or not)

        Args:
            _ (dict, optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """
        
        return Msg(
            name=self.name,
            role="assistant",
            content={},
        )
    
    def call_agent_get_prompt(self,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        msg = Msg("user",
                content="get_prompt",
                role="assistant",
                kwargs=kwargs,
                func=func_name
                )
        return self.agent(msg)
    
    def call_agent_func(self,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name
                )
        return_msg = self.agent(msg)
        # if isinstance(return_msg, PlaceholderMessage):
        #     return_msg.update_value()
        return return_msg

    def call_agent_reply_prompt(self,
                        prompt) -> Msg:
        
        prompt_msg = Msg(self.agent.name,
                    content="reply_prompt",
                    role="assistant",
                    prompt = prompt
                    )
        msg_finish = self.agent(prompt_msg)
        return msg_finish
    

    def call_manager_func(self,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        """
        call manager function

        Args:
            tool_name (str): _description_
            func_name (str): _description_
            kwargs (dict, optional): _description_. Defaults to {}.

        Returns:
            Msg: _description_
        """
        
        msg = Msg("user",
                content="call_function",
                role="user",
                kwargs=kwargs,
                func=func_name
                )

        return self.manager_agent(msg)



    def step(self,
             agent_msgs: List[Msg]= [],
             use_tools:bool = False,
             return_tool_exec_only: bool = False,
             return_intermediate_steps :bool = False,
             ) -> Msg:
        if not isinstance(agent_msgs,list):
            agent_msgs = [agent_msgs]
        intermediate_steps = []
        steps_idx = 0
        
        prompt = []
        if use_tools:
            res_tool_msg = self.call_manager_func(
                "get_prompt_tool_msgs")
            

        while(steps_idx < self.max_retrys):
            
            memory_msgs = self.get_agent_memory_msgs().content
            if use_tools and steps_idx < self.max_tool_iters:
                if return_tool_exec_only:
                    prompt = [*agent_msgs, 
                            #   *memory_msgs,
                              *res_tool_msg.content]
                else:
                    prompt = [*agent_msgs, 
                              *memory_msgs,
                              *res_tool_msg.content]
            else:
                prompt = [*agent_msgs,
                          *memory_msgs]
            
            steps_idx +=1
            # Step 1: Thought
            # Generate LLM response
            # update_memory 应该在这里load agent_msgs
           
            msg_finish = self.call_agent_reply_prompt(prompt)
            msg_finish.content # 强制阻塞

            if msg_finish.get("fail",False):
                print(msg_finish.content)
                continue

            if msg_finish.get("finish",False):
                self.agent.speak(f" ITER {steps_idx}, STEP: FINISH".center(70, "#"))
                if return_intermediate_steps:
                    msg_finish.update({"intermediate_steps":intermediate_steps})
                # Skip the next steps if no need to call tools
                self.call_agent_func("clear_short_memory").content
                return msg_finish
           
            # Step 2: Action
            self.agent.speak(f" ITER {steps_idx}, STEP: ACTION ".center(70, "#"))
            try:
                execute_results = self.call_manager_func(msg_finish.func,
                                                        kwargs={
                                                        "function_call_msg":msg_finish.content}).content
            except Exception as e:
                execute_results = []
                print(msg_finish, e)
            assert isinstance(execute_results,list)
            intermediate_steps.extend(execute_results)

            # Prepare prompt for execution results
            execute_results_prompt = "\n".join(
                [
                    FUNCTION_RESULT_PROMPT.format_map(res_one[1])
                    for res_one in execute_results
                ],
            )
            # Add title
            execute_results_prompt = (
                FUNCTION_RESULT_TITLE_PROMPT + execute_results_prompt
            )

            # Note: Observing the execution results and generate response are
            # finished in the next loop. We just put the execution results
            # into memory, and wait for the next loop to generate response.

            # Record execution results into memory as a message from the system
            msg_res = Msg(
                name = self.agent.name,
                content=execute_results_prompt,
                role="assistant",
            )
            self.agent.speak(msg_res)
            self.agent.observe([msg_finish,msg_res])
            
            if return_tool_exec_only and \
                steps_idx == self.max_tool_iters: # 如果仅仅进行
                if return_intermediate_steps:
                    msg_finish.update({"intermediate_steps":intermediate_steps})
                self.call_agent_func("clear_short_memory").content
                return msg_finish
        
        msg_finish = Msg(
            "system",
            "The agent has reached the maximum iterations.",
            role="system",
        )

        if return_intermediate_steps:
            msg_finish.update({"intermediate_steps":intermediate_steps})

        self.call_agent_func("clear_short_memory").content
        return msg_finish
    
    @abstractmethod
    def get_agent_memory_msgs(self) -> List[Msg]:
        """get agent memory messages"""
        pass

    def observe(self, x) -> None:
        self.agent.observe(x)