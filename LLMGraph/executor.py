import logging


from .initialization import (load_environment,
                             prepare_task_config,
                             update_env_config)
from LLMGraph.utils.io import readinfo
from LLMGraph.agent.control_agent import ControlAgent
from agentscope.agents import DialogAgent, UserAgent
import os
import time
from agentscope.message import Msg

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


# 删掉load agent，因为environment中不止agent参与，不限制参与类型


class Executor():
    def __init__(self,
                 environment,
                 ex_idx:str):
        self.environment = environment
        self.ex_idx = ex_idx# 标识实验的index
        self.start_time = time.time()
        self.environment.initialize()

    # def get_to_dist_args()

    @classmethod
    def from_task(cls, 
                  args:dict):
        """Build an LLMGraph from a task name.
        The task name should correspond to a directory in `tasks` directory.
        Then this method will load the configuration from the yaml file in that directory.
        """
        # Prepare the config of the task
        task_config, config_path,task_path = prepare_task_config(args["config"],args["task"])
        launcher_file_path = args["launcher_save_path"]

        import time
        import os
        save_dir = task_config.pop("save_root_dir","")
        
        time_stamp = time.time()
        save_dir = os.path.join(task_path,
                                f"{save_dir}/{time_stamp}")

        env_config = task_config.pop('environment')
       
        env_config["task_path"] = task_path
        env_config["config_path"] = config_path
        
        env_config["launcher_args"] = readinfo(launcher_file_path)

        use_agent_config = task_config.pop("use_agent_config",False)
        if use_agent_config:
            env_type = env_config["env_type"]
            control_agent = ControlAgent(env_type,"default")
            user_agent = UserAgent()
            # instruction = "you need to provide the some instruction about the graph you want to generate"
            # print(instruction)
            # requirement = user_agent()
            kwargs = {
                "requirement": args["user_input"] + "\n",
                "graph_type": env_type
            }
            control_msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func="get_environment_config")
            agent_configs = control_agent.reply(control_msg)
            agent_configs = agent_configs.content
            env_config = update_env_config(env_config, agent_configs, env_type)

        environment = load_environment({**env_config})
        
        
        return cls(environment = environment,
                   ex_idx = time_stamp)


    

    
    def run(self):
        """Run the environment from scratch until it is done."""
                    
        
        while not self.environment.is_done():
            self.environment.step()
            self.save()
            
        self.save()
        return self.ex_idx

    
    def save(self):
        self.environment.save(self.start_time)
            

    def reset(self):
        self.environment.reset()
        
    def test(self):
        self.environment.test()
    
    def eval(self):
        self.environment.eval()