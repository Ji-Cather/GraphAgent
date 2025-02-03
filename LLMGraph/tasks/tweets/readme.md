For the configs in "LLMGraph/tasks/tweets/configs"
## config name
```yaml
big: This configuration represents the largest networks we have explored in the macro-level alignment experiment.
small: Simulation demo. Run this setup first to get a better understanding of the simulation environment.

mid_hub: This configuration represents the medium-sized networks we have explored in the macro-level alignment experiment for shrinking diameter. (w. Reranking, hub_connect=True)
mid_hubFalse: This configuration represents the medium-sized networks we have explored in the macro-level alignment experiment for shrinking diameter. (w.o. Reranking, hub_connect=False)

user: The setup config file for control agent.
```


## config file
For user-tweet simulation environment, we use the following config to setup the simulation environment. Here's the detailed explanation of hyperparameters:

```yaml
environment:
  env_type: social


  time_configs:
    start_time: 2024-04-12 # The starting date for the simulation environment.
    cur_time: 2024-04-12 # The current date for the simulation environment.
    end_time: 2024-12-31 # The ending date for the simulation.
    social_time_delta: 1 # The time interval (in months) between the simulation step for online social activity.
    people_add_delta: 1 # The time interval (in months) at which people are added to the online platform.


  social_configs:
    max_people: 100 # The maximum number of people in the online platform.
    add_people_rate: 0.25 # The rate at which people are added to the online platform per simulation step.
    delete_people_rate: 0.25 # The rate at which people are added to the online platform per simulation step.

  # the hyperparameters related to the Item Agent
  managers:
    social:
      social_data_dir: data/systhetic_chop
      data_name: twitter_large_chop-20
      generated_data_dir: data/generated

      # hyperparameters for recall retriever
      retriever_kwargs:
        type: graph_vector_retriever
        search_kwargs:
          k: 20

      # hyperparameters for reranking 
      tool_kwargs:
        filter_keys: ["follow", "big_name", "topic"]
        hub_connect: True

      # core user rate
      control_profile:
        hub_rate: 0.2
      


  # the hyperparameters related to the Actor Agent
  agent:
    type: social_agent
    llm:
      config_name: gpt-4o-mini
    memory:
      type: action_memory

# when set to true, the Control Agent setup the config instead 
use_agent_config: false
```