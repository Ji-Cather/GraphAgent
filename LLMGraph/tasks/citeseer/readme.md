

For the configs in "LLMGraph/tasks/citeseer/configs"
## config name
```yaml
big: This configuration represents the largest networks we have explored in the macro-level alignment experiment.
small: Simulation demo. Run this setup first to get a better understanding of the simulation environment.

fast-gpt3.5-*: These configurations are used for the graph expansion experiment, based on simulations with GPT-3.5-based agents. The seed networks vary in size, indicated by the suffixes 100, 445, and 2840.
fast-gpt3.5-100: Seed network with 100 nodes.
fast-gpt3.5-445: Seed network with 445 nodes.
fast-gpt3.5-2840: Seed network with 2840 nodes.

user: The setup config file for control agent.
```

## config file
For author-paper simulation environment, we use the following config to setup the simulation environment. Here's the detailed explanation of hyperparameters:

```yaml
environment:
  env_type: article
  
  # the hyperparameters related to the Actor Agent
  agent: 
    llm: # the background LLM
      config_name: gpt-4o-mini
    type: article_agent # the agent type

    # article agent is equipped with two kinds of memories
    social_memory: 
      reflection: false
      summary_threshold: 10
      type: social_memory
    write_memory:
      type: rational_memory

    # General setup for every paper written by the actor agent
    article_write_configs:
      author_num: 5                  # Number of co-authors per paper
      citations: 10                  # Default citation number (ignored if use_graph_deg is true)
      communication_num: 2           # Number of discussion rounds among co-authors per simulation step
      max_refine_round: 1            # Maximum number of refinement rounds for each paper
      use_graph_deg: true            # Citation number is determined by the degree of the seed graph

    # Simulation control parameters
    max_paper_num: 100               # Maximum number of papers to control the simulation stop

    # Time-related configurations in the simulation environment
    time_configs:
      article_num_per_delta: 50      # Number of articles written per simulation step
      author_num_per_delta: 30       # Number of authors added per simulation step
      author_time_delta: 30          # Authors are added every 30 days
      cur_time: 2004-01-01           # Start time of the simulation
      end_time: 2011-01-01           # End time of the simulation (controls the simulation stop)
      round_time_delta: 5            # Time delta between two simulation steps (in days)



  # the hyperparameters related to the Item Agent
  managers: 
    article:
      article_dir: data/article
      article_meta_path: data/article_meta_info.pt
      author_path: data/author.pt
      experiment: 
      # The order of items presented to the actor agent is shuffled by default.
      # We have conducted other experiments adjusting the information revealed to actor agents,
      # which have shown an impact on the final network structure. These experiments will be presented in future work.
      - shuffle
      generated_article_dir: data/generated_article
      model_config_name: gpt-4o-mini
      
      # hyperparameters for recall retriever
      retriever_kwargs: 
        type: graph_vector_retriever
        search_kwargs:
          k: 20
          score_cite: true
        
      # hyperparameters for reranking 
      tool_kwargs:
        filter_keys:
        - big_name
        - topic
        - write_topic

      # core user rate
      control_profile:
        hub_rate: 0.1
        
# when set to true, the Control Agent setup the config instead 
use_agent_config: false 
```
