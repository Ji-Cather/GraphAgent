For the configs in "LLMGraph/tasks/movielens/configs"
## config name
```yaml
big: This configuration represents the largest networks we have explored in the macro-level alignment experiment.
small: Simulation demo. Run this setup first to get a better understanding of the simulation environment.
user: The setup config file for control agent.
```

## config file
For user-movie simulation environment, we use the following config to setup the simulation environment. Here's the detailed explanation of hyperparameters:

```yaml
environment:
  env_type: movie

  # Time-related configurations in the simulation environment
  time_configs:
    start_time: 1997-01-01 # The starting date for the simulation environment. [All movies and users >= this date will be included]
    cur_time: 1997-12-01 # The current date for the simulation environment.
    end_time: 1999-09-20 # The ending date for the simulation.
    movie_time_delta: 4 # The time interval (in months) between the release of movies. 
    watcher_time_delta: 1 # The time interval (in months) at which watchers (or viewers) are added to the simulation.
    watcher_add: True # Indicates whether new watchers can be added during the simulation.
    watcher_num: -1 # Specifies the number of watchers. A value of -1 indicates that all available watchers should be included.

  movie_rate_configs:
    min_rate_all: 100


  # the hyperparameters related to the Item Agent
  managers:
    movie:
      link_movie_path: data/ml-processed_small_test/ml-25m-links.csv
      movie_data_dir: data/ml-processed_small_test
      ratings_data_name: ratings_top10.npy
      generated_data_dir: data/generated_data

      # core user rate
      control_profile:
        hub_rate: 0.0

      # hyperparameters for reranking 
      tool_kwargs:
        filter_keys: [
          "watched_movie_ids",
          "interested_genres"
        ]

      # hyperparameters for recall retriever
      retriever_kwargs:
        type: graph_vector_retriever
        search_kwargs:
          k: 10

      html_tool_kwargs:
        upper_token: 500
        url_keys:
          - imdbId_url
          - tmdbId_url
        llm_kdb_summary: False
        llm_url_summary: False

  # the hyperparameters related to the Actor Agent
  agent:
    type: movie_agent
    llm:
      config_name: gpt-4o-mini
      max_tokens: 1000
      temperature: 0.8
    memory:
      type: movie_memory

# when set to true, the Control Agent setup the config instead 
use_agent_config: false
```