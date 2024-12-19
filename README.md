# GraphAgent
LLM-based Agent Framework for Large-Scale Dynamic Graph Generation.


## GraphAgent Framework
Before we begin, please set your api key in "LLMGraph\llms\default_model_configs.json", and format it like:
```json
[{
        "model_type": "openai_chat",
        "config_name": "gpt-3.5-turbo-0125",
        "model_name": "gpt-3.5-turbo-0125",
        "api_key": "sk-*",
        "generate_args": {
            "max_tokens": 2000,
            "temperature": 0.8
        }
}]
```

Install agentscope v0.0.4 from https://github.com/modelscope/agentscope/


Then create the experiment, and install the required packages:
    ```
    pip install -i "requirements.txt"
    ```

### build citation network

- To start building citation network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python main.py --task citeseer --config "large" --build # build from citeseer
    ```

### build film review network

- To start building film review network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python main.py --task movielens --config "large" --build # build from movielens 
    ```
### build social network

- To start building social network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python main.py --task tweets --config "large" --build # build from synthhetic tweet data
    ```

### build from input
python main.py --user_input "I want to build a citation network"

### an illustration visualization of social network generation
visualization/social_network.mp4

### experiments
LLMGraph/experiments

<video width="640" height="480" controls>
  <source src="LLMGraph/experiments" type="video/mp4">
</video>