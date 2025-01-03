[# GraphAgent
LLM-based Agent Framework for Large-Scale Dynamic Graph Generation.


## GraphAgent Framework
Before we begin, please set your api key in "LLMGraph\llms\default_model_configs.json", and format it like:
```json
\[{
        "model_type": "openai_chat",
        "config_name": "gpt-3.5-turbo-0125",
        "model_name": "gpt-3.5-turbo-0125",
        "api_key": "sk-*",
        "generate_args": {
            "max_tokens": 2000,
            "temperature": 0.8
        }
}\]
```

create a virtual environment for LLMGraph
```cmd
    conda create --name LLMGraph python=3.9
    conda activate LLMGraph
```

pip install agentscope\[distributed\] v0.0.4 from https://github.com/modelscope/agentscope/
```cmd
    git clone https://github.com/modelscope/agentscope/
    git reset --hard 1c993f9
    # From source
    pip install -e .[distribute]
```

Then create the experiment, and install the required packages:
    ```
    pip install -i "requirements.txt"
    ```

### build social network

- To start building social network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    export PYTHONPATH=./
    python start_launchers.py
    python main.py --task tweets --config "small" --build # build from synthhetic tweet data
    
    # evaluation social networks
    python evaluate/social/main.py
    ```


### an illustration visualization of social network generation
visualization/social_network.mp4

### experiments
LLMGraph/experiments

<video width="640" height="480" controls>
  <source src="LLMGraph/experiments" type="video/mp4">
</video>](README.md)