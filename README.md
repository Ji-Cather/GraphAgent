# GraphAgent
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

### Build Network Demo
```cmd
    export PYTHONPATH=./
```

- To start building social network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python start_launchers.py
    python main.py --task tweets --config "small" --build # build from synthhetic tweet data
    
    # follow/action/friend networks
    python evaluate/social/main.py
    ```

- To start building movie rating network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python start_launchers.py
    python main.py --task movielens --config "small" --build # build from synthhetic tweet data
    
    # movie rating/user projection networks
    python evaluate/movie/main.py
    ```

- To start building citation network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python start_launchers.py
    python main.py --task movielens --config "small" --build # build from synthhetic tweet data
    
    # citation networks and etc.
    python evaluate/article/main.py
    ```


### An Illustration Vedio of Tweet Networks
visualization/social_network.mp4
<video width="640" height="480" controls>
  <source src="LLMGraph/experiments" type="video/mp4">
</video>

### Experiments
LLMGraph/experiments

### Build networks from user input only
Setting up a simulation environment from a prompt can be an efficient and straightforward way to get started, you can simply run:
```cmd
python main.py --user_input "I want to simulate authors interaction with papers. I want to generate a highly-clustered citation network with high average degree, with many well-known authors."  --build

python main.py --user_input "I want to simulate users interaction with movies. I want to generate a highly-clustered movie rating network with high average degree."  --build

python main.py --user_input "I want to simulate users interaction in tweet patform. I want to generate a highly-clustered online social networks  with high average degree."  --build

```