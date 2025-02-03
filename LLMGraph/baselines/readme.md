# micro evluation 
Metrics with reference to https://github.com/Genentech/bandwidth-graph-generation

For the config setting used in paper (gpt-3.5-turbo): Use the configuration fast_gpt3.5_445.

For the LLaMA setting: Use the configuration fast_vllm_445_test.

*To avoid result randomness due to seed graph degree: we set use_graph_deg: False, citations: 2.

## train dataset
```cmd
    python LLMGraph/baselines/get_expansion_dataset.py
```
or use default: "LLMGraph/baselines/baseline_checkpoints/llmcitationciteseer.pkl"



## generate sampled gag results
```cmd
    python LLMGraph/baselines/get_gag_graphs.py
```

## evaluate
```cmd
    python LLMGraph/baselines/eval_pred_graphs.py
```


<!-- # macro evaluation
result checkpoints
"LLMGraph/tasks/citeseer/configs/fast_vllm"
"LLMGraph/tasks/movielens/configs/test_movie_up"
"LLMGraph/tasks/tweets/configs/llama_test_1e6" -->