# micro evluation 
Metrics with reference to https://github.com/Genentech/bandwidth-graph-generation

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