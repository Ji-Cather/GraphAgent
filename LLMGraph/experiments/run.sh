# generation
python main.py --task citeseer --config "large" --build 
python main.py --task movielens --config "large" --build 
python main.py --task tweets --config "large" --build

# shrinking diameter
python main.py --task tweets --config "llama_test_7000_p0.0025" --build
python main.py --task tweets --config "llama_test_7000_p0.0025_hubFalse" --build


# baselines:
# refer to LLMGraph\baselines

# ablation seed graph
python main.py --task citeseer --config "fast_gpt3.5_0.1k" --build
python main.py --task citeseer --config "fast_gpt3.5_subgraph" --build
python main.py --task citeseer --config "fast_gpt3.5_allgraph" --build

# evaluation of graph structure
python evaluate/article/main.py --task task --config config
python evaluate/movie/main.py --task task --config config
python evaluate/social/main.py --task task --config config