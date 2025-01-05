# large graphgeneration
python main.py --task citeseer --config "big" --build 
python main.py --task movielens --config "big" --build 
python main.py --task tweets --config "big" --build

# shrinking diameter
python main.py --task tweets --config "big" --build
python main.py --task tweets --config "big_hubFalse" --build


# baselines:
# refer to LLMGraph\baselines

# ablation seed graph
python main.py --task citeseer --config "fast_gpt3.5_0.1k" --build
python main.py --task citeseer --config "fast_gpt3.5_subgraph" --build
python main.py --task citeseer --config "fast_gpt3.5_allgraph" --build