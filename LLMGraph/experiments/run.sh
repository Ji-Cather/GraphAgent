# large graph generation
python main.py --task citeseer --config "big" --build 
python main.py --task movielens --config "big" --build 
python main.py --task tweets --config "big" --build

# shrinking diameter
python main.py --task tweets --config "mid_hub" --build
python main.py --task tweets --config "mid_hubFalse" --build


# baselines:
# refer to LLMGraph\baselines
