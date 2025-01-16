import numpy as np
import torch
path = "LLMGraph/tasks/citeseer/data/article_meta_info.pt"
data =torch.load(path)
print(len(data))