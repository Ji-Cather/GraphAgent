import numpy as np
import datetime
path = "LLMGraph/tasks/movielens/data/ml-processed_small_test/users.npy"
users = np.load(path,allow_pickle=True)
# small movie: 1091 user: 6040 (movielens-1M)
users = users[:100]
np.save(path, users)