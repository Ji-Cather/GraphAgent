import pandas as pd
import os
root = "LLMGraph/tasks/general/data/Dianping"

for file in os.listdir(root):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(root, file))
        print(file.split(".")[0],df.columns)
        print(df.head(n=1))
