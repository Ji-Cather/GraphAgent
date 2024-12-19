import matplotlib.pyplot as plt

import pandas as pd
import os
import shutil
import json
import yaml
import time
import openai
import matplotlib.dates as mdates
from LLMGraph.utils.io import writeinfo,readinfo





configs =[
    # "search_shuffle_base_nosocial_gpt3.5",
    # "search_shuffle_base_gpt3.5_ver1_longtime",
    # "search_shuffle_base_nosocial_vllm",
    # "search_shuffle_base_vllm",
    # "search_shuffle_base_nosocial_qwen2",
    "search_shuffle_base_qwen2",
    # "search_shuffle_base_nosocial_gpt4-mini",
    # "search_shuffle_base_gpt4-mini",
]
labels =[
    # "without social network",
    "with social network",
]
task ="llm_agent_2"
article_meta_paths =[
    "LLMGraph/tasks/{task}/configs/{config}/data/article_meta_info.pt".format(
        task=task,
        config=config
    )
    for config in configs
]

# 绘制累计增长分布图
num_count = {}
plt.figure(figsize=(10, 6))
for idx, article_meta_path in enumerate(article_meta_paths):
    data = readinfo(article_meta_path)
    data ={
        k: {"time": v["time"]}
        for k,v in data.items()
    }
    num_count[configs[idx]] = data
    writeinfo(f"num_count.json",num_count)

    # 将数据转换为DataFrame
    df = pd.DataFrame(data).T  # 转置DataFrame，使得title作为行索引
    
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m")

    # 计算每个时间点的累计数量
    df_filtered = df[df['time'] > '2023-04-01']
    df_filtered = df_filtered[df_filtered['time'] < '2024-12-01']

    # 计算每个时间点的累计数量
    df_cumulative = df_filtered.groupby('time').size().cumsum().reset_index(name='cumulative_count')
        
    plt.plot(df_cumulative['time'], df_cumulative['cumulative_count'], marker='o', linestyle='-',
             label=labels[idx%2])
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
# plt.xlabel('Time')
plt.ylabel('Cumulative Article Count')
plt.title('Cumulative Growth of Articles Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"evaluate/experiment/llm_agent/article_number.pdf")
