import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# 数据准备
data = {
    'content': "Paper Content",
    'cite': "Paper Citation",
    'paper_time': "Paper Timeliness",
    'topic': "Paper Topic",
    'author': 'Author Name',
    'author_cite': "Author Citation",
    'country': "Author Country",
    # 'social': "Social",
}
index_map ={
    'modularity': "Modularity",
    'assortativity': "Assortativity",
    'homophily': "Homophily",
    'Transitivity': "Transitivity",
    'Average_Clustering': "Average Clustering",
    'effective diameter': "Effective Diameter",
}



llm = "vllm"
# llm = "gpt3.5"
df_path = f"evaluate/experiment/llm_agent/{llm}/impact_indicator_summary_article_citation.csv"
df = pd.read_csv(df_path,index_col=0)
df.index = [index_map[x] for x in df.index]
df.columns = [data.get(x, x) for x in df.columns]
df = df[list(data.values())]
# 绘图
features = df.index
x_labels = df.columns

fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 20), sharex=True)
fig.subplots_adjust(hspace=0.5)

for i, feature in enumerate(features):
    df.loc[feature].plot(kind='bar', ax=axes[i], color='skyblue')
    axes[i].set_title(f"{feature}")
    axes[i].set_ylabel("Causal Effect Coefficient")
    axes[i].set_ylim(-1, 1)  # 假设所有值在这个范围内，根据需要调整
    axes[i].axhline(0, color='black',linewidth=0.5)
    axes[i].set_xticks(range(len(x_labels)))
    axes[i].set_xticklabels(x_labels, rotation=45, ha='right')

# 设置总的x轴标签
# plt.xlabel("Factors")
plt.tight_layout()
plt.savefig(f"evaluate/experiment/llm_agent/{llm}/impact.pdf")
