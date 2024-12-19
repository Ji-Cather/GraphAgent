import pandas as pd
import matplotlib.pyplot as plt
label_df_map = {
    "w.o. ReRanking":"LLMGraph/tasks/tweets/configs/llama_test_7000_p0.0025_hubFalse/evaluate/friend_matrix.csv",
    "w. ReRanking":"LLMGraph/tasks/tweets/configs/llama_test_7000_p0.0025/evaluate/friend_matrix.csv"
}


# 创建一个图形和一个子图
fig, ax = plt.subplots()

for label, path in label_df_map.items():
    df = pd.read_csv(path,index_col=0)
    label_df_map[label] = df

min_size = max(label_df_map.values(), key=len).shape[0]
min_size = 28

for label, df in label_df_map.items():
    diameter = df["effective diameter"].to_list()[:min_size]
    ax.plot(diameter, label=label, linewidth=2)

# Adjust legend position
plt.legend(fontsize=24, loc='upper right', bbox_to_anchor=(1.15, 1))

# ax.set_xticks([])
ax.tick_params(axis='x', labelsize=20) 
ax.tick_params(axis='y', labelsize=20) 
ax.set_xlabel('Time',fontsize = 24)
ax.set_ylabel('$D_e$',fontsize = 24)
plt.grid(True)
plt.ylim(ymin=0)

plt.tight_layout(rect=[0, 0.2, 1, 1])  # 留出图例的空间
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1,
          fontsize=22)

plt.savefig("evaluate/visualize/for_paper/diameter_comparison.pdf")
