import powerlaw
import matplotlib.pyplot as plt
from LLMGraph.utils.io import readinfo
import numpy as np

json_paths = [
    "LLMGraph/tasks/citeseer/configs/10000_full_vllm/evaluate/degree/article_citation.json",
    "LLMGraph/tasks/citeseer/configs/10000_full_vllm/evaluate/degree/co_citation.json",
    "LLMGraph/tasks/citeseer/configs/10000_full_vllm/evaluate/degree/author_citation.json",
    "LLMGraph/tasks/tweets/configs/llama_test_1e6/evaluate/20240421/degree/action.json",
    "LLMGraph/tasks/tweets/configs/llama_test_1e6/evaluate/20240421/degree/follow.json",
    "LLMGraph/tasks/tweets/configs/llama_test_1e6/evaluate/20240421/degree/friend.json"
]

graph_names = ["Citation Network",
               "Co-Citation Network",
               "Author-Citation Network",
               "Action Network",
               "Follow Network",
               "Friend Network"
               ]

fig, axs = plt.subplots(3, len(graph_names)//3, figsize=(5*len(graph_names)//3,5*3), 
    sharey=False)


idy = 0
colors = ["black","red","blue"]
legend_size = 20
title_size = 22
legend_size_lower = 18
tick_size = 16
for graph_name, json_path in zip(
    graph_names,
    json_paths
):
    
    degree_list = readinfo(json_path)
    
    results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                    fit_method="KS",
                                    )
    degree_counts = np.bincount(degree_list)
    degrees = np.arange(len(degree_counts))
    degrees = degrees[degree_counts > 0]
    degree_counts = degree_counts[degree_counts > 0]
    ax = axs[idy%3][idy//3]
    results.plot_pdf(ax=ax,
                    color = colors[0],
                    marker='o',  
                    markersize=4,
                    linestyle='None',  # 不绘制连接线
                    markeredgecolor='blue', 
                    markerfacecolor='none',
                    label='Log-binned Degree')
    results.plot_pdf(color=colors[2],
                         ax=ax,
                         marker='d',  
                         linestyle='-', 
                         linear_bins = True,
                         linewidth=2, 
                         markersize=3, 
                         label='Linearly-binned Degree',)
    
    results.power_law.plot_pdf(ax=ax,
                                #    color='#79cafb', 
                                color = colors[1],
                                linestyle='-', 
                                linewidth=3,
                                label='Power Law Fit')
    alpha = results.power_law.alpha
    xmin = results.power_law.xmin
    sigma = results.power_law.sigma
    D = results.power_law.D
    # 在图表中添加文本信息
    textstr = '\n'.join((
        r'$\alpha=%.2f$' % (alpha,),
        r'$D_k=%.2f$' % (D,),
        r'$\mathrm{k_{min}}=%.0f$' % (int(xmin)),
        ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.9, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=props)
    ax.set_title(graph_name, fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    idy += 1


fig.text(0.5, 0.08, r'Degree $k$', ha='center', fontsize=legend_size)
fig.text(0.01, 0.5,r'$P_{k}$', fontsize=legend_size)  # 单独设

plt.subplots_adjust(bottom=0.13)
handles_all, labels_all = axs[0][-1].get_legend_handles_labels()
fig.legend(loc='lower center',ncol=2, fontsize=legend_size_lower,handles=handles_all,labels=labels_all)

plt.savefig("evaluate/visualize/for_paper/degree_pl_all.pdf")