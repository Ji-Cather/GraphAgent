from evaluate.article.build_graph import (
        build_author_citation_graph,
        build_country_citation_graph,
        build_relevance_array,
        build_group_relevance_array,
        build_citation_group_array,
        build_citation_graph,
        build_bibliographic_coupling_network,
        build_co_citation_graph,
        build_co_authorship_network
    )
from LLMGraph.utils.io import readinfo, writeinfo
import powerlaw
import numpy as np
import os
import matplotlib.pyplot as plt


def ks_critical_value(n, alpha):
    """
    计算单样本 K-S 检验的临界值

    参数：
    n (int)：样本的大小
    alpha (float)：显著性水平

    返回：
    critical_value (float)：临界值
    """
    # 计算临界值
    critical_value = np.sqrt(-0.5 * np.log(alpha) / n)
    return critical_value

def calculate_alpha(n, critical_value):
    alpha = np.exp(np.power(critical_value,2)*n/(-0.5))
    return alpha

llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o-mini",
        # "qwen2": "QWEN2-70B",
        # "llama8b":"LLAMA-3-8B",
        "vllm": "LLAMA-3-70B",
        "gt":"Seed"
    }

config_templates_map ={
    # "LLM-Agent":[
    #     ("llm_agent_2","search_shuffle_base_gpt3.5"),
    #     ("llm_agent_2","search_shuffle_base_gpt4-mini"),
    #     # ("llm_agent_2","search_shuffle_base_qwen2"),
    #     ("llm_agent_2","search_shuffle_base_vllm"),
    #     ("llm_agent_4","gt"),
    # ],
    # "LLM-Agent":[
    #     ("llm_agent_4","search_shuffle_base_gpt3.5"),
    #     ("llm_agent_4","search_shuffle_base_gpt4-mini"),
    #     # ("llm_agent_2","search_shuffle_base_qwen2"),
    #     ("llm_agent_4","search_shuffle_base_vllm"),
    #     ("llm_agent_1","gt"),
    # ],
    # "Cora":[
    #     # ("cora","fast_gpt3.5"),
    #     # ("cora","fast_vllm"),
    #     # ("cora","fast_gpt4-mini"),
    #     # ("cora","fast_qwen2"),
    #     ("cora","fast_gpt3.5"),
    #     ("cora","fast_gpt4-mini"),
    #     # ("cora","fast_llama8b"),
    #     ("cora","fast_vllm"),
    #     ("cora","gt"),
    # ],
    "Citeseer":[
        ("citeseer","fast_gpt3.5"),
        ("citeseer","fast_gpt4-mini"),
        # ("citeseer","fast_llama8b"),
        ("citeseer","fast_vllm"),
        ("citeseer","gt"),
        # ("citeseer","fast_gpt3.5_2"),
        # ("citeseer","fast_vllm_2"),
        # ("citeseer","fast_gpt4-mini_2"),
        # ("citeseer","fast_qwen2"),
    ],
}


task_names_map=list(config_templates_map.keys())

def find_peak(arr):
    n = len(arr)
    
    if n == 0:
        return -1  # 如果数组为空，返回-1
    
    # 如果只有一个元素，返回该元素的索引
    if n == 1:
        return 0
    
    # 检查边界情况
    if arr[0] > arr[1]:
        return 0
    if arr[n - 1] > arr[n - 2]:
        return n - 1

    # 查找内部的峰值
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            return i
    
    return -1  # 如果没有峰值，返回-1


def draw_power_law(graphs,graph_name):
    model_num = len(llm_name_map)
    legend_size = 20
    title_size = 26
    legend_size_lower = 24
    tick_size = 20
    # 绘制幂律分布的对数-对数图
    save_dir = "evaluate/visualize/for_paper"
    # fig, ax = plt.subplots(len(task_names_map), model_num, figsize=(model_num*4, 4.5*len(task_names_map)), 
    # sharey=True)
    fig, ax = plt.subplots(len(task_names_map), model_num, figsize=(model_num*4, 5*len(task_names_map)), 
    sharey=True)
    if len(task_names_map) ==1:
        ax = [ax]
    # colors = ['gray','black','r',"#293890"]
    # colors = ['gray','black','#d15f2a',"#2e668a"]
    # colors = ['gray','black','#d15f2a',"lightblue"]
    colors = ['gray','black','r',"b"]

    llm_names = list(llm_name_map.keys())
    llm_names_all = []
    for i in range(len(graphs)//model_num):
        llm_names_all.extend(llm_names)

    handles_all, labels_all = [], []

    for G, llm_name,idx in zip(graphs,llm_names_all, list(range(len(graphs)))):
        # degree_list = [G.degree(n) for n in G.nodes()]
        degree_list = [G.in_degree(n) for n in G.nodes()]
        idx_i = idx // model_num
        idx_j = idx % model_num
        # if idx_i == len(task_names_map)-1:
        #     xmin = 3
        # elif idx_i == 0:
        #     xmin = 4
        xmin = 1
        # xmin = None
        # xmin = 10
        results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                    # fit_method="KS",
                                    xmin=xmin,
                                    )
        
        # if idx_i == len(task_names_map)-1:
        #     xmax = 50
        # else:
        #     xmax = 3e2
        # results.power_law.xmax = xmax
        # results = powerlaw.Fit(list(degree_list), 
        #                         discrete=True,
        #                             # fit_method="KS",
        #                             xmin=xmin,
        #                             xmax=xmax # 为了plot
        #                             )
        
        
        # if idx_i == len(task_names_map)-1:
        #     results.power_law.xmax = 50
        # ax[idx_i][idx_j].set_xlim(left=0)
        # plot 
        peak = find_peak(degree_list)
        avg_degree = sum(degree_list)/ G.number_of_nodes()
        degree_counts = np.bincount(degree_list)
        degrees = np.arange(len(degree_counts))
        degrees = degrees[degree_counts > 0]
        degree_counts = degree_counts[degree_counts > 0]

        degree_counts = degree_counts / np.sum(degree_counts)
        # ax[idx_i][idx_j].loglog(degrees, 
        #                         degree_counts, 
        #                         color = colors[2],
        #                         marker='o',  
        #                         # linewidth=2, 
        #                         linestyle='None',  # 不绘制连接线
        #                         markersize=4,
        #                         label='Original Degree')


        
        results.plot_pdf(color=colors[1],
                         ax=ax[idx_i][idx_j],
                         marker='d',  
                         linestyle='-', 
                         linear_bins = True,
                         linewidth=2, 
                         markersize=3, 
                        #  original_data=True,
                         markerfacecolor=colors[1],  # Marker face color
                         markeredgecolor=colors[1],  # Marker edge 
                         label='Linearly-binned Degree',)
        results.plot_pdf(ax=ax[idx_i][idx_j],
                         color=colors[2], 
                         linestyle='None',
                         marker='o', 
                         linewidth=2, 
                         markersize=4, 
                         markerfacecolor=colors[2],  # Marker face color
                         markeredgecolor=colors[2],  # Marker edge color
                         label='Log-binned Degree')

        
        if idx_i ==0:
            handles_all, labels_all = ax[idx_i][idx_j].get_legend_handles_labels()

        # if idx_i == len(task_names_map)-1:
        #     ax[idx_i][idx_j].set_xlim(right=50)
        # results.plot_pdf(ax=ax[idx_i][idx_j],
        #                  color='b', 
        #                     marker='o', 
        #                     label='Degree',)
        
        # 拟合的幂律分布
        if idx_i ==0:
            results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                sigma_threshold=.1
                                    # fit_method="KS",
                                    # xmin=3,
                                    )
        else:
            results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                sigma_threshold=.1
                                # sigma_threshold=.2
                                    # fit_method="KS",
                                    # xmin=xmin,
                                    )
            
        
        D = results.power_law.D
        alpha = results.power_law.alpha
        xmin_c = results.power_law.xmin
        sigma = results.power_law.sigma
        kappa = results.power_law.Kappa
        # label=f"$\\alpha$ = {alpha:.2f}\n$D$ = {D:.2f}\npeak={degree_list[peak]:.2f}\navg_k={avg_degree:.2f}\n$k_{{min}}={xmin_c:.2f}$"
        # label=f"$\\alpha$ = {alpha:.2f}\n$D$ = {D:.2f}\n$k_{{min}}={xmin_c:.2f}$"
        # label=f"$\\alpha$ = {alpha:.2f}\n$D$ = {D:.3f}\n$\\bar k={avg_degree:.2f}$\n$Kappa = {sigma:.2f}"
        
        
        D_01thres = ks_critical_value(len(results.data),0.01)
        


        if D <= D_01thres:
            label=f"$\\alpha$ = {alpha:.2f}\n$D*$ = {D:.3f}\n$\\bar k={avg_degree:.2f}$"
        else:
            label=f"$\\alpha$ = {alpha:.2f}\n$D$ = {D:.3f}\n$\\bar k={avg_degree:.2f}$"

        results.power_law.plot_pdf(ax=ax[idx_i][idx_j],
                                #    color='#79cafb', 
                                color = colors[3],
                                    linestyle='--', 
                                    linewidth=2.5,
                                    label=label)
        handles, labels = ax[idx_i][idx_j].get_legend_handles_labels()
        ax[idx_i][idx_j].legend(loc='lower left', fontsize=legend_size,frameon=False,
                                handles=handles[-1:], labels=labels[-1:])
        ax[idx_i][idx_j].tick_params(axis='both', labelsize=tick_size)
        ax[idx_i][idx_j].tick_params(axis='y', labelsize=tick_size)

        if idx_i ==0:
            handles_all.append(handles[-1])
            labels_all.append("Power-Law Fit Line")
        if idx_j == model_num-1:
            # 创建第二个y轴，共享x轴
            ax2 = ax[idx_i][idx_j].twinx()

            # 设置第二个y轴不显示刻度线
            ax2.yaxis.set_ticks([])

            # 在右侧y轴中央添加文本
            text = list(task_names_map)[idx_i]
            ax2.set_ylabel(text, rotation=270, labelpad=50,fontsize=title_size)
            ax2.yaxis.set_label_coords(1.1, 0.5)
            ax2.tick_params(axis='both', labelsize=tick_size)
            pass

        if idx_i ==0:
            ax[idx_i][idx_j].set_title(llm_name_map[llm_name], fontsize=title_size)

    for i in range(len(task_names_map)):
        for j in range(model_num):
            ax[i][j].tick_params(axis='both', labelsize=tick_size)
    

    # 图形设置
    plt.xlabel(r'$k$', fontsize=18)
    #fig.text(0.5, 0.07, r'$k$', ha='center', fontsize=title_size)
    fig.text(0.5, 0.16, r'$k$', ha='center', fontsize=title_size)
    # ax[0][0].set_ylabel(r'Cumulative distributions of $k$, $P_{k}$', fontsize=16)
    # fig.text(0.04, 0.5, r'$P(k)$', va='center', rotation='vertical', fontsize=title_size)
    fig.text(0.01, 0.45, r'$P(k)$', va='center', rotation='vertical', fontsize=title_size)
    # fig.text(0.01, 0.45, r'$F(k)$', va='center', rotation='vertical', fontsize=title_size)
    # fig.grid(True)
    # plt.subplots_adjust(top=0.8, bottom=0.12, left=0.1, right=0.9, hspace=0,wspace=0)
    plt.subplots_adjust(top=0.8, bottom=0.3, left=0.1, right=0.9, hspace=0,wspace=0)
    
    fig.legend(loc='lower center',ncol=4, fontsize=legend_size_lower,handles=handles_all,labels=labels_all)
    # save_path = os.path.join(save_dir,f"{graph_name}_degree_xmin{xmin}.pdf")
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)
    save_path = os.path.join(save_dir,f"{graph_name}_degree_xmin{xmin}_all_2.pdf")
    plt.savefig(save_path)
    plt.clf()

def get_data(task,config):
    data_path = "LLMGraph/tasks/{task}/configs/{config}/data"
    article_meta_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                              "article_meta_info.pt"))
    author_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                       "author.pt"))
    return article_meta_info,author_info



if __name__ == "__main__":
    graphs = []

    for task_name in config_templates_map.keys():
        commands = config_templates_map[task_name]
        if task_name == "LLM-Agent":
            max_nodes = 1000
        else:
            # max_nodes = 10000
            max_nodes = 10000
        for command in commands:
            task_name,config = command
            article_meta_info,author_info = get_data(task_name,config)
            graph = build_citation_graph(article_meta_info)
            if graph.number_of_nodes() > max_nodes:
                graph = graph.subgraph(list(graph.nodes())[:max_nodes])
            print(graph.number_of_nodes(),graph.number_of_edges())
            graphs.append(graph)

    draw_power_law(graphs,"citation")