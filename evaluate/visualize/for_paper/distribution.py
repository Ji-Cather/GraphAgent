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
from matplotlib import font_manager as fm 
import seaborn as sns
import pandas as pd


task_names_map={
    "cora":"Cora",
    "citeseer":"Citeseer",
    "llm_agent_2":"LLM-Agent"
}

llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "vllm": "LLAMA-3-70B",
        "gpt4-mini": "GPT-4o-mini",
        "qwen2": "QWEN2-70B"
    }

config_templates_map ={
    "cora":["fast_{llm}"],
    "citeseer":["fast_{llm}"],
    "llm_agent_2":["search_shuffle_base_{llm}"]
}


    

def draw_power_law(graphs,graph_name):
    
    # 绘制幂律分布的对数-对数图
    save_dir = "evaluate/visualize/for_paper"
    fig, ax = plt.subplots(2, 4, figsize=(16, 4*len(task_names_map)), sharey=False)
    
    llm_names = list(llm_name_map.keys())
    llm_names_all = []
    for i in range(len(graphs)//4):
        llm_names_all.extend(llm_names)

    for G, llm_name,idx in zip(graphs,llm_names_all, list(range(len(graphs)))):
        degree_list = [G.degree(n) for n in G.nodes()]
        idx_i = idx // 4
        idx_j = idx % 4
        # if idx_i == len(task_names_map)-1:
        #     xmin = 3
        # else:
        #     xmin = None
        # xmin = None
        xmin = 3
        results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                    # fit_method="KS",
                                    xmin=xmin,
                                    )
        alpha = results.power_law.alpha
        xmin = results.power_law.xmin
        sigma = results.power_law.sigma

        if idx_i == len(task_names_map)-1:
            xmax = 50
        else:
            xmax = results.power_law.xmax
        results.power_law.xmax = xmax
        results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                    # fit_method="KS",
                                    xmin=xmin,
                                    xmax=xmax # 为了plot
                                    )
        
        kwargs = {
            "color":'b', 
            "linestyle":'--', 
            "linewidth":2, 
            "label":'Data'
        }
        
        # results.plot_pdf(color='b', 
        #                  linestyle='-', linewidth=2, label='Data')
        results.plot_pdf(ax=ax[idx_i][idx_j],
                         color='b', 
                            marker='o', 
                            label='Data',)
        # results.plot_pdf(color='g',
        #                  marker='d',  linear_bins = True,
        #                  label='Linearly-binned Data',)
        # 拟合的幂律分布
        D = results.power_law.D
        results.power_law.plot_pdf(ax=ax[idx_i][idx_j],
                                   color='r', 
                                    linestyle='--', 
                                    linewidth=2,
                                    label=f"$\\alpha$ = {alpha:.2f}") 
        
        if idx_j ==3:
            # 创建第二个y轴，共享x轴
            ax2 = ax[idx_i][idx_j].twinx()

            # 设置第二个y轴不显示刻度线
            ax2.yaxis.set_ticks([])

            # 在右侧y轴中央添加文本
            text = list(task_names_map.values())[idx_i]
            ax2.set_ylabel(text, rotation=270, labelpad=50,fontsize=18)
            ax2.yaxis.set_label_coords(1.1, 0.5)

        if idx_i ==0:
            ax[idx_i][idx_j].set_title(llm_name_map[llm_name], fontsize=16)
        ax[idx_i][idx_j].legend(loc='upper right', fontsize=14)

    # 图形设置
    # plt.xlabel(r'$k$', fontsize=18)
    fig.text(0.5, 0.04, r'$k$', ha='center', fontsize=18)
    # ax[0][0].set_ylabel(r'Cumulative distributions of $k$, $P_{k}$', fontsize=16)
    fig.text(0.08, 0.5, r'Cumulative distributions of $k$, $P_{k}$', va='center', rotation='vertical', fontsize=18)
    # fig.grid(True)
    # plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.9, hspace=0.4)
    # fig.legend(loc='lower center',ncol=4, fontsize=16)
    save_path = os.path.join(save_dir,f"{graph_name}_degree.pdf")
    plt.savefig(save_path)
    plt.clf()

def draw_ks_all_distribution(task,config):
    model_map = {
        "LN": "lognormal",
        "LNP": "lognormal_positive",
        "SE": "stretched_exponential",
        "EXP": "exponential",
        "TPL": "truncated_power_law",
        "PL": "power_law"
    }
    
    model_legend_map = {
        "LN": "Log-Normal",
        "LNP": "Log-Normal Positive",
        "EXP": "Exponential",
        "SE": "Stretched Exponential",
        "TPL": "Truncated Power-Law",
        "PL": "Power-Law"
    }

    # Use a lighter, high saturation academic color palette
    colors_a = sns.color_palette("muted", len(model_map))
    fig, ax = plt.subplots(1, 4, figsize=(15, 6), sharey=False)

    labels = []
    idx = 0

    patterns = ['/', '\\', '|', '-', '+', 'x', 'o']
    for llm_name, llm_plt_name in llm_name_map.items():
        config_name = f"search_shuffle_base_{llm_name}"
        llm_path_pl = os.path.join(llms_path_map[llm_plt_name], "article_citation_all_power_law.csv")
        df = pd.read_csv(llm_path_pl, index_col=0)
        ks_list = []
        for model_name, model_id in model_map.items():
            ks_list.append((df.loc[model_id, "KS"].mean(), df.loc[model_id, "KS"].std()))
        ax1 = ax[idx]
        # Bar plot
        bars = ax1.bar(list(model_map.keys()), [x[0] for x in ks_list], yerr=[x[1] for x in ks_list], capsize=12, color=colors_a, alpha=0.8)
        id_p =0
        for bar, label in zip(bars, model_map.keys()):
            bar.set_label(model_legend_map[label])
            # bar.set_hatch(patterns[id_p])
            id_p+=1


        labels.extend([bar.get_label() for bar in bars])

        ax1.set_xticks(range(len(model_map)))
        ax1.set_xticklabels(list(model_map.keys()), fontsize=14)
        label = ax1.axhline(y=0.1, color='#696969', linestyle='--', label='KS threshold = 0.1', alpha=0.7)
        labels.append(label.get_label())
        # ax1.legend(fontsize=16)
        ax1.set_title(llm_plt_name, fontsize=16)
        idx+=1

    for ax_ in ax:
        y = ax_.get_yticks()
        y = ["{:.2f}".format(i) for i in y]
        ax_.set_yticklabels(y, fontsize=14)
    ax[0].set_ylabel('K-S Distance', fontsize=16)
    
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.9, hspace=0.4)
    
    fig.legend(labels=labels[:7], loc='lower center', ncol=4, fontsize=16)
    # Save the figure
    plt.savefig(f"graph3/pl_reason_figures/3_ks_all_distribution.pdf")

if __name__ == "__main__":
    graphs = []
    max_nodes = 5000
    for task_name in task_names_map.keys():
        configs = []
        config_templates = config_templates_map[task_name]
        for llm in llm_name_map.keys():
            configs.extend([config_template.format(llm = llm) 
                            for config_template in config_templates])
        for config in configs:
            article_meta_info,author_info = get_data(task_name,config)
            graph = build_citation_graph(article_meta_info)
            if graph.number_of_nodes() > max_nodes:
                graph = graph.subgraph(list(graph.nodes())[:max_nodes])
                
            graphs.append(graph)

    draw_power_law(graphs,"citation")