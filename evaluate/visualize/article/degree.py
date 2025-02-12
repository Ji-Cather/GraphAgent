import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import torch
import json
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
from functools import cmp_to_key
import pandas as pd
import matplotlib.dates as mdates
def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list





def compute_nll(true_samples, generated_samples):
    # 使用真实样本估计正态分布参数
    mu = np.mean(true_samples)
    sigma = np.std(true_samples)
    
    # 计算生成样本的NLL
    nll = -np.mean(norm.logpdf(generated_samples, loc=mu, scale=sigma))
    return nll


def plot_degree_figures(G:nx.DiGraph,
                 save_dir:str,
                 graph_name:str):
    plot_pk_k(G,save_dir,graph_name)
    # plot_k_t(G,save_dir,graph_name)
    # plot_shrinking_diameter(G,k=100,save_dir=save_dir,graph_name=graph_name)
    


def plot_pk_k(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    # 计算所有节点的出度
    import powerlaw
    from scipy.stats import kstest
    degree_list = [G.degree(n) for n in G.nodes()]
    
    degree_list = sorted(degree_list,reverse=True)
    plt.figure(figsize=(10, 6))
    # 使用powerlaw进行幂律分布拟合
    try:
        # results = powerlaw.Fit(list(degree_list), discrete=True,
        #                        fit_method="KS")

        results = powerlaw.Fit(list(degree_list), discrete=True,
                        fit_method="KS")
    except:
        pass
    
    alpha = results.power_law.alpha
    xmin = results.power_law.xmin
    sigma = results.power_law.sigma
    degree_list = list(filter(lambda x:x>xmin, degree_list))
    # 在图表中添加文本信息
    textstr = '\n'.join((
        r'$\alpha=%.2f$' % (alpha,),
        r'$\mathrm{x_{min}}=%.2f$' % (xmin,),
        r'$\sigma=%.2f$' % (sigma,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.8, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # 计算度分布
    degree_counts = np.bincount(degree_list)
    degrees = np.arange(len(degree_counts))
    results.plot_pdf(color='b', 
                    marker='o', 
                    label='Log-binned Data',)
    # results.plot_pdf(color='g',
    #                  marker='d',  linear_bins = True,
    #                  label='Linearly-binned Data',)
    # 拟合的幂律分布
    results.power_law.plot_pdf(color='r', 
                               linestyle='--', 
                               linewidth=2,
                               label='Power Law Fit, $\\alpha$ = {alpha:.2f}'.format(alpha=alpha))
    # 移除度为0的点
    degrees = degrees[degree_counts > 0]
    degree_counts = degree_counts[degree_counts > 0]


    # 使用双对数坐标系绘制度分布图
    # plt.loglog(degrees, degree_counts, 'bo', linewidth=2, markersize=8,label='Degree distribution')
    
    # plt.title('Log-Log Plot of Paper Citation Network Degree Distribution')
    plt.xlabel('Degree $k$', fontsize=16)  # 单独设置坐标轴标签字体大小
    plt.ylabel('PDF of Degree Distribution, $P(k)$', fontsize=16)  # 单独设置坐标轴标签字体大小
    plt.grid(True, which="major", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_degree.pdf"))
    plt.clf()

def print():
    import json
    with open("LLMGraph/tasks/llm_agent/data/article_meta_info.json",'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    cites = [ value["cited"] for value in data_list.values()]
    
    # 计算度分布
    degree_counts = np.bincount(cites)
    degrees = np.arange(len(degree_counts))

    # 移除度为0的点
    # degrees = degrees[degree_counts > 0]
    # degree_counts = degree_counts[degree_counts > 0]


    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, degree_counts, 'bo-', linewidth=2, markersize=8)
    plt.title('Log-Log Plot of Paper Citation Network Degree Distribution')
    plt.xlabel('Degree (k)')
    plt.ylabel('Frequency (P(k))')
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.savefig("degree.pdf")
    plt.clf()
    
    


# 定义比较函数
def compare_items(item1, item2):
    item1 = item1[1]
    item2 = item2[1]
    # 检查time属性，如果都存在，则直接比较time
    if item1.get("time")is not None and item2.get("time") is not None:
        return (item1.get("time")> item2.get("time")) - (item1.get("time")< item2.get("time"))
    # 如果item1的time不存在，而item2的time存在，认为item1小于item2
    elif item1.get("time")is None and item2.get("time")is not None:
        return 1
    # 如果item2的time不存在，而item1的time存在，认为item1大于item2
    elif item1.get("time")is not None and item2.get("time")is None:
        return -1
    # 如果两者的time都不存在，比较round_id
    else:
        return (item1.get("round_id") > item2.get("round_id")) - \
            (item1.get("round_id") < item2.get("round_id"))
  

def plot_k_t(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    nodes = G.nodes()
    
    # 按照node的time进行排序

    nodes = sorted(nodes._nodes.items(), key=cmp_to_key(compare_items))
    
    node_indexs = [node[0] for node in nodes]
    node_indexs.reverse()
    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    # 计算度分布
    degrees = [G.in_degree(n) for n in node_indexs]
    dates = [G.nodes(data=True)[n]["time"] for n in node_indexs]
    dates = pd.to_datetime(dates)
    plt.plot(dates, degrees, 'bo-', linewidth=2, markersize=8)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.title('t - k')
    plt.xlabel('t')
    plt.ylabel('k')
    plt.grid(True, which="both", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_t_k.pdf"))
    plt.clf()
    
    

    
    
def divide_list(lst, 
                k,
                start = 0 # 记录当前位置
                ):
    # lst 分成k份
    # 计算每份的基本长度
    n = len(lst) - start
    base_size = n // k
    remainder = n % k
    
    # 创建结果列表
    result = []
    for i in range(k):
        # 如果还有余数，当前分组加1
        end = start + base_size + (1 if i < remainder else 0)
        # 将当前分组添加到结果列表
        result.append(lst[:end])
        # 更新下一分组的起始位置
        start = end
    
    return result
    
def plot_shrinking_diameter(DG:nx.DiGraph,
                            k:int = 100,
                            save_dir:str= "",
                            graph_name:str = ""):
    nodes = DG.nodes()
    nodes = sorted(nodes._nodes.items(), key=cmp_to_key(compare_items)) # 早->晚
    
    nodes_chunk = divide_list(nodes,k)
    
    sub_graphs = []
    sub_nodes = []
    for node_chunk in nodes_chunk:
        sub_nodes.extend([node[0] for node in node_chunk])
        sub_graph = DG.subgraph(sub_nodes).copy()
        sub_graphs.append(sub_graph)
    
    diameters =[]
    for sub_graph in sub_graphs:
        diameter = calculate_diameter(sub_graph)
        diameters.append(diameter)
    
    # 假设直径d随时间t缩减，这里我们使用一个简单的例子：d = e^(-t)
    t = np.linspace(0, len(nodes), k)  # 时间从0到5，总共100个点

    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(t, diameters, label='Shrinking Diameter')  # 绘制直径随时间变化的图表
    plt.title('Diameter Shrinking Over Time')  # 图表标题
    plt.xlabel('Time')  # x轴标签
    plt.ylabel('Diameter')  # y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_diameter.pdf"))
    plt.clf()

def calculate_diameter(DG:nx.DiGraph):
    lengths = dict(nx.all_pairs_shortest_path_length(DG))

    # 初始化直径为0
    diameter = 0

    # 遍历所有路径长度，找出最大值
    for source, targets in lengths.items():
        for target, length in targets.items():
            if length > diameter:
                diameter = length
    return diameter





if __name__ =="__main__":
    # 示例：评估有向图
    # G_true = nx.fast_gnp_random_graph(100, 0.05, directed=True)
    # G_generated = nx.fast_gnp_random_graph(100, 0.05, directed=True)


    
    # plot_degree_figures()
    # print()
    pass
