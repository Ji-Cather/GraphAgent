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
from datetime import datetime

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1. / (2. * sigma ** 2)
    dist = torch.sum(x**2, 1).unsqueeze(1) + torch.sum(y**2, 1) - 2 * torch.matmul(x, y.t())
    return torch.exp(-beta * dist)

def compute_mmd(x, y, sigma=1.0):
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    res = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return res.tolist()

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
    # plot_shrinking_diameter(G,n=100,save_dir=save_dir,graph_name=graph_name)
 
def plot_shrinking_diameter(df:pd.DataFrame,
                            save_dir:str,
                            graph_name:str):
    # df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].to_list()
    dates = [str(date) for date in dates]
    # 绘制折线图
    plt.plot(dates, df['follower_diameter'], label='Follower Diameter')
    plt.plot(dates, df['friend_diameter'], label='Friend Diameter')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('Follower and Friend Diameters Over Time')
    plt.xlabel('Date')
    plt.ylabel('Diameter')
    # 选择要显示的日期值
    date_values = []
    
    thunk_size = len(dates)//5
    thunk_size = 1 if thunk_size==0 else thunk_size
    for idx in range(0,len(dates),thunk_size):
        date_values.append(dates[idx])
    # 设置x轴刻度
    plt.xticks(date_values, rotation=45)


    save_path = os.path.join(save_dir,f"{graph_name}_diameter.pdf")
    os.makedirs(save_dir,exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    
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

# 定义比较函数
def compare_items(item1, item2):
    item1 = item1[1]
    item2 = item2[1]
    date_1 = datetime.strptime(item1.get("date"), "%Y%m%d")
    date_2 = datetime.strptime(item2.get("date"), "%Y%m%d")
    return 1 if date_1 > date_2 else -1


def plot_k_t(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    nodes = G.nodes()
    
    # 按照node的time进行排序
    nodes = sorted(nodes._nodes.items(), key=cmp_to_key(compare_items))
    
    node_indexs = [node[0] for node in nodes]
    node_indexs.reverse()
    # 计算度分布
    degrees = [G.in_degree(n) for n in node_indexs]
    
    times = [G.nodes(data=True)[n]["date"] for n in node_indexs]

    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    plt.plot(times, degrees, 'bo-', linewidth=2, markersize=8)
    plt.title('Time - Degree')
    plt.xlabel('Time')
    plt.ylabel('Degree k')
    plt.grid(True, which="both", ls="--")
    plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_t_k.pdf"))
    plt.clf()
    
def plot_indegree_outdegree(G:nx.DiGraph,
                            save_dir:str,
                        graph_name:str):
    # 计算每个节点的入度和出度
    in_degrees = np.array([d for n, d in G.in_degree()])
    out_degrees = np.array([d for n, d in G.out_degree()])

    # 绘制散点图
    plt.scatter(in_degrees, out_degrees, color='blue', label='Nodes')

    # 绘制y=x的虚线作为参考
    x = np.linspace(0, max(in_degrees.max(), out_degrees.max()), 100)
    plt.plot(x, x, linestyle='--', color='red', label='y=x')

    # 添加图例和标签
    plt.legend()
    if graph_name == "follow":
        plt.xlabel('In-Degree')
        plt.ylabel('Out-Degree')
    # plt.title('In-Degree vs Out-Degree Distribution')
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_average_degree.pdf")
    plt.savefig(save_path)
    plt.clf()

  
    
def divide_list(lst, k):
    # 计算每份的基本长度
    n = len(lst)
    base_size = n // k
    remainder = n % k
    
    # 创建结果列表
    result = []
    
    # 记录当前位置
    start = 0
    
    for i in range(k):
        # 如果还有余数，当前分组加1
        end = start + base_size + (1 if i < remainder else 0)
        # 将当前分组添加到结果列表
        result.append(lst[start:end])
        # 更新下一分组的起始位置
        start = end
    
    return result

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

def plot_densification_power_law(matrix_dir:str,
                        save_dir:str,
                        graph_name:str):
    matrix_names = ["action",
                    "follow",
                    "friend"]
    from scipy.stats import linregress

    for matrix_name in matrix_names:
        matrix_path = os.path.join(matrix_dir, f"{matrix_name}_matrix.csv")
        matrix = pd.read_csv(matrix_path,index_col = 0)
        dates = []
        for index, row in matrix.iterrows():
            date = index.split("_")[0]
            dates.append(date)
        y = matrix["Edges"].to_numpy()
        x = matrix["Nodes"].to_numpy()

        # 线性回归拟合
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # 准备回归线数据
        line = slope * x + intercept

        # 绘制数据点
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data')

        # 绘制回归线
        plt.plot(x, line, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.4f}')

    # 设置图表的标题和轴标签
    plt.title('Log-Log plot of Edges vs Nodes')
    plt.xlabel('Log of Number of Nodes')
    plt.ylabel('Log of Number of Edges')

    # 添加图例
    plt.legend()

    # 启用网格
    plt.grid(True, which="both", ls="--")

    # 使用log-log比例
    plt.xscale('log')
    plt.yscale('log')

    save_path = os.path.join(save_dir, 
                            f"{graph_name}_densification_power_law.pdf")
    # 显示图表
    plt.savefig(save_path)
    plt.clf()


def plot_mean_degree(matrix_dir:str,
                        save_dir:str,
                        graph_name:str):
    matrix_names = [
                    # "action",
                    # "follow",
                    "friend"]
    
    for matrix_name in matrix_names:
        matrix_path = os.path.join(matrix_dir, f"{matrix_name}_matrix.csv")
        matrix = pd.read_csv(matrix_path,index_col = 0)
        dates = []
        for index, row in matrix.iterrows():
            date = index.split("_")[0]
            dates.append(date)
        mean_degrees = matrix["Mean Degree"].to_list()
        import powerlaw
        results = powerlaw.Fit(list(mean_degrees), discrete=True)

        alpha = results.power_law.alpha
        xmin = results.power_law.xmin
        print(f"{matrix_name} graph: alpha = {alpha}")
        # 绘制曲线图
        plt.plot(dates, mean_degrees, 
                 marker='o', linestyle='-', label=f"{matrix_name} graph")
        
    
    save_path = os.path.join(save_dir, 
                                f"{graph_name}_average_degree.pdf")
    plt.title('Average Degree of Graph over Time (Densification Law)')
    plt.xlabel('Time')
    plt.ylabel('Average Degree')
    plt.legend()
    plt.grid(True)

    # 选择要显示的日期值
    date_values = [] 
    thunk_size = len(dates)//5
    thunk_size = 1 if thunk_size==0 else thunk_size
    for idx in range(0,len(dates),thunk_size):
        date_values.append(dates[idx])
    plt.xticks(date_values, rotation=45)

    plt.savefig(save_path)
    plt.clf()



if __name__ =="__main__":
    pass
    # 示例：评估有向图
    # G_true = nx.fast_gnp_random_graph(100, 0.05, directed=True)
    # G_generated = nx.fast_gnp_random_graph(100, 0.05, directed=True)

    
    # print()
    # plot_k_t_llm_agent()
