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
from datetime import datetime, date
    
def plot_degree_figures(G:nx.Graph,
                 save_dir:str,
                 graph_name:str):
    # plot_pk_k(G,save_dir,graph_name)
    # plot_k_t_movie(G,save_dir,graph_name)
    # plot_node_t_movie(G,save_dir,graph_name)
    plot_node_variance(G,save_dir,graph_name)
    



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

# def plot_pk_k_movie(G:nx.Graph,
#             save_dir:str,
#             graph_name:str):
    
    
#     # 计算所有节点的出度
#     if nx.is_directed(G):
#         # 计算所有节点的出度
#         nodes_of_part_1 = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
#         in_degrees = [G.in_degree(n) for n in nodes_of_part_1]
#     else:
#         nodes_of_part_1 = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
#         in_degrees = [G.degree(n) for n in nodes_of_part_1]
    
#     ratings = []
#     for target_node in nodes_of_part_1:
#         if isinstance(G,nx.DiGraph):
#             in_edges = G.in_edges(target_node, data=True)
#         else:
#             in_edges = G.edges(target_node, data=True)
#             # 计算入边上'rating'属性的平均值
#         sum_of_ratings = sum(attr['rating'] for _, _, attr in in_edges)
#         average_rating = sum_of_ratings / len(in_edges) if in_edges else float('nan')  # 注意避免除以0
#         ratings.append(average_rating)

#     # 计算度分布
#     # degree_counts = np.bincount(in_degrees)
#     # degrees = np.arange(len(degree_counts))
#     degree_bins = {}
#     for node, in_degree, avg_rating in zip(nodes_of_part_1, in_degrees, ratings):
#         if in_degree ==0:continue
#         if in_degree in degree_bins.keys():
#             degree_bins[in_degree].append([node,avg_rating])
#         # 如果degree不存在于bins中，创建一个新的键值对
#         else:
#             degree_bins[in_degree] = [[node,avg_rating]]

#     # 移除度为0的点
#     degree_bins = dict(sorted(degree_bins.items(), key=lambda x: x[0]))
#     avg_ratings = []
#     for degree, nodes in degree_bins.items():
#         avg_ratings.append(sum([node[1] for node in nodes]) / len(nodes))
    
#     degrees = list(degree_bins.keys())
#     degree_counts = [len(v) for v in degree_bins.values()]
#     average_rating = avg_ratings

#     # 创建一个新的图形和轴对象
#     # 使用双对数坐标系绘制度分布图
#     plt.figure(figsize=(10, 6))
#     fig, ax1 = plt.subplots()

#     # 绘制loglog图（左侧纵坐标）
#     ax1.loglog(degrees, degree_counts, 'b', linewidth=2, markersize=8)
#     ax1.set_xlabel('Degree (k)')
#     ax1.set_ylabel('Frequency (P(k))', color='b')

#     # 为了分别调整左右y轴的风格，修改刻度颜色
#     for tl in ax1.get_yticklabels():
#         tl.set_color('b')

#     # Optional: 添加图例
#     ax1.legend(loc='upper left')


#     # # 使用twinx()函数创建一个共享x轴但独立y轴的新轴对象（右侧纵坐标）
#     # ax2 = ax1.twinx()

#     # # 绘制scatter图（右侧纵坐标）
#     # ax2.scatter(degrees, average_rating, color='r', label='rating score')
#     # ax2.set_ylabel('average rating score', color='r')
#     # for tl in ax2.get_yticklabels():
#     #     tl.set_color('r')
#     # ax2.legend(loc='upper right')

    
    

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     plt.savefig(os.path.join(save_dir,f"{graph_name}_degree.pdf"))
#     plt.clf()


            
def compare_items_movie(item1, item2):
    item1 = item1[1]
    item2 = item2[1]
    date_1 = datetime.strptime(item1.get("timestamp"), "%Y%m%d")
    date_2 = datetime.strptime(item2.get("timestamp"), "%Y%m%d")
    return 1 if date_1 > date_2 else -1
    

def plot_node_variance(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    from scipy.fftpack import fft
    nodes_of_part_1 = list(filter(lambda d:d[1]['bipartite'] == 1, G.nodes(data=True)))
    # 按照node的time进行排序
    nodes = sorted(nodes_of_part_1, key=cmp_to_key(compare_items_movie))

    data = {
        'node_id': [node[0] for node in nodes],
        'timestamp': [datetime.strptime(node[1].get("timestamp"),
                                        "%Y%m%d") for node in nodes]
    }

    df = pd.DataFrame(data)

    # 将时间字符串转换为pandas的datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df = df.query('timestamp < "2001-01-01"')
    # 选择合适的时间间隔，比如，按天、按周、按月分箱
    # 这里以按天为例，设置'1D'表示按天分箱
    # 你可以根据需要设置为'7D', '1M', '1Y'等

    # 使用pd.Grouper进行分箱，然后计数
    # 'freq'参数定义分箱的频率
    grouped = df.groupby(pd.Grouper(key='timestamp', freq='1D'))

    # 进行计数
    count_per_bin = grouped['node_id'].count()
    normalized_values = (count_per_bin.values - count_per_bin.values.min()) / (count_per_bin.values.max() - count_per_bin.values.min())

    plt.figure(figsize=(6, 10))

    ax1 = plt.subplot(2, 1, 1)


    # 左边的 Y 轴
    ax1.bar(count_per_bin.index, normalized_values, color='skyblue', alpha=0.7,)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Proportion of Movie Number', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

   
    node_indexs = [node[0] for node in nodes]
    
    # 计算度分布
    if nx.is_directed(G):
        degrees = [G.in_degree(n) for n in node_indexs]
    else:
        degrees = [G.degree(n) for n in node_indexs]
    
    data = {
        'timestamp': [datetime.strptime(node[1].get("timestamp"),
                                        "%Y%m%d") for node in nodes],
        'degree': degrees
    }

    # 创建DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df = df.query('timestamp < "2000-11-01"')
    # 注意，我们需要设定时间戳列为index，才能使用resample
    df.set_index('timestamp', inplace=True)
    
    
    # 计算每个箱（每天）的平均度数
    average_degree_per_bin = df.resample('1D').mean()['degree']
   
    # 可视化每个时间箱中的平均度数
    average_degree_per_bin = average_degree_per_bin.fillna(0)

    normalized_values = (average_degree_per_bin.values - average_degree_per_bin.values.min()) / (average_degree_per_bin.values.max() - average_degree_per_bin.values.min())
    
    normalized_values_filtered = np.array([x for x in normalized_values if x > 0.05])

    values_fft = fft(normalized_values_filtered)
    N = len(normalized_values_filtered)
    frequencies = np.fft.fftfreq(N, d = 1)
    positive_freqs = frequencies[:N // 2]
    positive_fft = np.abs(values_fft)[:N // 2]
    peak_freq = positive_freqs[np.argmax(positive_fft)]
    
    peak_index = np.argmax(positive_fft)
    peak_freq = positive_freqs[peak_index]
    peak_amplitude = positive_fft[peak_index]
    noise_indices = np.where(positive_freqs != peak_freq)
    noise_amplitude = np.mean(positive_fft[noise_indices])
    SNR = peak_amplitude / noise_amplitude
    print(f"SNR: {graph_name}", SNR)

    # # 生成拟合的正弦曲线
    # time = np.arange(0, len(average_degree_per_bin.index), 1)  # 时间
    # amplitude = np.max(positive_fft) / N
    # fit_sine_wave = amplitude * np.sin(2 * np.pi * peak_freq * time)

    # ax1.plot(average_degree_per_bin.index, np.abs(fit_sine_wave), label='Fitted Sine Wave', color='coral', linestyle='--',
    #          )
    # plt.legend()
    # 右边的 Y 轴
    ax2 = ax1.twinx()
    ax2.bar(average_degree_per_bin.index,normalized_values, color='#000080', alpha=1.0,)
    # ax2.set_ylabel('Proportion 2', color='coral')
    ax2.tick_params(axis='y', labelcolor='#000080')
    ax2.set_ylabel('Proportion of $k$', color='#000080')
    plt.xticks([])
    plt.xlabel('Time')
    # plt.legend()
    # 计算傅里叶变换

    ax1 = plt.subplot(2, 1, 2)
    
    ax1.stem(frequencies[1:N // 2], np.abs(values_fft)[1:N // 2], 'b', markerfmt=" ", basefmt="-b", label = f"Degree Distribution ($k$), SNR = {SNR:.2f} dB")
    
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum for $k$')
    plt.grid(True)
    # 添加标题和标签
    # plt.title('Node Count per Day')
    
    plt.savefig(os.path.join(save_dir,f"{graph_name}_movie_day.pdf"))
    # plt.ylabel('Proportion of Movie Number')


def plot_node_t_movie(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    nodes_of_part_1 = list(filter(lambda d:d[1]['bipartite'] == 1, G.nodes(data=True)))
    # 按照node的time进行排序

    nodes = sorted(nodes_of_part_1, key=cmp_to_key(compare_items_movie))

    data = {
        'node_id': [node[0] for node in nodes],
        'timestamp': [datetime.strptime(node[1].get("timestamp"),
                                        "%Y%m%d") for node in nodes]
    }

    df = pd.DataFrame(data)

    # 将时间字符串转换为pandas的datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 选择合适的时间间隔，比如，按天、按周、按月分箱
    # 这里以按天为例，设置'1D'表示按天分箱
    # 你可以根据需要设置为'7D', '1M', '1Y'等

    # 使用pd.Grouper进行分箱，然后计数
    # 'freq'参数定义分箱的频率
    grouped = df.groupby(pd.Grouper(key='timestamp', freq='1D'))

    # 进行计数
    count_per_bin = grouped['node_id'].count()
    normalized_values = (count_per_bin.values - count_per_bin.values.min()) / (count_per_bin.values.max() - count_per_bin.values.min())

    # 开始绘图
    plt.figure(figsize=(10, 6))  # 可以调整画布大小

    plt.bar(count_per_bin.index, normalized_values, color='skyblue')

    # 添加标题和标签
    plt.title('Node Count per Day')
    plt.xlabel('Time')
    plt.ylabel('Proportion of Movie Number')

    # 优化x轴日期标签显示
    plt.gcf().autofmt_xdate()

    # 显示网格
    plt.grid(True, which="both", ls="--")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_movie_day.pdf"))
    plt.clf()
    

def plot_k_t_movie(G:nx.DiGraph,
            save_dir:str,
            graph_name:str):
    nodes_of_part_1 = list(filter(lambda d:d[1]['bipartite'] == 1, G.nodes(data=True)))
    # 按照node的time进行排序

    nodes = sorted(nodes_of_part_1, key=cmp_to_key(compare_items_movie))
    
    node_indexs = [node[0] for node in nodes]
    
    # 计算度分布
    if nx.is_directed(G):
        degrees = [G.in_degree(n) for n in node_indexs]
    else:
        degrees = [G.degree(n) for n in node_indexs]
    
    
    data = {
        'timestamp': [datetime.strptime(node[1].get("timestamp"),
                                        "%Y%m%d") for node in nodes],
        'degree': degrees
    }

    # 创建DataFrame
    df = pd.DataFrame(data)
    # 使用双对数坐标系绘制度分布图
    plt.figure(figsize=(10, 6))
    # 将timestamp转换为Pandas datetime类型
    # 注意，我们需要设定时间戳列为index，才能使用resample
    df.set_index('timestamp', inplace=True)

    # 计算每个箱（每天）的平均度数
    average_degree_per_bin = df.resample('1D').mean()['degree']
    # 可视化每个时间箱中的平均度数
    plt.bar(average_degree_per_bin.index, 
            average_degree_per_bin.values, color='#000080')


    # plt.plot(node_times, degrees, 'bo-', linewidth=2, markersize=8)
    # 添加图表标题和轴标签
    plt.title('Average Degree per Day')
    plt.xlabel('Date')
    plt.ylabel('Average Degree')
    # 优化x轴日期标签显示
    plt.gcf().autofmt_xdate()
    plt.grid(True, which="both", ls="--")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,f"{graph_name}_t_k.pdf"))
    plt.clf()



