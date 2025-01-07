import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import seaborn as sns
import networkx as nx
from LLMGraph.utils.io import readinfo,writeinfo
# 计算基尼系数
def gini_coefficient(x):
    n = len(x)
    x = np.array(x)
    x_sum = np.sum(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return 2 * np.sum((2 * index - n - 1) * x) / (n * x_sum) - (n + 1) / n

def plot_gini(citation_data:dict,
              save_dir:str,
              data_type="author"):
    import numpy as np

    writeinfo(os.path.join(save_dir,f"in_degrees_{data_type}.json"),
              citation_data)
    # 提取引用数量
    citations = list(citation_data.values())
    # 计算基尼系数
    gini = gini_coefficient(citations)

    # # 绘制基尼系数曲线
    # plt.figure(figsize=(8, 6))
    # plt.plot(np.cumsum(np.sort(citations)) / np.sum(citations), label='Lorenz curve for {data_type}: gini = {gini:.2f}')
    # plt.plot([0, 1], [0, 1], 'k--', label=f'Line of perfect equiality')
    # plt.title('Lorenz curve')
    # plt.xlabel(f'Cumulative share of {data_type}')
    # plt.ylabel('Cumulative share of citations')
    # plt.fill_between(np.linspace(0, 1, len(citations)), np.linspace(0, 1, len(citations)), alpha=0.1)
    # plt.grid(True)
    # plt.legend()

    # 提取cited数量并按从小到大排序。
    cited_counts = sorted(citation_data.values())

    # 计算累计份额。
    cum_cited = np.cumsum(cited_counts)
    cum_cited_share = cum_cited / cum_cited[-1]

    # 构建洛伦兹曲线的坐标。
    # lorenz_curve_x = np.arange(1, len(cited_counts) + 1) / len(cited_counts)
    lorenz_curve_x = np.insert(np.arange(1, len(cited_counts) + 1) / len(cited_counts), 0, 0)
    lorenz_curve_y = np.insert(cum_cited_share, 0, 0)  # 插入0在开始位置。

    # 计算对角线下的面积和洛伦兹曲线下的面积。
    area_under_diagonal = 0.5
    area_under_lorenz_curve = np.trapz(lorenz_curve_y, lorenz_curve_x)

    # 计算基尼系数。
    gini_index = (area_under_diagonal - area_under_lorenz_curve) / area_under_diagonal

    # 输出基尼系数。
    print("基尼系数为:", gini_index)

    # 绘制基尼系数曲线。
    plt.figure()
    plt.fill_between(lorenz_curve_x, 0, lorenz_curve_y, alpha=0.7, color='skyblue', label='Lorenz curve')
    plt.plot(lorenz_curve_x, lorenz_curve_y, label=f'Lorenz curve for {data_type}: gini = {gini:.2f}', color='darkblue')

    plt.plot([0, 1], [0, 1], 'k--', label=f'Line of perfect equiality',color='darkgreen')
    plt.title('Lorenz curve')
    plt.xlabel(f'Cumulative share of {data_type}')
    plt.ylabel('Cumulative share of citations')
    plt.grid(True)
    plt.legend()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,f"{data_type}_gini.pdf")
    plt.savefig(save_path)
    plt.clf()

def plot_self_citation(G:nx.DiGraph,
                       save_dir:str,
                       threshold:int = 1000):
    # G = G.subgraph(nodes= list(G.nodes())[:threshold])
    # 初始化两个字典来存储总引用次数和自引用次数
    total_citations = {}
    self_citations = {}
    country_pub_nums = {}

    countrys = readinfo("evaluate/article/country.json")
    countrys_list = []
    for v in countrys.values():
        for v_ in v:
            countrys_list.append(v_.lower())

    for c in countrys_list:
        total_citations[c] = 0
        self_citations[c] = 0

    # 遍历图中的所有边，统计总引用次数和自引用次数
    for n1, n2 in G.edges():
        country1_list = G.nodes(data=True)[n1]["country"]
        country2_list = G.nodes(data=True)[n2]["country"]
        for country1 in country1_list[:1]:
            for country2 in country2_list[:1]:
                if country1 == country2:
                    self_citations[country1] += 1
                
                total_citations[country1] +=1
                
    for node,node_info in list(G.nodes(data=True)):
        country_list = node_info["country"][:1]
        for country in country_list:
            country_pub_nums[country] = country_pub_nums.get(country, 0) + 1
    os.makedirs(save_dir,exist_ok=True)
    writeinfo(os.path.join(save_dir,f"country_pub_nums.json"),
              country_pub_nums)

    total_citations = dict(sorted(total_citations.items(),key = lambda x:x[1], reverse=True))
    # 计算每个国家的自引率
    self_citation_rates = {
        country: self_citations[country] / total_citations[country]
        for country in total_citations
        if total_citations[country] > 0  # 防止除以零
    }
    self_citation_rates = dict(sorted(self_citation_rates.items(),key = lambda x:x[1], reverse=True))
    
    writeinfo(os.path.join(save_dir,f"self_citation_rate.json"),
              self_citation_rates)
    writeinfo(os.path.join(save_dir,f"citation_rate.json"),
              total_citations)
    
    # 计算绘图所需的数据
    countries = list(self_citation_rates.keys())  # 国家名称列表
    rates = list(self_citation_rates.values())  # 对应的自引率列表

    # 创建绘图
    # plt.figure(figsize=(10, 6))  # 可以调整大小以更好地适应国家数量
    plt.bar(countries, rates, color='skyblue')  # 创建条形图

    # 添加标题和轴标签
    plt.title('Self-Citation Rates by Country')
    plt.xlabel('Country')
    plt.ylabel('Self-Citation Rate')

    # 改善横坐标标签显示
    plt.xticks(rotation=45, ha='right')  # 将国家名称标签旋转45度以减少重叠

    # 显示图表
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图表区域

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,f"self_citation_rate.pdf")
    plt.savefig(save_path)
    plt.clf()