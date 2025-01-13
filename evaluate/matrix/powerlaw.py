
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import os

def pdf(data, xmin=None, xmax=None, linear_bins=False, **kwargs):
    """
    Returns the probability density function (normalized histogram) of the
    data.

    Parameters
    ----------
    data : list or array
    xmin : float, optional
        Minimum value of the PDF. If None, uses the smallest value in the data.
    xmax : float, optional
        Maximum value of the PDF. If None, uses the largest value in the data.
    linear_bins : float, optional
        Whether to use linearly spaced bins, as opposed to logarithmically
        spaced bins (recommended for log-log plots).

    Returns
    -------
    bin_edges : array
        The edges of the bins of the probability density function.
    probabilities : array
        The portion of the data that is within the bin. Length 1 less than
        bin_edges, as it corresponds to the spaces between them.
    """
    from numpy import logspace, histogram, floor, unique,asarray
    from math import ceil, log10
    data = asarray(data)
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)

    if xmin<1:  #To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
        xmax2=xmax/xmin
        xmin2=1
    else:
        xmax2=xmax
        xmin2=xmin

    if 'bins' in kwargs.keys():
        bins = kwargs.pop('bins')
    elif linear_bins:
        bins = range(int(xmin2), ceil(xmax2)+1)
    else:
        log_min_size = log10(xmin2)
        log_max_size = log10(xmax2)
        number_of_bins = ceil((log_max_size-log_min_size)*10)
        bins = logspace(log_min_size, log_max_size, num=number_of_bins)
        bins[:-1] = floor(bins[:-1])
        bins[-1] = ceil(bins[-1])
        bins = unique(bins)

    if xmin<1: #Needed to include also data x<1 in pdf.
        hist, edges = histogram(data/xmin, bins, density=True)
        edges=edges*xmin # transform result back to original
        hist=hist/xmin # rescale hist, so that np.sum(hist*edges)==1
    else:
        hist, edges = histogram(data, bins, density=True)

    return edges, hist


def calculate_power_law(G:nx.DiGraph,
                        save_dir:str,
                        graph_name = "G",
                         plt_flag=True,
                         xmin:int = 3):
    
    compare_distributions = ['truncated_power_law',
                             'lognormal',
                              'lognormal_positive',
                                'stretched_exponential',
                                 'exponential',
                                 ]
    # degree_types = ["all", "in", "out"] if isinstance(G, nx.DiGraph) else ["all"]
    degree_types = ["in"]

    power_law_dfs = {}
    for degree_type in degree_types:
        alpha, xmin, sigma, KS_list, R_list, p_value_LL_list = calculate_macro_properties(G, 
                                                                                          xmin=xmin,
                                                                                   plt_flag=plt_flag,
                                                                                   save_dir=save_dir,
                                                                                   graph_name=graph_name,
                                                                                   degree_type=degree_type,
                                                                                compare_distributions=\
                                                                                    compare_distributions)
        df = pd.DataFrame()
        for R, p_value_LL, KS, compare_distribution in zip(R_list, 
                                                           p_value_LL_list, 
                                                           KS_list[:-1], 
                                                           compare_distributions):
            df.loc[compare_distribution,"p"] = p_value_LL
            df.loc[compare_distribution,"LR"] = R
            df.loc[compare_distribution,"KS"] = KS


        df.loc["power_law",'alpha'] = alpha
        df.loc["power_law",'xmin'] = xmin
        df.loc["power_law",'sigma'] = sigma
        df.loc["power_law",'KS'] = KS_list[-1]

        power_law_dfs[degree_type] = df

    return power_law_dfs

def calculate_macro_properties(G:nx.Graph, 
                               save_dir:str,
                                graph_name:str,
                                xmin:int = 3,
                                plt_flag:bool = False,
                               degree_type="all",
                               compare_distributions = [
                                   'lognormal',
                                 'exponential',
                                 'stretched_exponential',
                                 'lognormal_positive',
                                 'truncated_power_law']
                               ):
    """
    计算优先连接指数及K-S检验值

    Params:
    - G (nx.Graph): 网络图
    - type (str): 数据类型，可以是 "article", "movielens", "social"（默认是 "article")

    Returns:
    - alpha (float): 拟合的幂律分布的指数
    - xmin (float): 拟合的幂律分布的最小阈值
    - ks_statistic (float): K-S 检验的统计量
    - p_value (float): K-S 检验的 p 值
    """
    os.makedirs(save_dir,exist_ok=True)
    import powerlaw
    from scipy.stats import kstest
    if isinstance(G, nx.DiGraph):
        if degree_type == "in":
            degree_list = [G.in_degree(n) for n in G.nodes()]
        elif degree_type == "out":
            degree_list = dict(G.out_degree()).values()
        else:
            degree_list = [G.degree(n) for n in G.nodes()]
    
    elif isinstance(G, nx.Graph):
        degree_list = [G.degree(n) for n in G.nodes()]
    
    # degree_list = sorted(degree_list,reverse=True)
    R_list, p_value_LL_list =[], []
    KS_list = []
    
    # 使用powerlaw进行幂律分布拟合
    try:
        # results = powerlaw.Fit(list(degree_list), discrete=True,
        #                        fit_method="KS")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm 
        font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
        font_prop = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)

        # 设置全局字体
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        if xmin==0:
            xmin = None
        # xmin = int(0.05*max(degree_list))
        
        results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                sigma_threshold = .1,
                                #     fit_method="KS",
                                    # xmin=xmin,
                                    )
        alpha = results.power_law.alpha
        xmin = results.power_law.xmin
        sigma = results.power_law.sigma

        if plt_flag and degree_type =="all":
            # 绘制幂律分布的对数-对数图
            plt.figure(figsize=(8, 6))
            kwargs = {
                "color":'b', 
                "linestyle":'--', 
                "linewidth":2, 
                "label":'Data'
            }
            
            # 原始数据的直方图
            data_save_path = os.path.join(save_dir,f"{graph_name}_degree.npy")
            np.save(data_save_path, np.array(list(degree_list)))
            # results.plot_pdf(color='b', 
            #                  linestyle='-', linewidth=2, label='Data')
            results.plot_pdf(color='b', 
                              marker='o', 
                              label='Log-binned Data',)
            # results.plot_pdf(color='g',
            #                  marker='d',  linear_bins = True,
            #                  label='Linearly-binned Data',)
            # 拟合的幂律分布
            D = results.power_law.D
            results.power_law.plot_pdf(color='r', 
                                       linestyle='--', 
                                       linewidth=2,
                                         label=f"Power Law Fit, $\\alpha$ = {alpha:.2f}, $D_{{ks}}$ = {D:.2f}")
            # results.truncated_power_law.plot_pdf(color='c', linestyle='--', linewidth=2, label='Truncated Power Law Fit')
            # results.lognormal.plot_pdf(color='g', linestyle='--', linewidth=2, label='Log-Normal Fit')
            # results.lognormal_positive.plot_pdf(color='k', linestyle='--', linewidth=2, label='Lognormal Positive Fit')
            # results.stretched_exponential.plot_pdf(color='m', linestyle='--', linewidth=2, label='Stretched Exponential Fit')
            # results.exponential.plot_pdf(color='y', linestyle='--', linewidth=2, label='Exponential fit')
            # degree_count = np.bincount(np.array(degree_list))
            # degrees = np.arange(len(degree_count))
            # plt.scatter(degrees, degree_count, marker='o', color='b', label='Degree Distribution')
            
            # 图形设置
            plt.xlabel(r'$k$', fontsize=18)
            plt.ylabel(r'Cumulative distributions of $k$, $P_{k}$', fontsize=18)
            # plt.title('Power Law Fit on Log-Log Plot', fontsize=18)
            plt.legend(loc='upper right', fontsize=18)
            plt.grid(True)
            save_path = os.path.join(save_dir,f"{graph_name}_degree.pdf")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(save_path)
            plt.clf()
        for compare_distribution in compare_distributions:
            R, p_value_LL = results.distribution_compare(compare_distribution, "power_law")
            KS = getattr(results, compare_distribution).D

            KS_list.append(KS)
            R_list.append(R)
            p_value_LL_list.append(p_value_LL)
            

        KS_list.append(results.power_law.D)
    except Exception as e:
        try:
            if plt_flag:
                # 绘制幂律分布的对数-对数图
                plt.figure(figsize=(8, 6))

                # 原始数据的直方图
                results.plot_pdf(color='b', linestyle='-', linewidth=2, label='Data')
            
                # 图形设置
                plt.xlabel(r'$x$', fontsize=15)
                plt.ylabel(r'$P(x)$', fontsize=15)
                # plt.title('Power Law Fit on Log-Log Plot', fontsize=18)
                plt.legend(loc='best')
                plt.grid(True)
                save_path = os.path.join(save_dir,f"{graph_name}_degree.pdf")
                plt.savefig(save_path)
                plt.clf()
        except:
            pass
        for compare_distribution in compare_distributions:
            try:
                R, p_value_LL = results.distribution_compare(compare_distribution, "power_law")
                KS = getattr(results, compare_distribution).D
                ll = getattr(results, compare_distribution).loglikelihood
            except:
                R, p_value_LL = np.nan, np.nan
                KS = np.nan
                ll = np.nan
            KS_list.append(KS)

            R_list.append(R)
            p_value_LL_list.append(p_value_LL)
            
        alpha = results.power_law.alpha
        xmin = results.power_law.xmin
        sigma = results.power_law.sigma
        KS_list.append(results.power_law.D)

        return alpha, xmin, sigma, KS_list, R_list, p_value_LL_list

    
        
    if plt_flag and degree_type =="all":
        likelihood_ratios = np.array(R_list)
        ks_values = np.array(KS_list)

        # 绘制柱状图比较各个分布的拟合程度
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        
        # Likelihood Ratio的柱状图
        ax[0].bar(compare_distributions, likelihood_ratios, color='b')
        ax[0].set_title('Likelihood Ratio Comparison')
        ax[0].set_ylabel('Likelihood Ratio')

        # KS的柱状图
        ax[1].bar([*compare_distributions,
                "power_law"], ks_values, color='g')
        ax[1].set_title('KS Statistic Comparison')
        ax[1].set_ylabel('KS Statistic')

        plt.tight_layout()
        save_path = os.path.join(save_dir,f"{graph_name}_fit_err.png")
        plt.savefig(save_path)
        plt.clf()


    return alpha, xmin, sigma, KS_list, R_list, p_value_LL_list