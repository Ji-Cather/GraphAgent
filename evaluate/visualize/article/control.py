import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
import os
from LLMGraph.utils.io import readinfo, writeinfo

def plot_nc(nc,
    nse,
    ncdeg,
    graph_name: str,
    save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # 绘制第一组数据的散点图
    plt.scatter(nc,nse, color='blue', label='NSE')

    # 绘制第二组数据的散点图
    plt.scatter(nc,ncdeg, color='red', label='Nc Deg')
    # 添加y=x的线
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='y=x')

    # 设置横轴和纵轴的范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # 添加轴标签
    plt.xlabel('NSE / Nc Deg')
    plt.ylabel('NC')

    # 添加图例
    plt.legend()

    # 显示图表
    save_path = os.path.join(save_dir, f"nc_{graph_name}.pdf")
    plt.savefig(save_path)
    plt.clf()

import re
def plot_err(beta_dict_path,
             save_dir:str):
    betas = readinfo(beta_dict_path)
    se_err_list = []
    deg_err_list = []
    degree_list = list(betas.keys())
    regex = r"(\d+)_\d+"
    degree_list = []
    for k,v in betas.items():
        degree = int(re.match(regex, k).group(1)) 
        degree_list.append(degree)
        v = v[0]
        se_err = abs(v["nc_se"] - v["nc"])
        deg_err = abs(v["nc_deg"] - v["nc"])
        se_err_list.append(se_err)
        deg_err_list.append(deg_err)
    plt.plot(degree_list, se_err_list, label="se_err")
    plt.plot(degree_list, deg_err_list, label="deg_err")
    plt.legend()
    fig_path = os.path.join(save_dir, "err.pdf")
    plt.savefig(fig_path)
    plt.clf()