import matplotlib.pyplot as plt

import networkx as nx

import pandas as pd
import os


def plot_shrinking_diameter(matrix_dir:str,
                            save_dir:str,
                            graph_name:str):
    matrix_names = ["action",
                    "follow",
                    "friend"]
    plt.figure(figsize=(6, 4))    
    lable_map ={
        "action":"Action Net.",
        "follow":"Follow Net.",
        "friend":"Friend Net."
    }

    for matrix_name in matrix_names:
        matrix_path = os.path.join(matrix_dir, f"{matrix_name}_matrix.csv")

        matrix = pd.read_csv(matrix_path,index_col = 0)
        dates = []
        for index, row in matrix.iterrows():
            date = index.split("_")[0]
            dates.append(date)
        diameters = matrix["effective diameter"].to_list()

        # 绘制曲线图
        plt.plot(dates, diameters, marker='o', linestyle='-', label=lable_map[matrix_name])

    save_path = os.path.join(save_dir, 
                                f"{graph_name}_shrinking_diameter.pdf")
    # plt.title('Shrinking Diameter Over Time',fontsize=20)
    plt.ylabel('$D_e$', fontsize=20)
    plt.legend(fontsize=14,loc = "upper left")  # 显示图例
    plt.xticks([],fontsize=14)
    plt.xlabel('Rounds', fontsize=20)
    # plt.xticks(fontsize=14)  # 设置x轴刻度的字体大小
    plt.yticks(fontsize=18)  # 设置x轴刻度的字体大小
    plt.grid(True)

    # 选择要显示的日期值
    date_values = [] 
    thunk_size = len(dates)//5
    thunk_size = 1 if thunk_size==0 else thunk_size
    for idx in range(0,len(dates),thunk_size):
        date_values.append(dates[idx])
    plt.yticks([2.5,5,7.5,10,15,17.5,20])
    # plt.xticks(date_values, rotation=30,fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

