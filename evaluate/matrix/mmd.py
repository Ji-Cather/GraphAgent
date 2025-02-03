import networkx as nx
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

# from evaluate.matrix.dist_helper import compute_mmd

def calculate_mmd_matrix(graph_true:nx.Graph, 
                        graph_generated:nx.Graph,
                        graph_name,
                        indicators = [
                        "degree",
                        "clustering",
                        "eigenvalue"]
                        ) -> pd.DataFrame:
    df = pd.DataFrame()
    # map_func = {
    #     "degree": calculate_mmd_degree,
    #     "clustering": calculate_mmd_clustering,
    #     "eigenvalue": calculate_mmd_eigenvalue
    # }

    map_func = {
        "degree": calculate_mmd_degree,
        "clustering": calculate_mmd_clustering,
        "eigenvalue": calculate_mmd_eigenvalue
    }

    for indicator, cal_func in map_func.items():
        if indicator not in indicators:
            continue
        try:
            df.loc[graph_name,"mmd_"+ indicator] = cal_func(graph_true, graph_generated)
        except:
            df.loc[graph_name,"mmd_"+ indicator] = np.nan
    return df


def calculate_mmd_degree(graph_true:nx.Graph, 
                         graph_generated:nx.Graph):
    
    
    degrees_G1 = np.array([d for n, d in graph_true.degree()]).reshape(-1, 1)  # 转换为2D数组以符合函数输入要求
    degrees_G2 = np.array([d for n, d in graph_generated.degree()]).reshape(-1, 1)

    # 使用RBF核计算MMD
    sigma = 1  # 核宽度参数，该值较大时模型变得更平滑，但太大可能导致特征不明显，需要根据数据调整
    mmd_value = compute_mmd_rbf(degrees_G1, degrees_G2, sigma=sigma)
    return mmd_value

def calculate_mmd_clustering(graph_true:nx.Graph, 
                         graph_generated:nx.Graph):
    # 计算局部聚类系数
    clustering_coeffs_G1 = np.array(list(nx.clustering(graph_true).values())).reshape(-1, 1)
    clustering_coeffs_G2 = np.array(list(nx.clustering(graph_generated).values())).reshape(-1, 1)

    # 使用RBF核计算MMD
    sigma = 1  # 核宽度参数，该值较大时模型变得更平滑，但太大可能导致特征不明显，需要根据数据调整
    mmd_value = compute_mmd_rbf(clustering_coeffs_G1, clustering_coeffs_G2, sigma=sigma)
    return mmd_value

def calculate_mmd_eigenvalue(graph_true:nx.Graph, 
                         graph_generated:nx.Graph):
    if isinstance(graph_true, nx.DiGraph):
        graph_true = graph_true.to_undirected()
    if isinstance(graph_generated, nx.DiGraph):
        graph_generated = graph_generated.to_undirected()
    # 计算归一化拉普拉斯矩阵
    L1_normalized = nx.normalized_laplacian_matrix(graph_true)
    L2_normalized = nx.normalized_laplacian_matrix(graph_generated)

    # 计算特征值，注意scipy中eigs函数要求输入为稀疏矩阵格式
    eigenvals_G1, _ = eigs(csr_matrix(L1_normalized), k=20, which='SM')  # k小于N-1
    eigenvals_G2, _ = eigs(csr_matrix(L2_normalized), k=20, which='SM')
    # eigenvals_G1, _ = eig(L1_normalized)  # k小于N-1
    # eigenvals_G2, _ = eig(L2_normalized)

    eigenvals_G1 = np.abs(eigenvals_G1.real).reshape(-1, 1)  # 取实部和绝对值处理复数
    eigenvals_G2 = np.abs(eigenvals_G2.real).reshape(-1, 1)

    # 使用RBF核计算MMD
    sigma = 1  # 核宽度参数，该值较大时模型变得更平滑，但太大可能导致特征不明显，需要根据数据调整
    mmd_value = compute_mmd_rbf(eigenvals_G1, eigenvals_G2, sigma=sigma)
    return mmd_value



def rbf_kernel(X, Y, sigma=1):
    """
    计算RBF核矩阵
    """
    X_sqnorms = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_sqnorms = np.sum(Y**2, axis=1).reshape(1, -1)
    
    # 计算样本间的欧氏距离的平方（||x-y||^2项）
    distances_sq = X_sqnorms + Y_sqnorms - 2 * np.dot(X, Y.T)
    
    # 应用RBF核公式
    K = np.exp(-distances_sq / (2 * sigma**2))
    return K

def compute_mmd_rbf(X, Y, sigma=1):
    """
    使用RBF核来计算两个样本集间的MMD
    """
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    
    mmd_square = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return np.sqrt(mmd_square)


