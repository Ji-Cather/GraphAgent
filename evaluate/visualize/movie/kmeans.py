import gensim
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.cluster import KMeans

from typing import List


def plt_topic_png(documents:List[str]=[]):
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # 假设您的论文数据是一个列表，文章（documents）包含在列表
    # documents = [
    # "Tech companies are investing heavily in artificial intelligence and machine learning.",
    # "Sports events are closed to the public this year due to health concerns.",
    # "The art gallery features contemporary paintings from various artists around the world.",
    # "Economic indicators suggest a rebound in stock markets after the initial downturn.",
    # "Regular exercise contributes to overall health and well-being.",
    # "Advancements in computer vision are revolutionizing tech industries.",
    # "The championship game will be played without fans in attendance.",
    # "Sculpture in the 21st century has embraced mixed media and digital technologies.",
    # "Interest rates are expected to rise as the economy recovers from the pandemic shock.",
    # "A balanced diet and good hydration can improve physical health and energy levels."]

    # 数据预处理
    stop_words = stopwords.words('english')
    texts = [[word for word in word_tokenize(document.lower()) if word.isalpha() and word not in stop_words]
            for document in documents]

    # 创建字典和语料库
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 创建LDA模型
    num_topics = 5  # 假设我们想提取5个主题，这可以根据您的需要更改
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=5, workers=2)

    # 打印主题
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # 为每个文档分配主题
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list            
        print(f"Document {i}: ", end='')
        for (topic_num, prob) in row:
            print(f"Topic {topic_num}: {prob:.3f}", end=', ')
        print()


    # 假设lda_model和corpus已经根据前一步骤准备好
    # 获取每个文档的主题分布
    topic_distributions = np.array([[prob for _, prob in lda_model.get_document_topics(bow)] 
                                    for bow in corpus])

    # 使用K-means进行聚类，n_clusters为期望的聚类数
    n_clusters = 3  # 根据您的数据集大小和多样性调整
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(topic_distributions)

    cluster_keywords = []
    # 打印每个聚类的中心点，并尝试解读它代表的主题
    for i, centroid in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i}:")
        # 选出每个聚类中心点中权重最高的3个主题
        top_topics = centroid.argsort()[-3:][::-1]
        for topic_idx in top_topics:
            # 打印每个主题的权重和对应的关键词
            print(f" Top Topic {topic_idx + 1}: {centroid[topic_idx]:.4f} --", " ".join([word for word, _ in lda_model.show_topic(topic_idx, topn=5)]))
        print("\n")
        cluster_keywords.append(' '.join(word for word, _ in lda_model.show_topic(top_topics[0], topn=3)))

    plot_kmeans(topic_distributions,
                kmeans,
                cluster_keywords)

def plot_kmeans(topic_distributions,
                kmeans,
                cluster_keywords):
    import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 
font_path = 'test/Times_New_Roman/TimesNewerRoman-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    from sklearn.decomposition import PCA

    # 使用PCA进行降维，减少到2维
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(topic_distributions)

    # 使用reduced_data和kmeans.labels_绘制聚类结果
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(set(kmeans.labels_)):
        # 绘制每个聚类的数据点
        plt.scatter(reduced_data[kmeans.labels_ == label, 0], 
                    reduced_data[kmeans.labels_ == label, 1], 
                    label = cluster_keywords[i],
                    alpha=0.5, 
                    edgecolors='w')

    # 在每个聚类上标记中心点
    centroids = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='black', label='Centroids')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clustering with PCA-reduced LDA Topics')
    plt.legend()
    plt.show()
    plt.savefig("kmeans_cluster.pdf")


if __name__ =="__main__":
    
    plt_topic_png("")