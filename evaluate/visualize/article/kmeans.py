import gensim
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from typing import List
from scipy.optimize import linear_sum_assignment

from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from scipy.spatial.distance import cosine
import os

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_sentence(sentence):
    # 使用tokenizer对句子进行编码，并返回Tensor
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # 从模型输出中提取最后一层的池化向量（默认是[cls]位置的向量）
    last_hidden_state = outputs.last_hidden_state
    # 返回池化后的嵌入向量
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return sentence_embedding

def calculate_similarity(sentence1, sentence2):
    # 对两个句子进行编码
    embedding1 = encode_sentence(sentence1)
    embedding2 = encode_sentence(sentence2)

    # 计算余弦相似度
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def match_list_strs(lda_topics, topics) -> dict:
    # 初始化相似度矩阵
    similarity_matrix = np.zeros((len(lda_topics), len(topics)))
    
    # 计算相似度并存储原始相似度以备后续查找次优解
    original_similarities = {}
    for i, topic1 in enumerate(lda_topics):
        for j, topic2 in enumerate(topics):
            similarity = calculate_similarity(topic1, topic2)
            similarity_matrix[i, j] = similarity
            if i not in original_similarities.keys():
                original_similarities[i] = []
            original_similarities[i].append((j, similarity))
    
    # 使用匈牙利算法寻找初始匹配
    row_ind, col_ind = linear_sum_assignment(similarity_matrix)
    
    # 输出匹配结果，处理重复值
    matches = {}
    for i in range(len(row_ind)):
        matches[i] = topics[col_ind[i]]
       
    
    return matches

def pad_list(list_of_lists):
    # 找到最长的子列表长度
    max_length = max(len(lst) for lst in list_of_lists)

    # 初始化一个形状为(n子列表, max_length)的数组，并填充0
    array_padded = np.zeros((len(list_of_lists), max_length))

    # 使用循环来填充非零值
    for i, lst in enumerate(list_of_lists):
        array_padded[i, :len(lst)] = lst
    return array_padded

def plt_topic_png(save_path:str,
                  documents:List[str]=[]
                  ):
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
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=5, workers=2)

    # 打印主题
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # 为每个文档分配主题
    topic_distributions = []

    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list            
        print(f"Document {i}: ", end='')
        topic_probs =[0 for i  in range(num_topics)]
        for (topic_num, prob) in row:
            print(f"Topic {topic_num}: {prob:.3f}", end=', ')
            topic_probs[topic_num] = prob
        topic_distributions.append(topic_probs)
        


    # 假设lda_model和corpus已经根据前一步骤准备好
    # 获取每个文档的主题分布
    # topic_distributions = [[prob for _, prob in lda_model.get_document_topics(bow)] 
    #                                 for bow in corpus]
    
    # 使用K-means进行聚类，n_clusters为期望的聚类数
    topic_distributions = np.array(topic_distributions)
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

    plot_kmeans(save_path,
                topic_distributions,
                kmeans,
                cluster_keywords)
    
def get_topics_available(dataset):
        if "llm_agent" in dataset:
            return [
                "Artificial Intelligence",
                "Machine Learning",
                "Computational Sciences",
                "Social and Cognitive Sciences",
                "Software and Systems",
                "Emerging Technologies"
            ]
        elif "citeseer" in dataset:
            return ["Agents","AI","DB","IR","ML","HCI"]
        elif "cora" in dataset:
            return ["Case Based",
                    "Genetic Algorithms",
                    "Neural Networks",
                    "Probabilistic Methods",
                    "Reinforcement Learning",
                    "Rule Learning",
                    "Theory"]
        else:
            return []

def plt_topic_given(save_dir:str,
                  task_name:str = "citeseer",
                  article_meta_data:dict = {}
                   # 根据您的数据集大小和多样性调整
                  ):
    
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders.text import TextLoader
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from LLMGraph.loader.article import DirectoryArticleLoader
    from langchain_core.prompts import PromptTemplate

    text_loader_kwargs={'autodetect_encoding': True}
    article_loader = DirectoryArticleLoader(
                         article_meta_data = article_meta_data,
                         path = "", 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
    docs = article_loader.load()
    prompt_template = PromptTemplate.from_template("""
Title: {title}
Cited: {cited}
Publish Time: {time}
Content: {page_content}""")
    documents = [prompt_template.format(
                                        title = doc.metadata["title"],
                                        cited = article_meta_data[doc.metadata["title"]]["cited"],
                                        time = doc.metadata.get("time",""),
                                       page_content = doc.page_content) 
                                       for doc in docs]

    save_path = os.path.join(save_dir,f"{task_name}_tsne_{len(docs)}.pdf")
    topics = get_topics_available(task_name)
    num_topics = len(topics)
    n_clusters = num_topics

    stop_words = stopwords.words('english')
    texts = [[word for word in word_tokenize(document.lower()) if word.isalpha() and word not in stop_words]
            for document in documents]

    # 创建字典和语料库
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 创建LDA模型
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=5, workers=2)

    # 打印主题
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # 为每个文档分配主题
    topic_distributions = []

    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list            
        print(f"Document {i}: ", end='')
        topic_probs =[0 for i  in range(num_topics)]
        for (topic_num, prob) in row:
            print(f"Topic {topic_num}: {prob:.3f}", end=', ')
            topic_probs[topic_num] = prob
        topic_distributions.append(topic_probs)
        
    # 使用K-means进行聚类，n_clusters为期望的聚类数
    topic_distributions = np.array(topic_distributions)
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
    
    import seaborn as sns
    colors = sns.color_palette("muted", len(topics))
    # colors_map_a = sns.color_palette("rocket", as_cmap=True)
    # colors_map_a = sns.color_palette('viridis', as_cmap=True)
    # colors_map_a = sns.color_palette("Paired",as_cmap=True)
    # # colors_map_a = sns.color_palette("hls", 8, as_cmap=True)
    # colors = colors(np.linspace(0, 1, len(topics)) )
    colors_map = dict(zip(topics, colors))

    assert len(cluster_keywords) == len(topics)

    matched_topics = match_list_strs(cluster_keywords,topics)
    topics_transfered = [matched_topics[idx] for idx, key in enumerate(cluster_keywords)]
    plot_kmeans(save_path,
                topic_distributions,
                kmeans,
                topics_transfered,
                [colors_map[key] for key in topics_transfered])

def plot_kmeans(save_path,
                topic_distributions,
                kmeans,
                cluster_keywords,
                colors):
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
    # pca = PCA(n_components=2, random_state=42)
    # reduced_data = pca.fit_transform(topic_distributions)

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    reduced_data = tsne_model.fit_transform(topic_distributions)

    # 使用reduced_data和kmeans.labels_绘制聚类结果
    fig = plt.figure(figsize=(10, 9))
    ax = plt.gca()
    for i, label in enumerate(set(kmeans.labels_)):
        # 绘制每个聚类的数据点
        ax.scatter(reduced_data[kmeans.labels_ == label, 0], 
                    reduced_data[kmeans.labels_ == label, 1], 
                    label = cluster_keywords[i],
                    color = colors[i],
                    alpha=0.5, 
                    edgecolors='w')

    # 在每个聚类上标记中心点
    # centroids = pca.transform(kmeans.cluster_centers_)
    # [(reduced_data[kmeans.labels_ == i, 0], reduced_data[kmeans.labels_ == i, 1])
    #  for i in range(n_cl)]
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='black', label='Centroids')

    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # 设置坐标轴标签与字体大小
    ax.set_xlabel('t-SNE feature 1', fontsize=22)
    ax.set_ylabel('t-SNE feature 2', fontsize=22)

    # 设置图例字体大小
    # plt.legend(fontsize=22)
    # ax.legend(fontsize=22, loc='upper center', ncols = 2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(labels, loc='lower center', ncols = 2 ,fontsize =22)
    plt.subplots_adjust(bottom = 0.3)  # 右侧留出空间
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()
    plt.savefig(save_path)


if __name__ =="__main__":
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    text_loader_kwargs={'autodetect_encoding': True}
    

    article_loader = DirectoryLoader("LLMGraph/tasks/llm_agent/configs/test_config_5_article_1500/data/generated_article", 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
    generated_docs = article_loader.load()
    origin_docs = DirectoryLoader("LLMGraph/tasks/llm_agent/data/article", 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs).load()
    
    docs =[*generated_docs]
    #docs =[*origin_docs]
    documents = [doc.page_content for doc in docs][:10000]
    plt_topic_png(f"article_tsne_{len(docs)}.pdf",
                  documents,
                  40,
                  10)