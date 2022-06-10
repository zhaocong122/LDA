import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings

warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

if __name__ == '__main__':
    PATH = "C:/Users/ASUS/Desktop/数学建模/数据/MU5735/output.txt"

    file_object2 = open(PATH, encoding='utf-8', errors='ignore').read().split('\n')  # 一行行的读取内容
    data_set = []  # 建立存储分词的列表
    for i in range(len(file_object2)):
        result = []
        seg_list = file_object2[i].split()
        for w in seg_list:  # 读取每一行分词
            result.append(w)
        data_set.append(result)
    print(data_set)

    dictionary = corpora.Dictionary(data_set)  # 构建词典
    corpus = [dictionary.doc2bow(text) for text in data_set]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=30, random_state=1)
    topic_list = lda.print_topics()
    print(topic_list)

    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        for j in i:
            listj.append(j[1])
        bz = listj.index(max(listj))
        print(i[bz][0])

    #计算困惑度
    def perplexity(num_topics):
        ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
        print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
        print(ldamodel.log_perplexity(corpus))
        return ldamodel.log_perplexity(corpus)
    #计算coherence
    def coherence(num_topics):
        ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
        print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
        ldacm = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='u_mass')
        print(ldacm.get_coherence())
        return ldacm.get_coherence()


    # 绘制困惑度折线图

    # z = [perplexity(i) for i in x]

    x = range(1, 15)
    # y = [coherence(i) for i in x]
    y = [perplexity(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('主题数目')
    plt.ylabel('perplexity大小')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('主题-perplexity变化情况')
    plt.show()






# 绘制困惑度折线图
# x = range(1,15)
# z = [perplexity(i) for i in x]
# plt.plot(x, y)
# plt.xlabel('主题数目')
# plt.ylabel('困惑度大小')
# plt.rcParams['font.sans-serif']=['SimHei']
# matplotlib.rcParams['axes.unicode_minus']=False
# plt.title('主题-困惑度变化情况')
# plt.show()