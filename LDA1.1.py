from gensim.models import LdaModel
import pandas as pd
from gensim.corpora import Dictionary
from gensim import corpora, models
import csv

# 准备数据
PATH = "C:/Users/ASUS/Desktop/数学建模/数据/MU5735/output.txt"

file_object2 = open(PATH, encoding='utf-8', errors='ignore').read().split('\n')  # 一行行的读取内容
data_set = []  # 建立存储分词的列表
for i in range(len(file_object2)):
    result = []
    seg_list = file_object2[i].split()
    for w in seg_list:  # 读取每一行分词
        result.append(w)
    data_set.append(result)

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

import pyLDAvis.gensim_models as gensims
#pyLDAvis.enable_notebook()
data = gensims.prepare(lda, corpus, dictionary)
import pyLDAvis
pyLDAvis.save_html(data, 'C:/Users/ASUS/Desktop/数学建模/数据/MU5735/topic.html')