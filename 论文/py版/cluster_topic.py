#对语料库进行聚类分析
#利用Kmeans进行分析

import sys
sys.path.append('G:/anconada/envs/py36/lib/site-packages')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import matplotlib.pyplot as plt
from gensim.models import word2vec
import  pandas as pd
import numpy as np
from prettytable import PrettyTable
import copy



path='C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
path_stop='C:/Users/Administrator/Desktop/data/评论/stop_words.txt'
stpwrd_dic = open(path_stop, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

print('总共有{0}个停用词'.format(len(stpwrdlst)))
corpus=[]
with open(path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        lines = line.lstrip('\ufeff').rstrip('\n')
        corpus.append(lines)

#对文档进行向量化
vectorizer = CountVectorizer(stop_words=stpwrdlst)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
weight_doc=tfidf.toarray()

count_matrix=vectorizer.fit_transform(corpus).toarray()
word=vectorizer.get_feature_names()
print(word[:100])
oi=vectorizer.vocabulary_
table=PrettyTable(['word','tf'])
df=pd.DataFrame(columns=['word','tf'])
sum_count=np.sum(count_matrix,axis=0)
print('sum_count',sum_count.shape)
sum_c_doc=np.sum(count_matrix,axis=1)

print(sum_count.shape)
count_m=[]
tf=copy.deepcopy(count_matrix)
m,n=count_matrix.shape
print(sum_c_doc[0])

for i in range(m):
    for j in range(n):
        tf[i,j]=tf[i,j]/(sum_c_doc[i]+1)
print(tf)
for word,id in oi.items():
    count_m.append([word,sum_count[id]])

scm=list(sorted(count_m,key=lambda x:x[1],reverse=True))

for i in range(20):
    print(scm[i][0],scm[i][1])
word_list = vectorizer.get_feature_names()
with open('C:/Users/Administrator/Desktop/data/评论/word_list.txt','w',encoding='utf-8') as f:
    for word in  word_list:
        f.write(word+' ')
    f.write('\r\n\r\n')
    f.close()
print(weight_doc)




#对token进行向量化
'''
model=word2vec.Word2Vec(word2vec.Text8Corpus(path),size=500,window=5,min_count=1)

vocabulary_word={}
word_list=model.wv.index2word

with open('C:/Users/Administrator/Desktop/data/评论/word_vocabulary.txt','w',encoding='utf-8') as f:
    for word in word_list:
        f.write(str(model[word]) )
        if word not in vocabulary_word.keys():
            vocabulary_word[word]=model[word]
    f.write('\r\n\r\n')
    f.close()
xl_word=list(vocabulary_word.values())

data=pd.DataFrame(xl_word)




clf=KMeans(n_clusters=20)
s=clf.fit(data)

print(clf.cluster_centers_)
#保存标签 且可视化 需要用到PCA

r1=pd.Series(clf.labels_).value_counts()
r2=pd.DataFrame(clf.cluster_centers_)
r = pd.concat([r2, r1], axis =1)

r.head()
r.columns =list(data.columns) + [u'类别数目']#重命名表头

r.head()



r_0 = pd.concat([data, pd.Series(clf.labels_, index = data.index)], axis =1)#详细输出每个样本对应的类别

r_0.columns =list(data.columns) + [u'聚类类别']#重命名表头

#r_0.to_excel(outputfile)#保存结果

def density_plot(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] =False #用来正常显示负号
    p = data.plot(kind='kde', linewidth =2, subplots =True, sharex =False)
    plt.show(p)
    return plt
for i in range(20):
    density_plot(data[r[u'聚类类别']==i])
'''




