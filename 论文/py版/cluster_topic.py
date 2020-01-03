#对语料库进行聚类分析
#利用Kmeans进行分析


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import matplotlib.pyplot as plt
from gensim.models import word2vec
import  pandas as pd



path='C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'

corpus=[]
with open(path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        lines = line.lstrip('\ufeff').rstrip('\n')
        corpus.append(lines)

#对文档进行向量化
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
weight_doc=tfidf.toarray()

word_list = vectorizer.get_feature_names()
with open('C:/Users/Administrator/Desktop/data/评论/word_list.txt','w',encoding='utf-8') as f:
    for word in  word_list:
        f.write(word+' ')
    f.write('\r\n\r\n')
    f.close()
 
#对token进行向量化

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





