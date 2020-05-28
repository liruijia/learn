
'''
参数设置：model--LDAmodel
          texts-- 语料库文本  使用进行标注设置的 ----不是必须的
          corpus--文本
          dictionary----id2word词典
          
'''
from gensim.models import word2vec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import pandas as pd
from gensim import corpora,models
from sklearn.model_selection import GridSearchCV
path1='C:/Users/Administrator/Desktop/data/corpus/train_neg_cup_corpus.txt'
path2='C:/Users/Administrator/Desktop/data/corpus/train_pos_cup_corpus.txt'
corpus1=[]
with open(path1,'r',encoding='utf-8') as f:
    for line in f.readlines():
        corpus1.append(line.strip().split(' '))
    f.close()
corpus2=[]
with open(path2,'r',encoding='utf-8') as f:
    for line in f.readlines():
        corpus2.append(line.strip().split(' '))
    f.close()
corpus0=corpus1+corpus2
id2word=corpora.Dictionary(corpus0)
corpus = [id2word.doc2bow(text) for text in corpus0]
ldamodel=models.ldamodel.LdaModel(iterations=200,corpus=corpus,num_topics=20,id2word=id2word)
for i in range(20):
    print('第{0}个主题的信息****************8'.format(i))
    print(ldamodel.show_topic(topicid=i,topn=20))
    
op=ldamodel.get_topics()
print(op)
coh=models.CoherenceModel(ldamodel,corpus=corpus,dictionary=id2word,coherence='u_mass')

print(coh.get_coherence())
kmean=KMeans(n_clusters=5)
kmean.fit(op)
pre_kmean=kmean.predict(op)
data=pd.DataFrame(op)
plt.scatter(data[45],data[187],c=pre_kmean)
plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

Z = linkage(data, 'ward')
f = fcluster(Z,3,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()

meanshift=MeanShift(bandwidth=7)
meanshift.fit(data)
pred_ms=meanshift.labels_
plt.scatter(data[0],data[1],c=pred_ms,cmap='prism')
plt.title('meanshift')
plt.show()




canshu={'alpha':[0.05,0.1,0.2,0.5],'beta':[0.05,0.1,0.2,0.5],'iteration':[50,200,500,700,1000],'topic_num':[5,10,15,20,30,50,70,90,100,120,150,200,220,250,270,300]}

dp=list(zip(canshu['alpha'],canshu['beta']))
score=[]
df_canshu=pd.DataFrame(columns=['alpha','beta','iteration','topic_num','score'])

i=0
j=0
for va1,va2 in dp:
    print('第{0}组的alpha以及beta'.format(j))
    for va3 in canshu['iteration']:
        for va4 in canshu['topic_num']:
            model=models.ldamodel.LdaModel(alpha=va1,eta=va2,iterations=va3,corpus=corpus,num_topics=va4,id2word=id2word)
            coh=models.CoherenceModel(model,corpus=corpus,dictionary=id2word,coherence='u_mass')
            score.append(coh.get_coherence())
            df_canshu.loc[i]=[va1,va2,va3,va4,coh.get_coherence()]
            i+=1
            print('第{0}组参数'.format(i))
            print(coh.get_coherence())
    j+=1
print(df_canshu)
df_canshu.to_csv('C:/Users/Administrator/Desktop/data/corpus/ldacanshuvalue.csv','w',encoding='utf-8')
