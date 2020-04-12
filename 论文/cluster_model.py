from gensim.models import word2vec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False


path_neg='C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/cut_pda_neg.txt'
path_pos='C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/cut_pda_pos.txt'
all_text=[]
with open(path_neg,'r',encoding='utf-8') as f:
    for line in f.readlines():
        lines=line.lstrip('\ufeff').rstrip('\n').split(' ')
        #print(lines)
        all_text.append(lines)
with open(path_pos, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
        all_text.append(lines)
model=word2vec.Word2Vec(all_text,size=300,window=5,min_count=3)

vocabulary_list=[]
word_list=model.wv.index2word
for word in word_list:
    vocabulary_list.append(model[word].tolist())
data=pd.DataFrame(vocabulary_list)
a=data[134].tolist()
sns.distplot(a)
plt.show()
clf=PCA(n_components=2)
clf.fit(data)
newX=clf.fit_transform(data)
newX=pd.DataFrame(newX)
print(newX.head())
sns.distplot(newX[0])
sns.distplot(newX[1])
c=['newx_0','newx_1']
plt.legend(c)
plt.show()


plt.scatter(data[0],data[1])

gmm=mixture.GaussianMixture(n_components=2)
gmm.fit(data)
pred_gmm=gmm.predict(data)
plt.scatter(newX[0],newX[1],c=pred_gmm)
plt.show()

kmean=KMeans(n_clusters=2)
kmean.fit(data)
pre_kmean=kmean.predict(data)
plt.scatter(data[45],data[187],c=pre_kmean)
plt.show()


meanshift=MeanShift(bandwidth=7)
meanshift.fit(data)
pred_ms=meanshift.labels_
plt.scatter(data[0],data[1],c=pred_ms,cmap='prism')
plt.title('meanshift')
plt.show()


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False
path='C:/Users/Administrator/Desktop/data/情感词汇本体/情感词汇本体.csv'
data=pd.read_csv(path,engine='python')
data['情感分类']=data['情感分类'].str.strip()
d=data.groupby('情感分类').count()
pp=d['词语'].values.tolist()
po=d.index.values.tolist()
ui=pd.DataFrame()
ui['count']=pp
ui['clarify']=po
ui['大分类']=ui['clarify'].map({'PA':'乐','PE':'乐','PD':'好','PH':'好','PG':'好',
                        'PB':'好','PK':'好',
                        'NA':'怒','NB':'哀','NJ':'哀','NH':'哀','PF':'哀',
                        'NI':'惧','NC':'惧','NG':'惧',
                        'NE':'恶','ND':'恶','NN':'恶','NK':'恶','NL':'恶','PC':'惊'})
sns.barplot(x='clarify',y='count',hue='大分类',data=ui)
plt.legend(loc=2)
plt.savefig('C:/Users/Administrator/Desktop/论文/picture/情感分类.png')
plt.show()

for rowindex,data in ui.iterrows():
    if data['大分类'] not in af:
        af[data['大分类']] = data['count']
    else:
        af[data['大分类']] += data['count']

fg=pd.DataFrame()
fg['大分类']=list(af.keys())
fg['count']=list(af.values())

sns.barplot(x='大分类',y='count',data=fg)
plt.xlabel('情感词汇本体库情感分类')
plt.ylabel('count')
plt.savefig('C:/Users/Administrator/Desktop/论文/picture/情感分类_大.png')
plt.show()



path0='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/程度级别词语（中文）.txt'
def getdata(path):
    data=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            print(line[0])
            if line=='\n' or line[0].isnumeric()  :
                continue
            else:
                data.append(line)
        f.close()
    return data
da1=getdata(path0)
print(len(da1))