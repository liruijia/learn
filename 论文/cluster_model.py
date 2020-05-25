import sys

sys.path.append('G:/anconada/envs/py36/lib/site-packages')
from gensim.models import word2vec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import json
from prettytable import PrettyTable
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False


path_total='C:/Users/Administrator/Desktop/data/corpus/handle_corpus_train.txt'

all_text=[]
data=open(path_total,encoding='utf-8').read()
corpus=json.loads(data.lstrip('\ufeff'))
corpus_total=corpus['正面']+corpus['反面']
for doc in corpus_total:
    all_text.append(' '.join(doc))

model=word2vec.Word2Vec(corpus_total,size=200,window=5,min_count=1)

word_list=['喜欢','失望','不好','满意']
table=PrettyTable()
for word in word_list:
    table.add_column(word,model.most_similar([word]))

print(table)

