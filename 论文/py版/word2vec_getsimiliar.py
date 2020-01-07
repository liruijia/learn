import sys
import pandas as pd
sys.path.append('G:/anconada/envs/py36/lib/site-packages')
from gensim.models import word2vec
from prettytable import PrettyTable

path='C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
#text=[]
#with open(path,'r',encoding='utf-8') as f:
#    for line in f.readlines():
#        text.append(line)
#    f.close()
#print('总共有{0}条评论'.format(len(text)))


sentences=word2vec.Text8Corpus(path)
model=word2vec.Word2Vec(sentences,size=500,window=6,min_count=5)

word_list=['好评','oppo','差评','质量']

table=PrettyTable()
path='C:/Users/Administrator/Desktop/data/评论/word_similiar.csv'
df=pd.DataFrame(columns=word_list)

for word in word_list:
    sim=model.most_similar(word,topn=50)
    df[word]=sim
    table.add_column(word,sim)
print(table)
df.to_csv(path)
