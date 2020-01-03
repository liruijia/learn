import sys
sys.path.append('G:/anconada/envs/py36/lib/site-packages')
from gensim.models import word2vec
from prettytable import PrettyTable
path='C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
sentences=word2vec.Text8Corpus(path)
model=word2vec.Word2Vec(sentences,size=500,window=6,min_count=5)

word_list=['好评','oppo','差评','质量']

table=PrettyTable()
for word in word_list:
    sim=model.most_similar(word,topn=10)
    table.add_column(word,sim)
print(table)
