'''
输入：语料中的评论
      情感主题模型得到的每一个词汇的情感标签以及强度
      以及关于语料库的情感词典，需要利用该词的情感强度这一个属性
      以及使用tf-idf权重矩阵，利用该矩阵得到的关于每一个词在每一个评论中的重要性
输出：输出每一个句子的情感极性得分以及情感情绪得分

:param sentiment_word: 利用情感主题混合模型得到的最后的情感-词汇矩阵，标记了每一个评论中每一个词汇的情感标签
:param sentiment_corpus: 关于语料库的情感词典
:param weight:利用tfidf计算得来的每个词汇在所有文档中的权重


 '''




from gensim import models,corpora
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd

path_corpus='C:/Users/Administrator/Desktop/data/corpus/handle_corpus_train.txt'
co=open(path_corpus,encoding='utf-8').read()
co=co.lstrip('\ufeff')
data=json.loads(co)
ui=data['正面']+data['反面']
corpus0=[]
for doc in ui:
    if len(doc)!=0:
        corpus0.append(doc)


sentence=set()
for doc in corpus0:
    if len(doc) != 0:
        sentence.add(' '.join(doc))
sentence=list(sentence)
id2word=corpora.Dictionary(corpus0)
word2id=id2word.token2id
corpus = [id2word.doc2bow(text) for text in corpus0]
model=models.TfidfModel(corpus,id2word=id2word)
weight_corpus=model[corpus]
#使用迭代的方式可以得到语料中每一则评论中每一个词汇的概率



def get_sentiment_score(pinglun,pinglun_index):
    '''
    :param sentiment_word: 利用情感主题混合模型得到的最后的情感-词汇矩阵，标记了每一个评论中每一个词汇的情感标签
    :param sentiment_corpus: 关于语料库的情感词典
    :param weight:利用tfidf计算得来的每个词汇在所有文档中的权重
    :return: 返回任意一则评论的情感得分
    '''
    weight=[]
    jx_list=[]
    wordid_list=[]
    for word in pinglun :
        id=word2id[word]
        cx=setiment_dict[word][1]
        jx,par_jx=word_sentiment_vocabulary[word]
        if cx =='d':
            weiht.append(4)
        else:

            weight.append(1)
        jx_list.append(jx)
        wordid_list.append(id)
    score=sum([weight[i]*jx_list[i]*weight_corpus[pinglun_index][wordid_list[i]][-1]  for i in rane(len(pinglun))])
    return score
#找出来一部分的数据——找出其中多个店铺的评论来进行情感值得计算  此时主要是针对水杯数据 利用referenceid来进行店铺的辨别
#每一个店铺有多个商品，因此要找到每一个店铺所包含的商品id
path_info='C:/Users/Administrator/Desktop/data/评论/product_info_cup_before.csv'
info_dianpu=pd.read_csv(path_info,engine='python',encoding='utf-8')

ii=info_dianpu['shop_id'].value_counts()

dianpu_info={}
count_ii=ii.head(n=8)
for i in range(len(count_ii)):
    product_id=info_dianpu[info_dianpu['shop_id']==count_ii.index[i]]['product_id'].values.tolist()
    shop_name=info_dianpu[info_dianpu['shop_id']==count_ii.index[i]]['shop_name'].unique()[0]
    if shop_name not in dianpu_info.keys():
        dianpu_info[shop_name]=[]
    dianpu_info[shop_name].extend(product_id)

path='C:/Users/Administrator/Desktop/data/评论/comment_info_cup_final.csv'
df_data=pd.read_csv(path,engine='python')
oo=df_data['referenceId'].value_counts()
pp=oo[oo>=300].index.tolist()
corpus_dianpu={}
for dianpu,product_id_list in dianpu_info.items():
    for product_id in product_id_list:
        if product_id in pp:
            ui = df_data[df_data['referenceId'] == product_id]['comment'].values.tolist()
        else:
            continue
        if dianpu not in corpus_dianpu.keys():
            corpus_dianpu[dianpu]=[]
        corpus_dianpu[dianpu].append(ui)











