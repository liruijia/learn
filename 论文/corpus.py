import pandas as pd
import random
import collections
from gensim.models  import word2vec
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import  os
import re
from zhon.hanzi import  punctuation
from sklearn.model_selection import train_test_split
import jieba.posseg as peg
path1='C:/Users/Administrator/Desktop/data/中文情感分析语料库/'
lj1=os.listdir(path1)
all_path=[]
for line in lj1:
    op=os.listdir(path1+line)
    for ming in op:
        all_path.append(path1+line+'/'+ming)
stop_path='C:/Users/Administrator/Desktop/data/评论/stop_words.txt'
stop_words=[]
stop_words.append('外形外观')
stop_words.append('屏幕音效')
stop_words.append(['拍照效果','运行速度','待机时间','其他特色'])
with open(stop_path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        lines=line.strip('\n').split(' ')
        stop_words.extend(lines)


def get_corpus(comment_iu):
    #存放了词汇以及词性标注
    corpus=[]
    for c in comment_iu:
        new_c=re.sub(r'[%s,\t,\\]+'%punctuation,' ',c)
        cut_c=peg.cut(new_c)
        new_doc=[]
        for word,flag in cut_c:
            #print(word,word.isalpha())
            if word not in stop_words:
                if word.isalpha() is True :
                    new_doc.append(word+'_'+flag)
                    #print(word)
        corpus.append(new_doc)
    return corpus

def getdata0(path):
    data=[]
    with open(path, 'r',encoding ='utf-8') as f:
        for line in f.readlines():
            if line[0].isdigit():
                continue
            else:
                if line != '\n':
                    lines=line.strip().split(' ')
                    data.extend(lines)
        f.close()
    return data

all_comment_pos=[]
all_comment_neg=[]
all_path_pos=[]
all_path_neg=[]
for path in all_path:
    if path[-7:]=='neg.txt':
        all_path_neg.append(path)
    else:
        all_path_pos.append(path)


for path in all_path_pos:
    data=getdata0(path)
    all_comment_pos.extend(data)
for path in all_path_neg:
    data=getdata0(path)
    all_comment_neg.extend(data)

all_corpus_neg=get_corpus(all_comment_neg)
all_corpus_pos=get_corpus(all_comment_pos)


#对于爬取到的文本进行整理

#京东语料
path='C:/Users/Administrator/Desktop/data/评论/comment_info_final.csv'
data=pd.read_csv(path,engine='python')
data['评价']=data['score'].map({5:'好评',4:'好评',3:'差评',2:'差评',1:'差评'})
dacou1=data[data['评价']=='差评']
dacou2=data[data['评价']=='好评']
daco1=dacou1['comment']
daco2=dacou2['comment']

random_index=random.sample(list(range(0,len(dacou2))),4*len(daco1))
daco2_1=dacou2.loc[random_index,'comment']

def get_comment_lx(dacop):
    comment=[]
    for ui in dacop:
        #print(ui)
        try:
            sp=ui.split('\n')
            lk=[]
            for oo in sp:
                kl=oo.split('：')
                lk.append(kl[-1])
            lk=','.join(lk)
            comment.append(lk)
        except Exception as result:
            print(result)
    return comment
comment_neg_oppo=get_comment_lx(daco1)
comment_pos_oppo=get_comment_lx(daco2_1)
comment3=get_comment_lx(daco2)
print(len(comment_neg_oppo),len(comment_pos_oppo))
corpus_neg_oppo=get_corpus(comment_neg_oppo)
corpus_pos_oppo=get_corpus(comment_pos_oppo)

#小米手机语料库信息整理

path='C:/Users/Administrator/Desktop/data/评论/comment_info_xiaomi_final.csv'
xiaomi=pd.read_csv(path,engine='python',encoding='utf-8')
xiaomi['pingjia']=xiaomi['score'].map({5:'好评',4:'好评',3:'差评',2:'差评',1:'差评'})
daxc1=xiaomi[xiaomi['pingjia']=='好评']
daxc2=xiaomi[xiaomi['pingjia']=='差评']
print('好评:{0},差评:{1}'.format(len(daxc1),len(daxc2))) #相差不是很多

comment_pos_xiaomi=daxc1['comment']
comment_neg_xiaomi=daxc2['comment']
corpus_neg_xiaomi=get_corpus(comment_neg_xiaomi)
corpus_pos_xiaomi=get_corpus(comment_pos_xiaomi)


#需要另存一份，作为训练集
with open('C:/Users/Administrator/Desktop/data/corpus/train_pos_xiaomi_corpus.txt','w',encoding='utf-8') as f:
    for doc in corpus_pos_xiaomi:
        f.write(' '.join(doc))
        f.write('\n')
    f.close()
with open('C:/Users/Administrator/Desktop/data/corpus/train_neg_xiaomi_corpus.txt','w',encoding='utf-8') as f:
    for doc in corpus_neg_xiaomi:
        f.write(' '.join(doc))
        f.write('\n')
    f.close()




#水杯语料库数据
path='C:/Users/Administrator/Desktop/data/评论/comment_info_cup_final.csv'
cup=pd.read_csv(path,engine='python')
cup['pingjia']=cup['score'].map({5:'好评',4:'好评',3:'差评',2:'差评',1:'差评'})
cup1=cup[cup['pingjia']=='好评']
cup2=cup[cup['pingjia']=='差评']
print(len(cup1),len(cup2))
comment_pos0=cup1['comment']
comment_neg0=cup2['comment']
comment_pos_cup=get_comment_lx(comment_pos0)
comment_neg_cup=get_comment_lx(comment_neg0)
corpus_neg_cup=get_corpus(comment_neg_cup)
corpus_pos_cup=get_corpus(comment_pos_cup)

with open('C:/Users/Administrator/Desktop/data/corpus/train_pos_cup_corpus.txt','w',encoding='utf-8') as f:
    for doc in corpus_pos_cup:
        if len(doc)!=0:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()
with open('C:/Users/Administrator/Desktop/data/corpus/train_neg_cup_corpus.txt','w',encoding='utf-8') as f:
    for doc in corpus_neg_cup:
        if len(doc)!=0:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()





#整合

total_corpus_neg=corpus_neg_xiaomi+corpus_neg_oppo+corpus_neg_cup+all_corpus_neg
total_corpus_pos=corpus_pos_xiaomi+corpus_pos_oppo+corpus_pos_cup+all_corpus_pos

train_pos_total_corpus,test_pos_total_corpus=train_test_split(total_corpus_pos,test_size=0.3)
train_neg_total_corpus,test_neg_total_corpus=train_test_split(total_corpus_neg,test_size=0.3)


with open('C:/Users/Administrator/Desktop/data/corpus/train_total_cut_neg.txt','w',encoding='utf-8') as f:
    for doc in train_neg_total_corpus:
        if len(doc) !=0:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()


with open('C:/Users/Administrator/Desktop/data/corpus/train_total_cut_pos.txt','w',encoding='utf-8') as f:
    for doc in train_pos_total_corpus:
        if len(doc)!=0:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()
with open('C:/Users/Administrator/Desktop/data/corpus/test_total_cut_neg.txt', 'w', encoding='utf-8') as f:
    for doc in test_neg_total_corpus:
        if len(doc) != 0:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()

with open('C:/Users/Administrator/Desktop/data/corpus/test_total_cut_pos.txt', 'w', encoding='utf-8') as f:
    for doc in test_pos_total_corpus:
        if len(doc) != 0:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()

print()
