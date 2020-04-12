'''得到2个情感标签'''
import pandas as pd
import random
import collections
from gensim.models  import word2vec
import numpy as np
import jieba.posseg as peg
import copy
from sklearn. feature_extraction.text import CountVectorizer, TfidfTransformer
#情感倾向值计算



#加载进来hownet情感词典以及中文情感词汇本体库情感词典
path0='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面评价词语（中文）.txt'
path1='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面评价词语（中文）.txt'
path2='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面情感词语（中文）.txt'
path3='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面情感词语（中文）.txt'
path4='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/程度级别词语（中文）.txt'


path5='C:/Users/Administrator/Desktop/data/台湾大学NTUSD简体中文情感词典/ntusd-negative.txt'
path6='C:/Users/Administrator/Desktop/data/台湾大学NTUSD简体中文情感词典/ntusd-positive.txt'
###台湾大学情感词典
sentiment_taiwan_dict={}
with open(path5,'r',encoding='utf-8') as f:
    for word in f.readlines():
        words=word.lstrip('\ufeff').rstrip('\n')
        sentiment_taiwan_dict[words]=1
    f.close()
    
with open(path6,'r',encoding='utf-8') as f:
    for word in f.readlines():
        words=word.lstrip('\ufeff').rstrip('\n')
        sentiment_taiwan_dict[words]=0
    f.close()

#知网情感词典
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
    data.pop(0)
    return data
def getdata(path_list):
    type_list=['正面','负面','正面','负面']
    final_data={}
    for i in range(len(path_list)-1):
        data0=getdata0(path_list[i])
        for word in data0:
            if type_list[i]=='正面':
                jx=0
            elif type_list[i]=='反面':
                jx=1
            final_data[word]=jx
        print('{0}词语加载完毕'.format(type_list[i]))
    
    return final_data


sentiment_hownet_word=getdata([path0,path1,path2,path3])
sentiment_two=sentiment_taiwan_dict
for word,jx in sentiment_hownet_word.items():
    if word in sentiment_two.keys():
        continue
    sentiment_two[word]=jx


#sentiment_dict1=dict(sentiment_taiwan_dict,**sentiment_hownet_word)


#处理情感词汇本体情感
path5='C:/Users/Administrator/Desktop/data/情感词汇本体/情感词汇本体.csv'
data_0=pd.read_csv(path5,engine='python')
data01=data_0[data_0['极性']==1]
data02=data_0[data_0['极性']==2]
ui=pd.concat([data01,data02])
ui['极性']=ui['极性'].map({1:0,2:1})
print(ui.head())
sentiment_dalian_word=dict(zip(ui['词语'],ui['极性']))
#sentiment_dict=dict(sentiment_dalian_word,**sentiment_dict1)
sentiment_dict=sentiment_two
for word, jx in sentiment_dalian_word.items():
    if word in sentiment_dict.keys():
        continue
    sentiment_dict[word]=jx

print('总的情感词典有:',len(sentiment_dict))

#进行整合情感词典  data 字典
f=open('C:/Users/Administrator/Desktop/data/corpus/sentiment_dict_goujian2.txt','w',encoding='utf-8')
f.write(str(sentiment_dict))
f.close()


#对于语料库中的单词寻
path_neg = 'C:/Users/Administrator/Desktop/data/corpus/train_neg_cup_corpus.txt'
path_pos = 'C:/Users/Administrator/Desktop/data/corpus/train_pos_cup_corpus.txt'

print('开始加载语料')
corpus_neg=[]
corpus_pos=[]

corpus_word_list={}
print(' 加载train neg')
with open(path_neg, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
        doc = []
        #print(lines)
        for word_flag in lines:
            word, flag = word_flag.split('_')
            #print(word, flag)
            if len(word)>=2:
                if word not in corpus_word_list.keys() :
                   corpus_word_list[word] = flag
                doc.append(word)
        #print(doc)
        corpus_neg.append(doc)
    f.close()
print(' 加载train pos')
with open(path_pos, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
        doc = []
        #print(lines)
        for word_flag in lines:
            word, flag = word_flag.split('_')
            #print(word, flag)
            if len(word)>=2:
                if word not in corpus_word_list.keys() :
                    corpus_word_list[word] = flag
                doc.append(word)
        #print(doc)
        corpus_pos.append(doc)
    f.close()


train_text_neg=corpus_neg
train_text_pos=corpus_pos


all_text_neg=train_text_neg
all_text_pos=train_text_pos


fcorpus={'正面':all_text_pos,'反面':all_text_neg}


corpus0=corpus_pos+corpus_neg
corpus_total=corpus0



sentence=set()
for doc in corpus_total:
    sentence.add(' '.join(doc))

with open('C:/Users/Administrator/Desktop/data/corpus/corpus_total_1.txt','w',encoding='utf-8') as f:
    for doc in corpus_total:
        f.write(str(doc))
    f.close()

#word2vec 求词向量 以及最相近词汇
model=word2vec.Word2Vec(corpus_total,size=500,window=5,min_count=1,workers=5)
# vectorizer = CountVectorizer()
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(vectorizer.fit_transform(sentence))
# weight_doc=coo_matrix(tfidf.toarray())
# word_list_c=vectorizer.vocabulary_
# word_weight=weight_doc.T
print('开始对语料库中的词汇构建情感词典信息，同时得到处理之后的corpus')





def get_corpus_sentiment(text_neg,text_pos):
    corpus_neg=text_neg
    corpus_pos=text_pos
    fcorpus={'正面':corpus_pos,'反面':corpus_neg}

#求语料库中词汇的情感标签
#对于语料库中的词汇求情感词典，若为动词则直接删除掉，若为动名词，人名、地名则直接删除，
    def getcorpussentiment(fcorpus):
        sentiment_corpus_total={}
        corpus=copy.deepcopy(fcorpus)
        word_list = list(sentiment_dict.keys())
        cunzai_word={}
        nocunzai_word={}
        shanchu_word={'正面':[],'反面':[]}
        for typ1 ,cor in fcorpus.items():
            for index_doc,doc in enumerate(cor):
                ui=[]
                for index_word,word in enumerate(doc):
                    if word in word_list:
                        cunzai_word[word] = typ1
                        sentiment_corpus_total[word] = sentiment_dict[word]
                    else:
                        # print('no',word)
                        if  corpus_word_list[word]=='t'  or corpus_word_list[word]=='ns'\
                           or corpus_word_list[word]=='s' or corpus_word_list[word]=='b'or corpus_word_list[word]=='nt' or \
                           corpus_word_list[word]=='nr' or corpus_word_list[word]=='m':
                            ui.append((word, index_doc))
                        else:
                            nocunzai_word[word] = typ1
                shanchu_word[typ1].append(ui)

        for ty1,da in shanchu_word.items():
            for doc in da:
                for word ,index_doc in doc:
                    # print('del word')
                   # print(doc,word)
                    # print(corpus[ty1][index_doc])
                    corpus[ty1][index_doc].remove(word)
        print(len(sentiment_corpus_total))

        #print('删除之后corpus')
        #print(list(nocunzai_word.keys())[:10])
        print('删除v、nr、ns之后在corpus中有{0}个词汇在sentiment_dict中出现了'.format(len(cunzai_word)))
        print('删除v、nr、ns之后在corpus中有{0}个词汇在sentiment_dict中未出现了'.format(len(nocunzai_word)))
        pp=0
        for word,ty in nocunzai_word.items():
            similar_word_list=model.most_similar(word)
            #print(similar_word)
            flag=0
            #print(word)
            #print(similar_word_list)
            try:
                for similar_word in similar_word_list:
                    #print(similar_word)
                    if similar_word[0] in sentiment_corpus_total.keys() :
                        #print(similar_word[0])
                        sentiment_corpus_total[word]  = sentiment_corpus_total[similar_word[0]]
                        print('word:({0}) 找到similar_word:({1})已有情感信息'.format(word, similar_word[0]))
                        flag = 1
                        break
                    elif   similar_word[0] in word_list:
                        sentiment_corpus_total[word] = sentiment_dict[word]
                        print('word:({0}) 找到similar_word:({1})已有情感信息'.format(word, similar_word[0]))
                        flag = 1
                        break
                    else:
                        flag=0
                        continue
                #print(ty)
            except  Exception  as result:
                print(result)
        return sentiment_corpus_total,corpus,shanchu_word
    sentiment_corpus_total,corpus,shanchu_word= getcorpussentiment(fcorpus)
    return  sentiment_corpus_total,corpus,shanchu_word
sentiment_corpus_total,corpus,shanchu_word=get_corpus_sentiment(all_text_neg,all_text_pos)




#得到删除之前的token个数，
word_list=[]
for doc in corpus_total:
    for word in doc:
        if word not in word_list:
            word_list.append(word)
word_list1=[]
for i,corpus0 in  corpus.items():
    for doc in corpus0:
        for word in doc:
            if word not in word_list1:
                word_list1.append(word)


#将删除的词汇导出来
##shanchu_word_list=shanchu_word['正面']+shanchu_word['反面']
##f=open('C:/Users/Administrator/Desktop/data/corpus/shanchu_word_list.txt','w',encoding='utf-8')
##f.write(shanchu_word_list)
##f.close()
print('删除之前的个数：{0}'.format(len(word_list)))
print('删除之后的个数：{0}'.format(len(word_list1)))

f=open('C:/Users/Administrator/Desktop/data/corpus/sentiment_corpus_train2.txt','w',encoding='utf-8')
f.write(str(sentiment_corpus_total))
f.close()

f=open('C:/Users/Administrator/Desktop/data/corpus/handle_corpus_train2.txt','w',encoding='utf-8')
f.write(str(corpus))
f.close()


