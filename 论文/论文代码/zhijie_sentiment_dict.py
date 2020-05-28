#直接加载进来外部情感词典，不进行构建
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
        sentiment_taiwan_dict[words]=2
    f.close()
    
with open(path6,'r',encoding='utf-8') as f:
    for word in f.readlines():
        words=word.lstrip('\ufeff').rstrip('\n')
        sentiment_taiwan_dict[words]=1
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
                jx=1
            elif type_list[i]=='反面':
                jx=2
            final_data[word]=jx
        print('{0}词语加载完毕'.format(type_list[i]))
    data={}
    pp=['程度most','程度very','程度over','程度more','程度shao','程度insufficiently']
    print(pp)
    i=0
    with open (path_list[-1],'r',encoding='utf-8') as f:
        for line in f.readlines():
            if line[0].isdigit():
                print(line[0])
                i+=1
                continue
            if line != '\n':
                #print(i)
                lines=line.strip().split('\n')[0]
                #print(lines)
                data[lines]=3
        print('程度副词加载完成')
        f.close()
    data.pop('\ufeff中文程度级别词语\t\t219(个数)')
    #print('data \n',data)
    #print('before:len_final:{0},len_data:{1}'.format(len(final_data),len(data)))
    final_data=dict(final_data,**data)
    #print(len(final_data))
    return final_data


sentiment_hownet_word=getdata([path0,path1,path2,path3,path4])
sentiment_two=sentiment_taiwan_dict
for word,jx in sentiment_hownet_word.items():
    if word in sentiment_two.keys():
        continue
    sentiment_two[word]=jx


#sentiment_dict1=dict(sentiment_taiwan_dict,**sentiment_hownet_word)


#处理情感词汇本体情感
path5='C:/Users/Administrator/Desktop/data/情感词汇本体/情感词汇本体.csv'
data_0=pd.read_csv(path5,engine='python')
data_0['极性']=data_0['极性'].map({0:0,1:1,2:2,3:3,7:2})

sentiment_dalian_word=dict(zip(data_0['词语'],data_0['极性']))
#sentiment_dict=dict(sentiment_dalian_word,**sentiment_dict1)
sentiment_dict=sentiment_two
for word, jx in sentiment_dalian_word.items():
    if word in sentiment_dict.keys():
        continue
    sentiment_dict[word]=jx

print('总的情感词典有:',len(sentiment_dict))

#进行整合情感词典  data 字典
f=open('C:/Users/Administrator/Desktop/data/corpus/sentiment_dict_goujian_zhijie.txt','w',encoding='utf-8')
f.write(str(sentiment_dict))
f.close()




#加载进来数据
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
            if word not in corpus_word_list.keys():
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
            if word not in corpus_word_list.keys():
                corpus_word_list[word] = flag
            doc.append(word)
        #print(doc)
        corpus_pos.append(doc)
    f.close()


train_text_neg=corpus_neg
train_text_pos=corpus_pos
fcorpus={'正面':train_text_pos,'反面':train_text_neg}
f=open('C:/Users/Administrator/Desktop/data/corpus/handle_corpus_zhijie.txt','w',encoding='utf-8')
f.write(str(fcorpus))
f.close()
