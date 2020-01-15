import pandas as pd
import random
import collections
from gensim.models  import word2vec
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#加载进来hownet情感词典以及中文情感词汇本体库情感词典
path0='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面评价词语（中文）.txt'
path1='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面评价词语（中文）.txt'
path2='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面情感词语（中文）.txt'
path3='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面情感词语（中文）.txt'
path4='C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/程度级别词语（中文）.txt'

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
    type_list=['正面评价','负面评价','正面情感','负面情感']
    final_data={}
    for i in range(len(path_list)-1):
        data0=getdata0(path_list[i])
        for word in data0:
            if word not in final_data.keys():
                final_data[word]=type_list[i]
            else:
                continue
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
                if lines not in data.keys():
                    data[lines]=pp[i-1]
        print('程度副词加载完成')
        f.close()
    data.pop('\ufeff中文程度级别词语\t\t219(个数)')
    #print('data \n',data)
    #print('before:len_final:{0},len_data:{1}'.format(len(final_data),len(data)))
    final_data=dict(final_data,**data)
    #print(len(final_data))
    return final_data
data=getdata([path0,path1,path2,path3,path4])


path5='C:/Users/Administrator/Desktop/data/情感词汇本体/情感词汇本体.csv'
data_0=pd.read_csv(path5,engine='python')

data_0['情感大分类']=data_0['情感分类'].map({'PA':'乐','PE':'乐',
                                             'PD':'好','PH':'好', 'PG':'好', 'PB':'好', 'PK':'好',
                                    'NF':'怒','NB':'哀','NJ':'哀','NH':'哀','PF':'哀','NI':'惧',
                                    'NC':'惧','NG':'惧','NE':'恶','ND':'恶','NN':'恶','NK':'恶',
                                    'NL':'恶','PC':'惊'})

data_u=data_0[['词语','情感大分类','强度','极性']]
data_u['情感大分类'].unique()
word0=list(data.keys())

#进行整合情感词典  data 字典
sentiment_dict=data_u[['词语','强度','情感大分类','极性']]
n=len(sentiment_dict)-1
up=sentiment_dict['词语'].values.tolist()
columns=sentiment_dict
for word in word0:
    if word in up:
        continue
    else:
        n+=1
        type_word=data[word]
        if type_word=='正面评价':
            sentiment_dict.insert(n,columns,[word,random.randint(1,9),'乐',1])
        elif type_word=='负面评价':
            sentiment_dict.insert(n,columns,[word,random.randint(1,9),'怒',2])
        elif type_word=='正面情感':
            sentiment_dict.insert(n,columns,[word,random.randint(1,9),'好',1])
        elif type_word=='负面情感':
            sentiment_dict.insert(n,columns,[word,random.randint(1,9),'恶',2])
        elif type_word=='程度most':
            sentiment_dict.insert(n,columns,[word,random.randint(5,9),'惊',0])
        elif type_word=='程度very':
            sentiment_dict.insert(n,columns,[word,random.randint(4,8),'恶',2])
        elif type_word=='程度over':
            sentiment_dict.insert(n,columns,[word,random.randint(3,6),'恶',2])
        elif type_word=='程度more':
            sentiment_dict.insert(n,columns,[word,random.randint(2,6),'好',0])
        elif type_word=='程度insufficiently':
            sentiment_dict.insert(n,columns,[word,random.randint(1,4),'哀',random.sample([0,2],1
                                                                                        )[0]])
        elif type_word=='程度shao':
            sentiment_dict.insert(n,columns,[word,random.randint(1,3),'哀',random.sample([0,2],
                                                                                        1)[0]])

sentiment_dict.to_csv('C:/Users/Administrator/Desktop/data/评论/sentiment_dict.txt',encoding='utf-8')


#对于语料库中的单词寻找情感词典
path6='C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/cut_pda_pos.txt'
path7='C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/cut_pda_neg.txt'





def sentiment_corpus(path):
    corpus = []
    with open(path6,'r',encoding='utf-8') as f:
        for line in f.readlines():
            lines=line.rstrip('\n').split(' ')
            #print(lines)
            corpus.append(lines)
    return corpus

corpus0=sentiment_corpus(path6)
corpus1=sentiment_corpus(path7)

fcorpus={'正面':corpus0,'反面':corpus1}
corpus=corpus0+corpus1
path_c='C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/cut_pda_all.txt'
with open(path_c,'w',encoding='utf-8') as f:
    for i in corpus :
        f.write(' '.join(i))
        f.write('\n')
    f.close()

sentence=[]
for doc in corpus:
    sentence.append(' '.join(doc))


model=word2vec.Word2Vec(corpus,size=200,window=5,min_count=1)
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentence))
weight_doc=tfidf.toarray()
word_list_c=vectorizer.vocabulary_
print(weight_doc.shape)
word_weight=weight_doc.T



def getcorpussentiment(fcorpus):
    sentiment_corpus_total={}
    word_list = sentiment_dict['词语'].values.tolist()
    cunzai_word={}
    nocunzai_word={}
    for typ1 ,cor in fcorpus.items():
        for doc in cor:
            for word in doc:
                if word in word_list:
                    # print('yes',word)
                    cunzai_word[word]=typ1
                    sentiment_corpus_total[word] = sentiment_dict.iloc[word_list.index(word)].values.tolist()
                else:
                    # print('no',word)
                    nocunzai_word[word]=typ1
                    sentiment_corpus_total[word]=[]
    print(len(sentiment_corpus_total))
    #print(list(nocunzai_word.keys())[:10])
    print('在corpus中有{0}个词汇在sentiment_dict中出现了'.format(len(cunzai_word)))
    print('在corpus中有{0}个词汇在sentiment_dict中未出现了'.format(len(nocunzai_word)))
    for word,ty in nocunzai_word.items():
        similar_word,par=model.most_similar(word)[0]
        #print(similar_word)
        if sentiment_corpus_total[similar_word] != []:
            sentiment_corpus_total[word]=sentiment_corpus_total[word]
            #print('word:({0}) 找到similar_word:({1})已有情感信息'.format(word,similar_word))
        else:
            #print('word:({0}) 未找到similar_word:({1})没有情感信息'.format(word, similar_word))
            #print(ty)
            if ty=='正面':
                jx=random.sample([1,0],1)[0]
                lx=random.sample(['好','乐'],1)[0]
                pd=random.randint(1,9)
                sentiment_corpus_total[word]=[word,pd,lx,jx]
            elif ty=='反面':
                jx=random.sample([2,0],1)[0]
                lx=random.sample(['怒','哀','恶','惧','惊'],1)[0]
                pd=random.randint(1,9)
                sentiment_corpus_total[word] = [word, pd, lx, jx]
    mm=0
    #######
    for word,da in sentiment_corpus_total.items():
        if len(da) ==0:
            sentiment_corpus_total[word]=[word,random.randint(1,9),random.sample(['怒','哀',
                                                                                  '恶','惧','惊',
                                                                                  '乐',
                                                                                  '好'],1)[0],
                                          random.randint(0,2)]
            mm+=1
    print('有{0}个词汇的情感是随机产生的'.format(mm))
    print(len(sentiment_corpus_total))
    return sentiment_corpus_total
sentiment_corpus_total= getcorpussentiment(fcorpus)

f=open('C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/sentiment_corpus.txt','w',encoding='utf-8')
f.write(str(sentiment_corpus_total))

f.close()

dat=list(sentiment_corpus_total.values())
set=pd.DataFrame(dat)
set.columns=['word','power','clarify','polarity']
#统计出语料库中的词汇，其多少表示正面，多少表示负面 以及兼有

jx={}
for index,row in set.iterrows() :
    if row['polarity']==1:
        jx['正面']=jx.get('正面',0)+1
    elif row['polarity']==0:
        jx['兼有']=jx.get('兼有',0)+1
    else:
        jx['负面']=jx.get('负面',0)+1

po={}
le=[]
e=[]
for index,row in set.iterrows() :
    if row['clarify'] == '好':
        po['好'] = po.get('好', 0) + 1
    elif row['clarify'] == '乐':
        po['乐'] = po.get('乐', 0) + 1
        le.append(row['word'])
    elif row['clarify'] == '恶':
        po['恶'] = po.get('恶', 0) + 1
        e.append(row['word'])
    elif row['clarify'] == '哀':
        po['哀'] = po.get('哀', 0) + 1
    elif row['clarify'] == '惊':
        po['惊'] = po.get('惊', 0) + 1
    elif row['clarify'] == '怒':
        po['怒'] = po.get('怒', 0) + 1
    elif row['clarify'] == '惧':
        po['惧'] =  po.get('惧', 0) + 1

by={}
for index,row in set.iterrows() :
    if row['polarity'] ==1:
        if row['clarify']=='好':
            by['好']=by.get('好',0)+1
        elif row['clarify'] =='乐':
            by['乐'] = by.get('乐', 0) + 1
        elif row['clarify']=='恶':
            by['恶'] = by.get('恶',0)+1
        elif row['clarify'] =='哀':
            by['哀'] = by.get('哀', 0) + 1
        elif row['clarify'] =='惊':
            by['惊'] = by.get('惊', 0) + 1
        elif row['clarify'] =='怒':
            by['怒'] = by.get('怒', 0) + 1
        elif row['clarify'] =='惧':
            by['惧'] = by.get('惧', 0) + 1

#提取一些表示

