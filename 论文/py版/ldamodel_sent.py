import sys

sys.path.append('G:/anconada/envs/py36/lib/site-packages')
from prettytable import PrettyTable
import re
import jieba
import os
import copy
from zhon.hanzi import punctuation
from scipy.misc import imread
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from prettytable import PrettyTable
import gc
import time
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from guppy import hpy
import json
import psutil
from gensim import corpora
from scipy.sparse import coo_matrix
from functools import partial

''' 在写论文的时候可以使用两组数据查看模型的效果'''


class ldamodel_sent():
    def __init__(self, topic_num, sentiment_num, alpha, beta, gamma,
                 cut_corpus, interation, final_info_sentword, df_info):
        # fianl_info_sentword  情感字典
        # df_info 为情感csv

        self.T = topic_num
        self.D = None
        self.S = sentiment_num
        self.V = None

        # 这两个词典用来指定词语的情感标签，若我们想要评论的喜怒哀乐（此时S=4）则使用df_info 中的情感大分类_表达作为该词的情感标签
        # 若想考查评论的态度（即正面评论还是反面评论，此时S=2）,则使用df_info中的情感大分类_态度作为该词的情感标签

        self.sentiment_dict = final_info_sentword  # 这是一个字典，key为word,value为list
        self.sentiment_map = df_info  # 这是一个数据框

        # 超参数设置
        self.alpha = alpha if alpha else 0.1
        self.beta = beta if beta else 0.1
        self.gamma = gamma if gamma else 0.1
        
        self.interation = interation if interation else 1000

        self.cut_corpus = cut_corpus
        self.word2id = None
        self.id2word = None
        self.dfs = None

        # 设置参数
        self.doc_sel_topic_count = None
        self.topic_sel_word_count = None
        self.doc_count = None
        self.topic_count = None
        self.sentiment_count = None
        self.topic_sentiment_count = None
        self.doc_sentiment_count = None
        self.sentiment_word_count=None
        self.topic_word_count=None

        self.doc_sel_topic = None
        self.topic_sel_word = None
        self.doc_sel = None

        self.z = None
        self.l = None

        if self.S == 4:
            print('我们开始查看评论的情感表达情况--喜怒哀乐')
        elif self.S == 2:
            print('我们开始考查评论的情感态度--正面或者反面')
        elif self.S == 7:
            print('我们开始考查评论的情感大类情况---乐、好、怒、哀、惧、恶、惊')

    def createdictionary(self):
        self.word2id={}
        self.V=0
        self.D=0
        wordnum=0
        for i ,doc in enumerate(self.cut_corpus):
            for j ,word in enumerate(doc):
                
                if word not in self.word2id.keys():
                    
                    if word != '':
                        wordnum+=1
                        self.word2id[word]=len(self.word2id)
                    else:
                        continue
                
        self.V=wordnum
        self.id2word=dict(zip(self.word2id.values(),self.word2id.keys()))
        self.D=len(self.cut_corpus)
        print('语料库中总共有{0} 个token'.format(self.V))
        print('word2id的长度：',len(self.word2id))
        print('id2word的长度',len(self.id2word))

        

    def initial(self):
        self.doc_sel_topic_count = np.zeros([self.D, self.S, self.T])
        print('the shape of doc_Sel_topic_count：',self.doc_sel_topic_count.shape)
        self.topic_sel_word_count = np.zeros([self.S, self.T, self.V])
        self.doc_count = np.zeros(self.D)
        self.sentiment_count = np.zeros(self.S)  ##这个不是必须的
        self.topic_count = np.zeros(self.T)  ##这个不是必须的
        self.topic_sentiment_count = coo_matrix((self.T, self.S)).toarray()
        self.doc_sentiment_count = coo_matrix((self.D, self.S)).toarray()
        self.sentiment_word_count=coo_matrix((self.S,self.V)).toarray()
        self.topic_word_count=coo_matrix((self.T,self.V)).toarray()
        
        self.doc_sel_topic = np.ndarray((self.D, self.S, self.T))
        self.topic_sel_word = np.ndarray((self.S, self.T, self.V))
        self.doc_sel = np.ndarray((self.D, self.S))

        self.z = coo_matrix((self.D, self.V), dtype=np.int8).toarray()  # 存放每个文档中每一个词的主题
        self.l = coo_matrix((self.D, self.V), dtype=np.int8).toarray()  # 存放每个文档中每个词的情感极性

        for i, doc in enumerate(self.cut_corpus):
            for j, word in enumerate(doc):
                try:
                    word_id = int(self.word2id[word])
                    topic = int(random.randint(0, self.T - 1))
                    #print('word is :{0}'.format(word))
                    senti_dalei = int(self.sentiment_dict[word][0])  # 需要注意的是这个地方随着查看的情感的不同需要一直改变
                    #print('情感大分类',senti_dalei)
                    if self.S == 7:
                        sentiment = int(senti_dalei)
                    elif self.S == 4:
                        
                        sentiment = self.sentiment_map[self.sentiment_map['情感大分类'] == senti_dalei]['情感大分类_表达'].tolist()[0]
                       
                    elif self.S == 2:
                        sentiment = self.sentiment_map[self.sentiment_map['情感大分类'] == senti_dalei]['情感大分类_态度'].tolist()[0]
                    else:
                        sentiment = int(random.randint(0, self.S - 1))
                    #print('label_sentiment:',sentiment)
                    #print('label_topic:',topic)
                    self.doc_sel_topic_count[i,sentiment,topic] += 1
                    self.topic_sel_word_count[sentiment, topic, word_id] += 1
                    self.doc_count[i] += 1
                    self.sentiment_count[sentiment] += 1
                    self.topic_count[topic] += 1
                    self.topic_sentiment_count[topic, sentiment] += 1
                    self.doc_sentiment_count[i, sentiment] += 1
                    self.sentiment_word_count[sentiment,word_id]+=1
                    self.topic_word_count[topic,word_id]+=1

                    self.z[i, word_id] = topic
                    self.l[i, word_id] = sentiment
                except Exception as result :
                    print(result)

    def _gibbsampling(self,cut_corpus, dstc, tswc, dc, sc, tc, tsc, dsc, twc,swc,z, l):
        for i, doc in enumerate(cut_corpus):
            for j, word in enumerate(doc):
                try:
                    word_id = int(self.word2id[word])
                    topic = int(self.z[i, word_id])
                    #print('word is :{0}'.format(word))
                    senti_dalei = int(self.sentiment_dict[word][0])
                    #print('大分类',senti_dalei)
                    if self.S == 7:
                        sentiment = senti_dalei
                    elif self.S == 4:
                        sentiment = self.sentiment_map[self.sentiment_map['情感大分类'] == senti_dalei]['情感大分类_表达'].tolist()[0]
                    elif self.S == 2:
                        sentiment = self.sentiment_map[self.sentiment_map['情感大分类'] == senti_dalei]['情感大分类_态度'].tolist()[0]
                    else:
                        sentiment = int(random.randint(0, self.S - 1))
                    #print('lable_sentoment',sentiment)
                    #print('label_topic',topic)
                    n_jkd = dstc[i, sentiment, topic] - 1
                    n_jkw = tswc[sentiment, topic, word_id] - 1
                    n_jk = tsc[topic, sentiment] - 1
                    n_kd = dsc[i, sentiment] - 1
                    n_d = dc[i] - 1
                    
                    new_topic, new_sentiment = list(map(int, self.resampling(n_jkd, n_jkw, n_jk, n_kd, n_d)))

                    z[i, word_id] = new_topic
                    l[i, word_id] = new_sentiment

                    dstc[i, sentiment, topic] -= 1
                    tswc[sentiment, topic, word_id] -= 1
                    tsc[topic, sentiment] -= 1
                    dsc[i, sentiment] -= 1
                    tc[topic] -= 1
                    sc[sentiment] -= 1
                    twc[topic,word_id]-=1
                    swc[sentiment,word_id]-=1

                    dstc[i, new_sentiment, new_topic] += 1
                    tswc[new_sentiment, new_topic, word_id] += 1
                    tsc[new_topic, new_sentiment] += 1
                    dsc[i, new_sentiment] += 1
                    tc[new_topic] += 1
                    sc[new_sentiment] += 1
                    twc[new_topic,word_id]+=1
                    swc[new_sentiment,word_id]+=1
                except Exception as result:
                    print(result)
        return dstc, tswc, dc, sc, tc, tsc, dsc,twc,swc, z, l

    def gibbssampling(self):
        for iter in range(self.interation):
            # 采用小批量进行训练
            self.doc_sel_topic_count, self.topic_sel_word_count, self.doc_count, self.sentiment_count, \
            self.topic_count, self.topic_sentiment_count, \
            self.doc_sentiment_count,self.topic_word_count,self.sentiment_word_count, self.z, self.l \
                = self._gibbsampling(self.cut_corpus,self.doc_sel_topic_count, self.topic_sel_word_count,
                                     self.doc_count, self.sentiment_count,
                                     self.topic_count, self.topic_sentiment_count,
                                     self.doc_sentiment_count, self.topic_word_count,self.sentiment_word_count,self.z, self.l)
            if iter % 10 == 0:
                print('已经迭代到了第{0}次了'.format(iter + 1))
        self.updateparam()

    ###验证模型

    def updateparam(self):
        start = time.time()
        for i in range(self.D):
            for j in range(self.S):
                self.doc_sel[i, j] = (self.doc_sentiment_count[i, j] + self.gamma)/(self.doc_count[i] + self.S * self.gamma)
        end = time.time()
        print('未使用partial 函数计算doc-sel矩阵所花费的时间为 : {0}'.format(end - start))

        for i in range(self.S):
            for j in range(self.T):
                for k in range(self.V):
                    self.topic_sel_word[i, j, k] = (self.topic_sel_word_count[i, j, k] + self.beta)/(self.topic_sentiment_count[j, i] + self.beta * self.V)
        for i in range(self.D):
            for j in range(self.S):
                for k in range(self.T):
                    self.doc_sel_topic[i, j, k] = (self.doc_sel_topic_count[i, j, k] + self.alpha)/(self.doc_sentiment_count[i, j] + self.T * self.alpha)
        print('参数更新完成******************* \n')
        return

    def resampling(self, n_jkd, n_jkw, n_jk, n_kd, n_d):
        pk = np.ndarray([self.T, self.S])
        for i in range(self.T):
            for j in range(self.S):
                pk[i, j] = float(n_jkd + self.alpha) * (n_jkw + self.beta) * (n_kd + self.gamma) / (
                        (n_kd + self.alpha * self.T) * (n_jk + self.beta * self.V) * (n_d + self.gamma * self.S))
                if i > 0 and j > 0:
                    pk[i, j] += pk[i, j - 1]
        # 轮盘方式随机选择主题
        u = random.random() * pk[self.T - 1, self.S - 1]
        for j in range(self.T):
            for k in range(self.S):
                if pk[j, k] >= u:
                    # print('get the new topic {0} and new sentiment {1}'.format(j,k))
                    return j, k
                else:
                    se = random.randint(0, self.S - 1)
                    to = random.randint(0, self.T - 1)
                    return to, se

    def predict(self, new_doc, isupdate=False):
        '''
            predict:new doc / comment
        '''
        # 对新文档进行切分等处理

        # 获取新文档中在word2id中存在的单词
        new_doc_word = list()
        for i ,doc in enumerate(new_doc):
            ii=[]
            for j,word in enumerate(doc):
                print(word)
                if word in list(self.word2id.keys()):
                    ii.append(word)
            new_doc_word.append(ii)
        n=len(new_doc)
        # 参数的设置  涉及到文档的矩阵需要重新设置一个新的，其余的不变
        new_dstc = np.zeros([n, self.S, self.T])
        new_dsc = np.zeros([n, self.S])
        new_dc = np.zeros([n])
        new_tswc = copy.deepcopy(self.topic_sel_word_count)
        new_sc = copy.deepcopy(self.sentiment_count)
        new_tsc = copy.deepcopy(self.topic_sentiment_count)
        new_tc = copy.deepcopy(self.topic_count)
        new_twc=copy.deepcopy(self.topic_word_count)
        new_swc=copy.deepcopy(self.sentiment_word_count)

        new_z = coo_matrix((n, self.V), dtype=np.int8).toarray()
        new_l = coo_matrix((n, self.V), dtype=np.int8).toarray()

        # 参数的更新，和之前的过程类似
        for i,doc in enumerate(new_doc_word):
            for j, word in enumerate(doc):
                word_id = int(self.word2id[word])
                topic = int(self.z[0, word_id])
                sentiment = int(self.l[0, word_id])

                new_dstc[i, sentiment, topic] += 1
                new_dsc[i, sentiment] += 1
                new_dc[i]+=1
                new_tswc[sentiment, topic, word_id] += 1
                new_sc[sentiment] += 1
                new_tsc[topic, sentiment] += 1
                new_tc[topic] += 1
                new_twc[topic,word_id]+=1
                new_swc[sentiment,word_id]+=1

                new_z[i, word_id] = topic
                new_l[i, word_id] = sentiment

        # 开始进行采样了
        for iter in range(0, self.interation):
            new_dstc, new_tswc, new_dc, new_sc, new_tc, new_tsc, new_dsc,new_twc,new_swc,new_z, new_l = \
                self._gibbsampling(new_doc_word,new_dstc, new_tswc, new_dc, new_sc, new_tc, new_tsc, new_dsc,new_twc,new_swc,new_z, new_l)

            print('new_doc 第{0}次训练'.format(iter + 1))
            # 此时要输出LDA模型的评价标准

        if isupdate == True:
            self.topic_sel_word_count = new_tswc
            self.sentiment_count = new_sc
            self.topic_sentiment_count = new_tsc
            self.topic_count = new_tc
            self.doc_sel_topic_count = np.r_[self.doc_sel_topic_count, new_dstc]
            self.doc_sentiment_count = np.r_[self.doc_sentiment_count, new_dsc]
            self.doc_count = np.r_[self.doc_count, new_dc]
            self.sentiment_word_count=new_swc
            self.topic_word_count=new_twc
            self.updateparam()
            print('加载new_doc之后选择更新参数，并更新完成')
        else:
            print('选择不更新参数')

        return [new_dstc, new_tswc, new_dsc, new_dc, new_tc, new_tsc, new_sc, new_z, new_l]

    def get_top_sentiment_topic(self, topnums):
        '''每一个sentiment下的top-topic
           可以利用prettyTable进行显示  
        '''
        table=PrettyTable()

        with open('C:/Users/Administrator/Desktop/data/评论/top_sentiment_topic_word.txt', 'w',encoding='utf-8') as f:
            for i in range(0, self.S):
                for j in range(self.T):
                    top_words = np.argsort(self.topic_sel_word[i,j :]).tolist()[0][:topnums]
                #print('输出的是每个topic的top_word的下标',top_words)
                    #print(top_words)
                    top_word = [self.id2word[kk] for kk in top_words]
                    table.add_column('sentiment{0}and topic_{1}'.format(i,j),top_word)
                    res = 'sentiment:{0},topic{1}:  \t {2} '.format(i,j,top_word)
                    f.write(res + '\n')
            f.close()
        print(table)
                # print(res)
    def get_sentiment_word(self,topnums):
        #得到每个sentiment下的word
        table=PrettyTable()
        with open('C:/Users/Administrator/Desktop/data/评论/top_sentiment_word','w',encoding='utf-8') as f:
            for i in range(self.S):
                top_words=np.argsort(self.sentiment_word_count[i,:]).tolist()[:topnums]
                #print(top_words)
                top_word=[self.id2word[j] for j in top_words]
                table.add_column('sentiment_{0}'.format(i),top_word)
                res='sentiment:{0} has topic_word  \t {1}'.format(i,top_word)
                f.write(res)
        print(table)
                
        
    def get_topic_word(self, topnums):
        '''得到每个topic下的top-word'''
    
        table=PrettyTable()
        with open('C:/Users/Administrator/Desktop/data/评论/top_topic_word_sel','w',encoding='utf-8') as f:
            for i in range(self.T):
                top_words=np.argsort(self.topic_word_count[i,:]).tolist()[:topnums]
                print(top_words)
                top_word=[self.id2word[j] for j in top_words]
                table.add_column('topic_{0}'.format(i),top_word)
                res='sentiment:{0} has topic_word  \t {1}'.format(i,top_word)
                f.write(res)
        print(table)
        

    def print_topic_word(self, doc_id, topic_list, word_nums=20):
        all_num = len(topic_list)
        table = PrettyTable()
        for i in topic_list:
            topword = np.argsort(self.topic_word[i, :])[:word_nums]
            table.add_column(i, [self.id2word[jj] for jj in topword])
        print(table)

        # 打印出来该文档上的主题分布以及在每个主题上面的个数的图形
        doc_topic_count = self.doc_topic_count[doc_id, :]
        sns.stripplot(x=list(range(0, all_num - 1)), y=doc_topic_count)
        for i in topic_list:
            sns.scatterplot(x=range(0, self.V - 1), y=self.topic_word[i, :])
            plt.show()
            sns.countplot(x=range(0, self.V - 1), hue=self.topic_word[i, :])
            plt.show()


if __name__ == '__main__':

    path = 'C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
    all_text = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
            yu=[]
            for word in lines:
                if word == ' ':
                    continue
                else:
                    yu.append(word)
            all_text.append(yu)
        f.close()
    comment_train, comment_test = train_test_split(all_text, test_size=0.3)

    path_1 = 'C:/Users/Administrator/Desktop/data/评论/final_info_sentword.txt'
    f = open(path_1, 'r', encoding='utf-8')
    data = f.read()
    test = re.sub('\'', '\"', data)
    test = test.lstrip('\ufeff')
    final_info_sentword = json.loads(test)

    path_2 = 'C:/Users/Administrator/Desktop/data/评论/df_info.csv'
    df_info = pd.read_csv(path_2, engine='python')
    
    df_info.columns=['id','情感分类','情感大分类','情感大分类_表达','情感大分类_态度']
    #print(df_info)

    # cut_corpus是cut_comment
    cut_corpus=all_text[:1000]
    M = ldamodel_sent(20, 4, 0.1, 0.1, 0.1, cut_corpus, 100, final_info_sentword, df_info)

    info = psutil.virtual_memory()

    print('没有运行initial 之前的内存使用情况')
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())

    M.createdictionary()
    M.initial()
    start = time.time()
    info = psutil.virtual_memory()

    print('没有运行Gibbs sampling 之前的内存使用情况')
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())

    M.gibbssampling()
    end = time.time()
    print('gibbssampling stage use {0} second'.format(end - start))
    test0 = [all_text[434:544]]
    new_dstc, new_tswc, new_dsc, new_dc, new_tc, new_tsc, new_sc, new_z, new_l = M.predict(test0)
    print(u'*********打印每个sentiment和topic下的top-word')
    M.get_top_sentiment_topic(topnums=20)
    print(u'*********打印每个sentiment下的top-word')
    M.get_sentiment_word(topnums=20)
    print(u'*********打印每个topic下的top-word')
    M.get_topic_word(topnums=20)
    print('运行Gibbs Sampling 之后的内存使用情况')
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())
