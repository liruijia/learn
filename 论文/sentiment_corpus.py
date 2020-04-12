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
from gensim import models,corpora
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from guppy import hpy
import json
import psutil
from gensim import corpora
from scipy.sparse import coo_matrix
from functools import partial
from scipy.special import gammaln, psi
import scipy
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report

''' 在写论文的时候可以使用两组数据查看模型的效果'''

class ldamodel_sent():
    def __init__(self, topic_num, sentiment_num, alpha, beta, gamma,corpus,
                 batch_size, iteration,sentiment_dict,label_total):
        self.T = topic_num
        self.D = None
        self.S = sentiment_num
        self.V = None
        self.corpus=corpus
        self.wordOccurenceMatrix=None
        self.label_total=label_total

        self.sentiment_dict = sentiment_dict  # 这是一个字典，key为word,value为list

        # 超参数设置
        self.alpha = alpha if alpha else 0.01
        self.beta = beta if beta else 0.01
        self.gamma = gamma if gamma else 0.01
        self.alpha_lz = None
        self.alphasum_l = None


        self.beta_lzw = None
        self.betasum_lz = None
        self.add_lw =None

        self.gamma_dl =None
        self.gammasum_d = None

        self.iteration = iteration if iteration else 1000
        self.batch_size=batch_size
        self.word2id = None
        self.id2word = None

        #
        self.coherence=None
        self.word_sentiment_vocabulary = {}
        self.sentiment_word_list = {}
        self.pre=[]
        self.recall=[]
        self.f1_score=[]
        
        # 设置参数
        self.doc_sel_topic_count = None
        self.topic_sel_word_count = None
        self.doc_count = None
        self.topic_count = None
        self.sentiment_count = None
        self.topic_sentiment_count = None
        self.sentiment_topic_count=None
        self.sentiment_topic_word_count=None
        self.doc_sentiment_count = None
        self.sentiment_word_count=None
        self.topic_word_count=None
        self.word_sentiment_vocabulary=None

        self.score=[]
        self.doc_sel_topic = None
        self.topic_sel_word = None
        self.sentiment_topic_word=None
        self.doc_sel = None

        self.z = None
        self.l = None
        self.all_loglikelihood=[]
        self.all_perplexity=[]


    def createdictionary(self):
        self.word2id={}
        self.V=0
        self.D=0
        wordnum=0
        for i ,doc in enumerate(self.corpus):
            for j ,word in enumerate(doc):
                if word not in self.word2id.keys():
                    if word != '':
                        wordnum+=1
                        self.word2id[word]=len(self.word2id)

                    else:
                        continue

        self.V=wordnum
        self.D=len(self.corpus)
        self.id2word=dict(zip(self.word2id.values(),self.word2id.keys()))
        print('语料库中总共有{0} 个token'.format(self.V))
        print('word2id的长度：',len(self.word2id))
        print('id2word的长度',len(self.id2word))



    def sampleFromDirichlet(self,gamma):
        return np.random.dirichlet(gamma)

    def sampleFromCategorical(self,theta):
        theta = theta/np.sum(theta)
        return np.random.multinomial(1, theta).argmax()
    def initial(self):
        self.alpha_lz = np.full((self.S, self.T), fill_value=self.alpha)
        self.alphasum_l = np.full((self.S,), fill_value=self.alpha * self.T)

        self.beta_lzw = np.full((self.S, self.T, self.V), fill_value=self.beta)
        self.betasum_lz = np.zeros((self.S, self.T), dtype=np.float)
        self.add_lw = np.ones((self.S, self.V), dtype=np.float)

        
        self.wordOccurenceMatrix=np.zeros((self.D,self.V))
        for i,doc in enumerate(self.corpus):
            for j,word in enumerate(doc):
                word_id=self.word2id[word]
                self.wordOccurenceMatrix[i,word_id] +=1
               
        for l in range(self.S):
            for z in range(self.T):
                for r in range(self.V):
                    self.beta_lzw[l][z][r] *= self.add_lw[l][r]
                    self.betasum_lz[l][z] += self.beta_lzw[l][z][r]

        self.gamma_dl = np.full((self.D, self.S), fill_value=0.0000001)
        self.gammasum_d = np.full(shape=(self.D), fill_value=0.0000001)
        for d in range(self.D):
            #self.gamma_dl[d][1] = 1.8
            self.gamma_dl[d][1] = self.gamma
            self.gamma_dl[d][2] = self.gamma
        for d in range(self.D):
            for l in range(self.S):
                self.gammasum_d[d] += self.gamma_dl[d][l]
        self.doc_sel_topic_count = np.zeros([self.D, self.S, self.T])
        print('the shape of doc_Sel_topic_count：',self.doc_sel_topic_count.shape)
        self.topic_sel_word_count = np.zeros([self.T, self.S, self.V])
        self.doc_count = np.zeros(self.D)
        self.sentiment_count = np.zeros(self.S)  ##这个不是必须的
        self.topic_count = np.zeros(self.T)  ##这个不是必须的
        self.topic_sentiment_count = coo_matrix((self.T, self.S)).toarray()
        self.doc_sentiment_count = coo_matrix((self.D, self.S)).toarray()
        self.sentiment_word_count=coo_matrix((self.S,self.V)).toarray()
        self.topic_word_count=coo_matrix((self.T,self.V)).toarray()

        self.sentiment_topic_count = coo_matrix((self.S, self.T)).toarray()
        self.sentiment_topic_word_count =np.zeros([self.S, self.T ,self.V])

        self.doc_sel_topic = np.ndarray((self.D, self.S, self.T))
        self.topic_sel_word = np.ndarray((self.T, self.S, self.V))
        self.doc_sel = np.ndarray((self.D, self.S))
        self.sentiment_topic_word=np.ndarray((self.S, self.T,self.V))
    


        self.z = coo_matrix((self.D, self.V), dtype=np.int8).toarray() # 存放每个文档中每一个词的主题
        self.l = coo_matrix((self.D, self.V), dtype=np.int8).toarray()  # 存放每个文档中每个词的情感极性

        print('开始赋予sentiment')
        gamma=[self.gamma for _ in range(self.S)]
        alpha=[self.alpha for _ in range(self.T)]
        self.alphaVec=np.full((self.D,self.S),fill_value=1/self.S)
        self.gammaVec=np.full((self.D, self.S, self.T), fill_value=1/self.T)
        for i, doc in enumerate(self.corpus):

            sentimentDistribution = self.sampleFromDirichlet(self.alphaVec[i,:])
            topicDistribution = np.zeros(( self.S,self.T))
            for s in range(self.S):
                topicDistribution[s, :] = self.sampleFromDirichlet(self.gammaVec[i,s,:])
            
            for j, word in enumerate(doc): 
                word_id = int(self.word2id[word])
                #topic = random.randint(0,self.T-1)
                topic = self.sampleFromCategorical(topicDistribution[s,:])
                if word in self.sentiment_dict.keys():
                    sentiment=self.sentiment_dict[word] # 需要注意的是这个地方随着查看的情感的不同需要一直改变
                else:
                    sentiment = self.sampleFromCategorical(sentimentDistribution)
                #print('sentiment',sentiment)
                self.doc_sel_topic_count[i,sentiment,topic] += 1
                #print(sentiment)
                self.topic_sel_word_count[topic,sentiment, word_id] += 1
                
                self.doc_count[i] += 1
                self.sentiment_count[sentiment] += 1
                self.sentiment_topic_count[sentiment,topic]+=1
                self.sentiment_topic_word_count[sentiment,topic,word_id]+=1

                self.topic_count[topic] += 1
                self.topic_sentiment_count[topic, sentiment] += 1
                self.doc_sentiment_count[i, sentiment] += 1
                self.sentiment_word_count[sentiment,word_id]+=1
                self.topic_word_count[topic,word_id]+=1

                self.z[i,word_id]=topic
                self.l[i,word_id]=sentiment
                



    def gibbssampling(self):
        for iter in range(self.iteration):
            # 采用小批量进行训练
            for i, doc in enumerate(self.corpus):
                
                for j, word in enumerate(doc):
                    try:
                        word_id = int(self.word2id[word])
                        topic = int(self.z[i,word_id])
                        sentiment=self.l[i,word_id]
                        
                        
                        if  self.doc_sel_topic_count[i, sentiment, topic]<=0:
                            self.doc_sel_topic_count[i, sentiment, topic]=0
                        else:
                            self.doc_sel_topic_count[i, sentiment, topic] -= 1

                        if self.topic_sel_word_count[topic,sentiment, word_id]<=0:
                            self.topic_sel_word_count[topic,sentiment, word_id]=0
                        else:
                            self.topic_sel_word_count[topic,sentiment, word_id] -= 1

                        if self.topic_sentiment_count[topic, sentiment]<=0:
                            self.topic_sentiment_count[topic, sentiment]=0
                        else:
                            self.topic_sentiment_count[topic, sentiment] -= 1

                        if self.doc_sentiment_count[i, sentiment]<=0:
                            self.doc_sentiment_count[i, sentiment] =0
                        else:
                            self.doc_sentiment_count[i, sentiment]-=1

                        if self.topic_count[topic]<=0:
                            self.topic_count[topic]=0
                        else:
                            self.topic_count[topic] -= 1

                        if self.sentiment_count[sentiment]<=0:
                            self.sentiment_count[sentiment]=0
                        else:
                            self.sentiment_count[sentiment] -=1

                        if self.topic_word_count[topic,word_id]<=0:
                            self.topic_word_count[topic,word_id]=0
                        else:
                            self.topic_word_count[topic,word_id]-=1

                        if self.sentiment_word_count[sentiment,word_id]<=0:
                            self.sentiment_word_count[sentiment,word_id]=0
                        else:
                            self.sentiment_word_count[sentiment,word_id]-=1
                            
                        if self.sentiment_topic_word_count[sentiment,topic,word_id]<=0:
                            self.sentiment_topic_word_count[sentiment,topic,word_id]=0
                        else:
                            self.sentiment_topic_word_count[sentiment,topic,word_id]-=1

                        if self.sentiment_topic_count[sentiment,topic]<=0:
                            self.sentiment_topic_count[sentiment,topic]=0
                        else:
                            self.sentiment_topic_count[sentiment,topic]-=1


                        new_topic, new_sentiment = self.resampling(i, word_id)
                        
#                        ind = self.sampleFromCategorical(probabilities_ts.flatten())
#                        new_topic, new_sentiment = np.unravel_index(ind, probabilities_ts.shape)
                        
                        self.z[i,word_id]=new_topic
                        self.l[i,word_id]=new_sentiment
                        
                        
                        self.doc_sel_topic_count[i, new_sentiment, new_topic] += 1
                        self.topic_sel_word_count[ new_topic,new_sentiment , word_id] += 1
                        self.topic_sentiment_count[new_topic, new_sentiment] += 1
                        self.doc_sentiment_count[i, new_sentiment] += 1
                        self.topic_count[new_topic] += 1
                        self.sentiment_count[new_sentiment] += 1
                        self.topic_word_count[new_topic,word_id]+=1
                        self.sentiment_word_count[new_sentiment,word_id]+=1
                        self.sentiment_topic_count[new_sentiment,new_topic]+=1
                        self.sentiment_topic_word_count[new_sentiment,new_topic,word_id ]+=1
                        

                    except Exception as result:
                        print('result:',result)
            
            
            #self.updateparam()
            #self.get_word_sentiment0(0.001)
            #score=self.pingu(0.0001,type='umass')
            #self.score.append(score)
##            if (iter+1)%10==0:
##                self.updateparam()
##                score,recall,f1_score=self.getzhibiao()
##                self.pre.append(score)
##                self.recall.append(recall)
##                self.f1_score.append(f1_score)
            print('开始第{0}次迭代'.format(iter+1))
        self.updateparam()
        print('训练过程结束')
        
    ###验证模型

    def updateparam(self):
        for i in range(self.D):
            for j in range(self.S):
                self.doc_sel[i, j] = (self.doc_sentiment_count[i, j] + self.gamma)/(self.doc_count[i] +
                                                                                             self.S * self.gamma)


        for i in range(self.T):
            for j in range(self.S):
                for k in range(self.V):
                    self.topic_sel_word[i, j, k] = (self.topic_sel_word_count[i, j, k] + self.beta)/\
                                                   (self.topic_sentiment_count[i, j] + self.beta * self.V)

        for i in range(self.S):
            for j in range(self.T):
                for k in range(self.V):
                    self.sentiment_topic_word[i, j, k] = (self.sentiment_topic_word_count[i, j, k] + self.beta)/\
                                                   (self.sentiment_topic_count[i, j] + self.beta * self.V)

        for i in range(self.D):
            for j in range(self.S):
                for k in range(self.T):
                    self.doc_sel_topic[i, j, k] = (self.doc_sel_topic_count[i, j, k] + self.alpha)/\
                                                  (self.doc_sentiment_count[i, j] + self.alpha *self.T )
        print('参数更新完成******************* \n')
        return


    
    
    def resampling(self,doc_id,word_id ):
        pk = np.ndarray([self.T, self.S])
        #print('开始训练')
        for t in range(self.T):
            for s in range(self.S):
                pk[t,s]= float((self.doc_sel_topic_count[doc_id,s,t] + self.alpha) *\
                          (self.topic_sel_word_count[ t,s,word_id] + self.beta_lzw[s,t,word_id]) * \
                          (self.doc_sentiment_count[doc_id, s]+ self.gamma_dl[doc_id,s]) / \
                          (self.doc_sentiment_count[doc_id, s] + self.alpha * self.T) *\
                          (self.topic_sentiment_count[t,s]+ self.betasum_lz[s,t]) * \
                          (self.doc_count[doc_id] + self.gammasum_d[doc_id]))
##        probabilities_ts = np.ones((self.T, self.S))
##        firstFactor = (self.doc_sentiment_count[doc_id] + self.alpha) / \
##            (self.doc_count[doc_id] + self.T * self.alpha)
##        secondFactor = (self.doc_sel_topic_count[doc_id, :, :] + self.gamma) / \
##            (self.doc_sentiment_count[doc_id, :] + self.S * self.gamma)[:, np.newaxis]
##        thirdFactor = (self.topic_sel_word_count[:, :, word_id] + self.beta) / \
##            (self.topic_sentiment_count + self.topic_sel_word_count.shape[0] * self.beta)
##        print('****',firstFactor.shape,secondFactor.shape,thirdFactor.shape)
##        probabilities_ts =np.dot(probabilities_ts , firstFactor[:, np.newaxis])
##        print(firstFactor.shape,secondFactor.shape,thirdFactor.shape)
##        probabilities_ts = np.dot(probabilities_ts,np.dot(secondFactor , thirdFactor))
##        probabilities_ts =probabilities_ts/np.sum(probabilities_ts)

                pk[t, s] += pk[t, s - 1]
##        # 轮盘方式随机选择主题5

        u = random.random()*pk[self.T-1,self.S-1]
        flag=0
        for j in range(self.T):
            for k in range(self.S):
                if pk[j, k] >= u:
                    flag=1
                    return j,k
        if flag==0:
            se = random.randint(0, self.S - 1)
            to = random.randint(0, self.T - 1)
            return to, se

##        pk /= np.sum(pk)
##        return pk 



    




    def get_top_sentiment_topic(self, topnums):
        '''每一个sentiment下的top-topic
           可以利用prettyTable进行显示
        '''
        table=PrettyTable()

        with open('C:/Users/Administrator/Desktop/data/评论/top_sentiment_topic_word.txt', 'w',encoding='utf-8') as f:
            for i in range(0, self.T):
                for j in range(self.S):
                    top_words = np.argsort(self.topic_sel_word[i,j :]).tolist()[:topnums]
                #print('输出的是每个topic的top_word的下标',top_words)
                    print(top_words)
                    top_word = [self.id2word[kk] for kk in top_words[0]]
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

    def get_wordsentiment(self,sentiment_index,topnum=20):
        '''
        得到给定一个情感下的每个主题的词汇，以及概率
        只输出其中三个topic/的topnum的词汇
        sentiment_word
        利用
        :return:
        '''
        print('显示sentiment_index={0}下的3个主题的关键词汇'.format(sentiment_index))
        table=PrettyTable()
        word_list=[]
        for i in range(3):
            values_sentiment_topic=self.sentiment_topic_word[sentiment_index,i,:]
            max_list=np.argsort(values_sentiment_topic,).tolist()[-topnum:]
            ui=[]
            for id in max_list:
                ui.append((self.id2word[id],values_sentiment_topic[id]))
            #print(ui)
            word_list.append(ui)
            table.add_column('topic_{0}'.format(i),top_word)
        return  word_list

    
    

    def get_word_sentiment0(self,seta):
        '''
        得到每个词语的情感极性
        :return:self.word_sentiment_vocabulary
        '''
        # 主要使用sentiment-topic-word
        print('开始求解sentiment_word_list')
        sentiment_word_list={}
        word_sentiment_vocabulary={}
        for i in range(self.V):
            values=self.sentiment_topic_word[:,:,i]
            
            # 获取到了每一个主题下每一个词汇的值，根据其最大值所在位置，找到该所所在的情感
            m_index=np.argmax(values,axis=1)
            max_list=np.array([values[ii ,m_index[ii]] for ii in range(2)])
            max_index=np.argmax(max_list)
            word_sentiment_vocabulary[self.id2word[i]]=(max_index,max_list[max_index])
            if max_list[max_index] >=seta :
                if max_index not in sentiment_word_list.keys():
                    sentiment_word_list[max_index]=[]
                sentiment_word_list[max_index].append([self.id2word[i],max_list[max_index]])
            else:
                continue
        self.sentiment_word_list=sentiment_word_list
        self.word_sentiment_vocabulary=word_sentiment_vocabulary
        print('得到所有词汇利用主题模型得到的情感极性')
        
        return
    
    def get_topic_word(self, topnums):
        '''得到每个topic下的top-word'''

        table=PrettyTable()
        with open('C:/Users/Administrator/Desktop/data/评论/top_topic_word_sel','w',encoding='utf-8') as f:
            for i in range(self.T):
                top_words=np.argsort(self.topic_word_count[i,:]).tolist()[:topnums]
                print(top_words)
                top_word=[self.id2word[j] for j in top_words]
                table.add_column('topic_{0}'.format(i),top_word)
                res='topic:{0} has topic_word  \t {1}'.format(i,top_word)
                f.write(res)
        print(table)


    def pingu(self,emplsion,type):
        '''
        :param 使用topic coherence  对模型进行评价  umass metric  ----score d(j)-表示语料库中包含词汇j的个数  d(i,j)表示词汇i和j共同出现的词汇、
            其中词汇i以及j取自同一个情感标签
        :return:topic coherence
        '''
        coherence_umass=[]
        for i in range(self.S):
            word_list=[word_par[0] for word_par in self.sentiment_word_list[i]]
            coherence_umass.append(sum([self._scoreumass(i,j,emplsion) for i in word_list for j in word_list] ))
        self.coherence=sum(coherence_umass)
        print('umass_metric 下的topic coherence：{0}'.format(self.coherence ))
        return self.coherence


    def _scoreumass(self,i,j,emplsion):
        '''
        :param i: 词汇i
        :param j: 词汇j
        :param emplsion:
        :return: score
        '''
        count_j=0
        count_ij=0
        for doc in self.corpus:
            if j in doc:
                count_j+=1
            if j in doc and i in doc:
                count_ij+=1
        score =np.log2((count_ij+emplsion)/count_j)
        return score
    
    def getword_sentiment(self):
        sentiment_word_list={}
        word_sentiment_vocabulary={}
        for i in range(self.V):
            values=self.sentiment_topic_word[:,:,i]
            
            # 获取到了每一个主题下每一个词汇的值，根据其最大值所在位置，找到该所所在的情感
            m_index=np.argmax(values,axis=1)
            max_list=np.array([values[ii ,m_index[ii]] for ii in range(4)])
            max_index=np.argmax(max_list)
            word_sentiment_vocabulary[self.id2word[i]]=(max_index,max_list[max_index])
            if max_list[max_index] >=0.00001:
                if max_index not in sentiment_word_list.keys():
                    sentiment_word_list[max_index]=[]
                sentiment_word_list[max_index].append([self.id2word[i],max_list[max_index]])
            else:
                continue
        return sentiment_word_list
        
    def getzhibiao(self):
        pre_label=[]
        dos=self.doc_sel
        doc=self.l
        sum0=0
        id2word=corpora.Dictionary(self.corpus)
        word2id=id2word.token2id
        corpus = [id2word.doc2bow(text) for text in self.corpus]
        model=models.TfidfModel(corpus,id2word=id2word)
        
        model_cxl=word2vec.Word2Vec(self.corpus,size=200,window=5,min_count=1,workers=5)
        wordlist=list(self.word2id.keys())
        sentiment_word_list=self.getword_sentiment()
        print(sentiment_word_list.keys())
        hownet_word_good=[word[0] for word in sentiment_word_list[1]]
        hownet_word_bad=[word[0] for word in sentiment_word_list[2]]
        weight_corpus=model[corpus]
##        hownet_word_good=[]
##        hownet_word_bad=[]
##        with open('C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面.txt') as f :
##            for lines in f.readlines():
##                line=lines.strip().split(' ')[0]
##                if line not in hownet_word_good:
##                    if line in wordlist:
##                        hownet_word_good.append(line)
##            f.close()
##        with open('C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/反面.txt') as f :
##            for lines in f.readlines():
##                line=lines.strip().split(' ')[0]
##                if line not in hownet_word_bad:
##                    if line in wordlist:
##                        hownet_word_bad.append(line)
##            f.close()
        #model.similarity(u"陕西省", u"山西省")
        for i,doc in enumerate(self.corpus):
            sum0=0
            sent={}
            for word in doc:
                sent[word]=sent.get(word,0)+1
            j=0
            for word in sent.keys():
                wordid=self.word2id[word]
                jx=self.l[i,wordid]
                ui=(sum([model_cxl.similarity(word,oo) for oo in hownet_word_good ])/len(hownet_word_good) -
                    sum([model_cxl.similarity(word,oo) for oo in hownet_word_bad])/len(hownet_word_bad))*sent[word]

                sum0+=ui*weight_corpus[i][j][1]
                j+=1
            if sum0 >0:
                pre_label.append(1)
            else:
                pre_label.append(2)
            

        #1.准确度、召回率、F1值、得到不同主题个数下的召回情况
        
        score=accuracy_score(self.label_total, pre_label)
        recall=metrics.recall_score(self.label_total, pre_label, average='macro')
        f1_score=metrics.f1_score(self.label_total, pre_label, average='weighted')

        
        target_names = ['sentiment_1','sentiment_2']
        print(classification_report(self.label_total, pre_label, target_names=target_names))
        return  score,recall,f1_score,pre_label

    
    
    def print_doc_topic_word(self, doc_id, topic_list, word_nums=20):
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
    path1='C:/Users/Administrator/Desktop/data/corpus/handle_corpus_train.txt'
    print('开始加载数据')
    da=open(path1,encoding='utf-8').read()
    da1=da.lstrip('\ufeff')
    data=json.loads(da1)
    corpus_total=data['正面'][:2500]+data['反面'][:2500]
    label_total=[1]*len(data['正面'][:2500])+[2]*len(data['反面'][:2500])
    
            
            
    
    sentiment_path='C:/Users/Administrator/Desktop/data/corpus/sentiment_corpus_train.txt'
    se = open(sentiment_path, encoding='utf-8').read()
    se = se.lstrip('\ufeff')
    sentiment_dict= json.loads(se)
    print('水杯评论加载完毕')
    
    corpus_cup=corpus_total
##    for doc in corpus_total:
##        if len(doc) !=0:
##            corpus_cup.append(doc)
    #comment_train, comment_test = train_test_split(corpus, test_size=0)
    comment_train=corpus_cup
    # cut_corpus是cut_comment
    #topic_list=[10,50,70,100]
    result=[]
    score_list=[]
    recall_list=[]
    f1_score_list=[]
    topic_nums=[2,30,100,10,50,80]
    
    P = ldamodel_sent(50, 4, 0.01, 0.01, 0.01, comment_train, len(comment_train), 10, sentiment_dict,label_total)
    P.createdictionary()
    print('开始初始化')
    P.initial()
    print('初始化阶段的doc_sentiment_count')
    print(P.doc_sentiment_count)
    print(P.doc_count)
    start=time.time()
    P.gibbssampling()
    score,recall,f1_score,pre_label=P.getzhibiao()
    print('score:{0},recall:{1},f1_score:{2}'.format(score,recall,f1_score))
    score_list.append(score)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
        
    

##    sns.pointplot(x=topic_nums,y=score_list)
##    plt.show()
    #获取预测得到的标签信息
    



    
##    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
##    import scipy.cluster.hierarchy as sch
##    from  sklearn.decomposition import PCA
##    pca=PCA()
##    x=M.topic_w
##    x_new=pca.fit_transform(x)
##    disMat = sch.distance.pdist(x_new,'euclidean') 
##    Z = linkage(disMat,  method='ward')
##    P=sch.dendrogram(Z)
##    plt.title('Dendrogram for the Agglomerative Clustering')
##    plt.xlabel('sample_index')
##    plt.ylabel('distance')
##    plt.show()
##
##    
##    labels_1 = fcluster(Z, t=1, criterion='inconsistent')
##    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr','op','*p','*r','*b','*d','*k']
##    j = 0 
##    for i in label_pred:
##        plt.plot([x_new[j,0]], [x_new[j,1]], mark[i], markersize = 5)
##        j +=1
##    topic_nums_new=np.max(labels_1)
##    
##    #根据聚类结果将topic进行合并
##    m,n=x.shape
##    new_twc=x
##    twc_new=np.zeros([topic_nums_new,n])
##    for i in range(topic_nums_new):
##        ui=[j for j in range(len(labels_1)) if labels_1[j]==i+1]
##        print(ui)
##        for ji in ui:
##            twc_new[i,:]+=new_twc[ji,:]
##                
##    #得到聚类之后的twc矩阵，进行聚类图展示
##    plt.scatter(twc_new[:,1],twc_new[:,0])
##    
##    #相应的将stwc中的信息进行合并，最后显示各个情感下不同topic的Word信息 表
##    m,n,p=M.sentiment_topic_word_count.shape
##    m1,n1=M.sentiment_topic_count.shape
##    stc_new = coo_matrix((m,topic_nums_new)).toarray()
##    stwc_new=np.zeros([m,topic_nums_new,p])
##
##    for i in range(m):
##        for j in range(topic_nums_new):
##            ui=[jj for jj in range(len(labels_1)) if labels_1[jj]==j+1]
##            for ji in ui:
##                stwc_new[i,j,:]+=new_stwc[i,ji,:]
##            stc_new[i,j]=sum(stwc_new[i,j,:])
##    
##    
##    
##    #得到sentiment_topic_word 
##    stw_new=np.zeros((m,topic_nums_new,p))
##    for i in range(m):
##        for j in range(topic_nums_new):
##            for k in range(p):
##                stw_new[i, j, k] = (stwc_new[i, j, k] + 0.01)/(stc_new[i, j] + 0.01* V)

    #画表-得到每个sentiment下的topic 下的top10word以及概率

    
    topnum=20
    id2word=P.id2word
    stw_new=P.sentiment_topic_word
    #好评
    table=PrettyTable()
    word_list=[]
    print('好评')
    for i in range(3):
        values_sentiment_topic=stw_new[1,i,:]
        max_list=np.argsort(values_sentiment_topic,).tolist()[-topnum:]
        ui=[]
        for id in max_list:
            ui.append((id2word[id],values_sentiment_topic[id]))
        #print(ui)
        word_list.append(ui)
        table.add_column('topic_{0}'.format(i),ui)
    print(table)

    
    good_word=[]
    for j in range(3):
        for i in range(len(word_list[j])):
            good_word.append(word_list[j][i][0])
    good_word0=[]
    for word in good_word:
        if word not in good_word0:
            good_word0.append(word)
    
    print(len(good_word))

    good_df=[]
    model=word2vec.Word2Vec(comment_train,size=200,window=5,min_count=1,workers=5)
    for i in good_word0:
        good_df.append(model[i].flatten().tolist())

    goo_df=pd.DataFrame(good_df)
    plt.scatter(goo_df[0],goo_df[1],marker='o')
    print(goo_df.columns)

    #bad
    print('差评')
    table=PrettyTable()
    word_list1=[]
    
    for i in range(3):
        values_sentiment_topic=stw_new[2,i,:]
        max_list=np.argsort(values_sentiment_topic,).tolist()[-topnum:]
        ui=[]
        for id in max_list:
            ui.append((id2word[id],values_sentiment_topic[id]))
        #print(ui)
        word_list1.append(ui)
        table.add_column('topic_{0}'.format(i),ui)
    print(table)

    
    bad_word=[]
    for j in range(3):
        for i in range(len(word_list1[j])):
            bad_word.append(word_list1[j][i][0])
    bad_word0=[]
    for word in bad_word:
        if word not in bad_word0:
            bad_word0.append(word)
    
    print(len(bad_word))

    bad_df=[]
    
    for i in bad_word0:
        bad_df.append(model[i].flatten().tolist())

    ba_df=pd.DataFrame(bad_df)

    #所以提取出来的词汇
    import pylab as mpl
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    total_df=pd.concat([ba_df,goo_df])
    total_df['label']=[2 for _ in range(len(ba_df))]+[1 for _ in range(len(goo_df))]
    total_df['word']=bad_word0+good_word0
    
    plt.scatter(total_df[7],total_df[3])
    for j,row in total_df.iterrows():
        print(row['label'])
        if row['label'] == 2:
            plt.text(row[7], row[3], row['word'], bbox=dict(facecolor='green',alpha=0.1))
        else:
            plt.text(row[7], row[3], row['word'], bbox=dict(facecolor='red',alpha=0.1))
    
    plt.axis('off')
    plt.savefig('C:/Users/Administrator/Desktop/data/corpus/good_bad_word.jpg')
    plt.show()
        
     #得到其中不同sentiment下的主题分布
    sentiment_topic_word=P.sentiment_topic_word
    sse=P.sentiment_topic_word[2,:,:]
    num_show_term=7
    topic_list=[0,3,6,9]
    for i, k in enumerate([1,3]):
        ax = plt.subplot(2, 1, i+1)
        item_dis_all = sse[k,:]
        item_dis = np.argsort(item_dis_all)[-num_show_term:]
        item_par=[item_dis_all[i] for i in item_dis]
        
        ax.plot(range(num_show_term), item_par, 'b*')
        
        word = [P.id2word[i] for i in item_dis]
        ax.set_ylabel(u"概率")
        for j in range(num_show_term):
            ax.text(j, item_par[j], word[j], bbox=dict(facecolor='green',alpha=0.1))
    plt.suptitle(u'负面情感标签下不同主题对应的top词汇概率图', fontsize=18)
    plt.savefig('C:/Users/Administrator/Desktop/论文/picture/jstzhuti_neg.jpg')
    plt.show()
    



    
    
    #得到其中几则评论的情感分布图

    test={} #test的key值为文档编号
    for i,doc in enumerate(comment_train):
        if len(doc)>=20:
            test[i]=doc
    #获得上面test中10则评论所对应的文档{-}情感分布
    
    ds=P.doc_sel
    print(ds)
    table=PrettyTable(['id','情感分布','comment'])
    for i in range(10):
        ui=[round(oo,3) for oo in ds[i,:]]
        table.add_row([i+1,ui,' '.join(comment_train[i][:6])+'...'])
    print(table)
    #最后得到word-sentiment-vocabulary
##    word_sentiment_vocabulary={}
##    for i in range(p):
##        values=stw_new[:,:,i]
##        # 获取到了每一个主题下每一个词汇的值，根据其最大值所在位置，找到该所所在的情感
##        m_index=np.argmax(values,axis=1)
##        max_list=np.array([values[ii ,m_index[ii]] for ii in range(4)])
##        max_index=np.argmax(max_list)
##        word_sentiment_vocabulary[id2word[i]]=(max_index,max_list[max_index])   

##    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
##    import scipy.cluster.hierarchy as sch
##    from  sklearn.decomposition import PCA
##    pca=PCA()
##    x=new_twc
##    x_new=pca.fit_transform(x)
##    disMat = sch.distance.pdist(x_new,'euclidean') 
##    Z = linkage(disMat,  method='ward')
##    P=sch.dendrogram(Z)
##    plt.title('Dendrogram for the Agglomerative Clustering')
##    plt.xlabel('sample_index')
##    plt.ylabel('distance')
##    plt.show()

    
#    labels_1 = fcluster(Z, t=1, criterion='inconsistent')
##    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr','op','*p','*r','*b','*d','*k']
##    j = 0 
##    for i in label_pred:
##        plt.plot([x_new[j,0]], [x_new[j,1]], mark[i], markersize = 5)
##        j +=1
##    topic_nums_new=np.max(labels_1)
##    
##    #根据聚类结果将topic进行合并
##    m,n=x.shape
##    new_twc=x
##    twc_new=np.zeros([topic_nums_new,n])
##    for i in range(topic_nums_new):
##        ui=[j for j in range(len(labels_1)) if labels_1[j]==i+1]
##        print(ui)
##        for ji in ui:
##            twc_new[i,:]+=new_twc[ji,:]
##                
##    #得到聚类之后的twc矩阵，进行聚类图展示
##    plt.scatter(twc_new[:,1],twc_new[:,0])
##    
##    #相应的将stwc中的信息进行合并，最后显示各个情感下不同topic的Word信息 表
##    m,n,p=M.sentiment_topic_word_count.shape
##    m1,n1=M.sentiment_topic_count.shape
##    stc_new = coo_matrix((m,topic_nums_new)).toarray()
##    stwc_new=np.zeros([m,topic_nums_new,p])
##
##    for i in range(m):
##        for j in range(topic_nums_new):
##            ui=[jj for jj in range(len(labels_1)) if labels_1[jj]==j+1]
##            for ji in ui:
##                stwc_new[i,j,:]+=new_stwc[i,ji,:]
##            stc_new[i,j]=sum(stwc_new[i,j,:])
##    
    
    
##    #得到sentiment_topic_word 
##    stw_new=np.zeros((m,topic_nums_new,p))
##    for i in range(m):
##        for j in range(topic_nums_new):
##            for k in range(p):
##                stw_new[i, j, k] = (stwc_new[i, j, k] + 0.01)/(stc_new[i, j] + 0.01* V)
##
    #画表-得到每个sentiment下的topic 下的top10word以及概率

    table=PrettyTable()
    topnum=20
    id2word=P.id2word
    
    word_list=[]
    for i in range(3):
        values_sentiment_topic=stw_new[1,i,:]
        max_list=np.argsort(values_sentiment_topic,).tolist()[-topnum:]
        ui=[]
        for id in max_list:
            ui.append((id2word[id],values_sentiment_topic[id]))
        #print(ui)
        word_list.append(ui)
        table.add_column('topic_{0}'.format(i),ui)
    print(table)

    ds=P.doc_sel


    op=ds[-10:]
    ol=comment_train[-10:]
    table=PrettyTable(columns=['分布','评论'])
    for i in range(0,10):
        table.add_row([op[i],' '.join(ol[i][:6])+'...'])
    print(table)
        
        
        

    
    #最后得到word-sentiment-vocabulary
##    word_sentiment_vocabulary={}
##    for i in range(p):
##        values=stw_new[:,:,i]
##        # 获取到了每一个主题下每一个词汇的值，根据其最大值所在位置，找到该所所在的情感
##        m_index=np.argmax(values,axis=1)
##        max_list=np.array([values[ii ,m_index[ii]] for ii in range(4)])
##        max_index=np.argmax(max_list)
##        word_sentiment_vocabulary[id2word[i]]=(max_index,max_list[max_index])

    #情感得分的计算

##    sentence=[]
##    for doc in corpus_cup:
##        sentence.append(' '.join(doc))
##    id2word=corpora.Dictionary(corpus_cup)
##    word2id=id2word.token2id
##    corpus = [id2word.doc2bow(text) for text in comment_train]
##    model=models.TfidfModel(corpus,id2word=id2word)
##    weight_corpus=model[corpus]
##    model_word2id=P.word2id
##    def get_sentiment_score(pinglun,pinglun_index):
##        jx_list=[]
##        wordid_list=[]
##        power=[]
##        weight_list=[]
##        for word in pinglun :
##            id_word=word2id[word]
##            jx=P.l[pinglun_index,model_word2id[word]]
##            if jx==0:
##                power.append(1)
##            elif jx==3:
##                power.append(3)
##            elif jx==2:
##                power.append(5)
##            else:
##                power.append(4)
##            jx_list.append(jx)
##            wordid_list.append(id_word)
##            for il in weight_corpus[int(pinglun_id)]:
##                if il[0]==id_word:
##                    weight_list.append(il[1])
##        score=sum([power[i]*weight_list[i]  for i in range(len(pinglun))])
##        return score
##
##
##    #按照店铺整理数据
    path_info='C:/Users/Administrator/Desktop/data/评论/product_info_cup_before.csv'
    info_dianpu=pd.read_csv(path_info,engine='python')

    ii=info_dianpu['shop_id'].value_counts()

    dianpu_info={}
    count_ii=ii.head(n=10)
    for i in range(len(count_ii)):
        product_id=info_dianpu[info_dianpu['shop_id']==count_ii.index[i]]['product_id'].values.tolist()
        shop_name=info_dianpu[info_dianpu['shop_id']==count_ii.index[i]]['shop_name'].unique()[0]
        if shop_name not in dianpu_info.keys():
            dianpu_info[shop_name]=[]
        dianpu_info[shop_name].extend(product_id)

    path='C:/Users/Administrator/Desktop/data/评论/comment_info_cup_final.csv'
    df_data=pd.read_csv(path,engine='python')
    oo=df_data['referenceId'].value_counts()
    pp=oo.index.tolist()
    corpus_dianpu={}
    for dianpu,product_id_list in dianpu_info.items():
        dianpu_corpus={}
        for product_id in product_id_list:
            if product_id not in dianpu_corpus.keys():
                dianpu_corpus[product_id]={}
            value = df_data[df_data['referenceId'] == product_id]['comment'].values.tolist()
            key= df_data[df_data['referenceId'] == product_id]['comment'].index.tolist()
            
            ui=dict(zip(key,value))
            dianpu_corpus[product_id]=ui
        corpus_dianpu[dianpu]=dianpu_corpus

##    #每一个店铺的corpus
    sum_dianpu=[]
    for dianpu,dianpu_comm in corpus_dianpu.items():
        ui=0
        for product_id,comm in dianpu_comm.items():
            ui+=len(dianpu_comm[product_id])
        sum_dianpu.append(ui)
    #求每一个店铺的情感
#    #得到不同店铺下，不同商品的评论个数
    kl=pd.DataFrame(columns=['shop_name','good_id','comment_num'])
    table=PrettyTable(columns=['shop_name','good_id','comment_num'])
    i=0
    for dianpu,dianpu_comm in corpus_dianpu.items():
        for product_id in dianpu_comm.keys():
            if len(dianpu_comm[product_id])>100:
                table.add_row([dianpu,product_id,len(dianpu_comm[product_id])])
                kl.loc[i]=[dianpu,str(product_id),len(dianpu_comm[product_id])]
                i+=1
    print(table)
##
##    #开始计算上述店铺的情感得分
##    #1021444044
    df=pd.DataFrame(columns=['店铺','商品id','score'])
    df['店铺']=['希诺官方旗舰店' for _ in range(3)]+['特美刻京东自营旗舰店' for _ in range(2)]+['万象京东自营旗舰店']+[\
        '希诺（HEENOOR）京东自营旗舰店' for _ in range(5)]+['优道京东自营旗舰店']+[\
            '富光京东自营旗舰店' for _ in range(3)]+['天喜京东自营旗舰店']
    df['商品id']=[1021444044,11289532859,32861622052,1206182,1206154,3011854,\
                4971449,4868161,6798151,4868307,5937314,3868923,2169821,5649596,2278718,6577674]
    for row,data in df.iterrows():
        try:
            product=data['商品id']
            dianpu=data['店铺']
            corpus_dp1=corpus_dianpu[dianpu][product]
            score_list=[]
            length_list=[]
            all_length=0
            for pinglun_id,pinglun in corpus_dp1.items():
                ui=comment_train[pinglun_id]
                score=get_sentiment_score(ui,pinglun_id)
                score_list.append(score)
                all_length+=len(ui)
                length_list.append(len(ui))
            all_score=sum([score_list[i] for i in range(len(corpus_dp1))])/len(corpus_dp1)
            print('score:',all_score)
            df.iloc[row]=[data['店铺'],data['商品id'],all_score]
        except Exception as result:
            print('result',result)


#获取这些商品中每一个score的个数




    #情感倾向
##    #计算每一个商品不同评论的情感倾向百分比，
##    import matplotlib.pyplot as plt
##    plt.rcParams['font.sans-serif']=['SimHei']
##    plt.rcParams['axes.unicode_minus']=False
##    def plot(product_id,dianpu,op):
##        dp_corpus=corpus_dianpu[dianpu][product_id]
##        dp=pd.DataFrame(columns=['评论编号','好评比例'])
##        i=0
##        for id_pl,pl in dp_corpus.items():
##            dl=P.doc_sel[id_pl,:]
##            max_value=round(dl[1],3)
##            dp.loc[i]=[id_pl,max_value]
##            i+=1
##        sns.distplot(dp['好评比例'],bins=[-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2],hist=False,label=dianpu+str(op))
##    
##    plot(1021444044,'希诺官方旗舰店',1)
##    plot(11289532859,'希诺官方旗舰店',2)
##    plot(32861622052,'希诺官方旗舰店',3)
##    plt.legend()
##    plt.savefig('C:/Users/Administrator/Desktop/论文/picture/希诺官方')
##    plot(1206182,'特美刻京东自营旗舰店',1)
##    plot(1206154,'特美刻京东自营旗舰店',2)
##    plt.savefig('C:/Users/Administrator/Desktop/论文/picture/特美刻官方')
##    plot(3011854,'万象京东自营旗舰店',1)
##    plt.savefig('C:/Users/Administrator/Desktop/论文/picture/万象')
##    plot(4971449,'希诺（HEENOOR）京东自营旗舰店',1)
##    plot(4868161,'希诺（HEENOOR）京东自营旗舰店',2)
##    plot(6798151,'希诺（HEENOOR）京东自营旗舰店',3)
##    plot(4868307,'希诺（HEENOOR）京东自营旗舰店',4)
##    plot(5937314,'希诺（HEENOOR）京东自营旗舰店',5)
##    plt.savefig('C:/Users/Administrator/Desktop/论文/picture/希诺京东')
##    #,'特美刻京东_1206182','特美刻_1206154'])
####                '万象京东_3011854','希诺京东_4971449','希诺京东_4868161','希诺京东_6798151',\
####                '希诺京东_4868307','希诺京东_5937314','优道京东_3868923'])
##    plt.xlabel('好评比例')
##    plt.title('各大商品的好评比例直方图')
##    plt.show()
##
##    #真实的好评度
##    mm=[0.98,0.99,0.98,0.94,0.90,0.99]
##    lo=[0.99,0.98,0.99,0.98,0.99]

    #求word2vec

##    model=word2vec.Word2Vec(corpus_total,size=200,window=5,min_count=1,workers=5)
##    
##    vectorizer = CountVectorizer()
##    transformer = TfidfTransformer()
##    tfidf = transformer.fit_transform(vectorizer.fit_transform(sentence))
##    weight_doc=coo_matrix(tfidf.toarray())
##    word_list_c=vectorizer.vocabulary_
##    word_weight=weight_doc.T
    #获取原始数据集的标签信息
    
##    corpus_total=data['正面']+data['反面']
##    label_total=[1]*len(data['正面'])+[2]*len(data['反面'])
##    pre_label=[]
##    #获取预测得到的标签信息
##    dos=P.doc_sel
##    for i in range(len(dos)):
##        ui=dos[i,1:3]
##        max_index=np.argmax(ui)
##        pre_lable.append(max_index+1)
##
##    #1.准确度、召回率、F1值、得到不同主题个数下的召回情况
##    from sklearn.metrics import accuracy_score
##    from sklearn import metrics
##    print(accuracy_score(y_true, y_pred))
##    metrics.recall_score(y_true, y_pred, average='macro')
##    metrics.f1_score(y_true, y_pred, average='weighted')
##
##    from sklearn.metrics import classification_report
##    target_names = ['class 0', 'class 1', 'class 2']
##    print(classification_report(y_true, y_pred, target_names=target_names))

    
    
        
        
        

        
        
    
