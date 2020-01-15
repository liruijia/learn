import sys
sys.path.append('G:/anconada/envs/py36/lib/site-packages')
from prettytable import PrettyTable
import re
import jieba
import os
import copy
from zhon.hanzi import punctuation
from scipy.sparse import coo_matrix
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
from gensim.models  import word2vec
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
import psutil

class ldamodel_nosent():
    def __init__(self, topic_num, alpha, batch_size,beta, cut_corpus, interation):
        self.T = topic_num
        self.D = None
        self.V = None

        # 超参数设置
        self.alpha = alpha if alpha else 0.1
        self.beta = beta if beta else 0.1
        self.interation = interation if interation else 1000
        self.batch_size=batch_size
        self.word2id=None
        self.id2word=None
        self.dfs=None
        self.cut_corpus=cut_corpus
        # 设置参数
        self.doc_topic_count = None
        self.topic_word_count = None
        self.doc_count = None
        self.topic_count = None


        self.doc_topic=None
        self.topic_word=None

        self.z = None


    def createdictionary(self):
        #cut_corpus ---是cut_comment_1 ---加载进来的的双层列表
        #dic=corpora.Dictionary(documents=self.cut_corpus)
        #self.word2id=dic.token2id
        #self.id2word=dict(zip(self.word2id.values(),self.word2id.keys()))
        #print(len(self.id2word) , len(self.word2id))
        self.word2id={}
        self.V=0
        self.D=0
        wordnum=0
        for i ,doc in enumerate(self.cut_corpus):
            
            for j ,word in enumerate(doc):
                
                if word not in self.word2id.keys():
                    wordnum+=1
                    self.word2id[word]=len(self.word2id)
                
        self.V=wordnum
        self.id2word=dict(zip(self.word2id.values(),self.word2id.keys()))
        self.D=len(self.cut_corpus)
        print('语料库中总共有{0} 个token'.format(self.V))
        print('word2id的长度：',len(self.word2id))
        print('id2word的长度',len(self.id2word))
        
                
                
        
        


    def initial(self):
        self.doc_topic_count = np.zeros([self.D, self.T])
        self.topic_word_count = np.zeros([self.T, self.V])
        self.doc_count = np.zeros(self.D)
        self.topic_count = np.zeros(self.T)  ##这个不是必须的

        self.doc_topic=np.ndarray([self.D,self.T])
        self.topic_word = np.ndarray([self.T, self.V])

        self.z = coo_matrix((self.D, self.V),dtype=np.int8).toarray()  # 存放每个文档中每一个词的主题


        for i, doc in enumerate(self.cut_corpus):
            #此时cut_corpus 含义同上
            for j, word in enumerate(doc):
                word_id=int(self.word2id[word])
                topic = random.randint(0, self.T - 1)


                self.doc_topic_count[i, topic] += 1
                self.topic_word_count[ topic, word_id] += 1
                self.doc_count[i] += 1
                self.topic_count[topic] += 1


                self.z[i, word_id] = topic

    def _gibbsampling(self,batch_corpus,doc_topic_count,topic_word_count,topic_count,z):
        for i, doc in enumerate(batch_corpus):
            for j, word in enumerate(doc):
                word_id = self.word2id[word]
                topic = int(self.z[i, word_id])

                n_dtc = self.doc_topic_count[i, topic] - 1
                n_twc = self.topic_word_count[topic, word_id] - 1
                n_tc = self.topic_count[topic] - 1
                n_dc = self.doc_count[i] - 1

                new_topic = self.resampling(n_dtc, n_twc, n_tc, n_dc)
                #print(new_topic)
                z[i, word_id] = new_topic

                doc_topic_count[i, topic] -= 1
                topic_word_count[topic, word_id] -= 1
                topic_count[topic] -= 1

                doc_topic_count[i, new_topic] += 1
                topic_word_count[new_topic, word_id] += 1
                topic_count[new_topic] += 1
            
        return doc_topic_count,topic_word_count,topic_count,z


    def gibbssampling(self):
        for iter in range(self.interation):
            batch_corpus=random.sample(self.cut_corpus,self.batch_size)
            self.doc_topic_count,self.topic_word_count,self.topic_count,self.z=self._gibbsampling(batch_corpus,
                                                                                                  self.doc_topic_count,
                                                                                                  self.topic_word_count,
                                                                                                  self.topic_count,self.z)
            print('已经迭代到了第{0}次了'.format(iter + 1))
        self.updateparam()


    def updateparam(self):
        for i in range(self.T):
            for j in range(self.V):
                self.topic_word[i, j] = (self.topic_word_count[i, j] + self.beta) / (
                            sum(self.topic_word_count[i, :]) + self.beta * self.V)
        for i in range(self.D):
            for j in range(self.T):
                self.doc_topic[i, j] = float((self.doc_topic_count[i, j] + self.alpha) /(
                    sum(self.doc_topic_count[i,:]) + self.alpha * self.T))

        print('参数更新完成******************* \n')
        return


    def resampling(self, dtc, twc, tc, dc):
        #Gibbs采样公式
        pk=np.ndarray([self.T])
        for i in range(self.T):
          pk[i] = float(dtc + self.alpha)*(twc +self.beta)/((dc + self.alpha*self.T)*(tc + self.beta*self.V))
          if i > 0:
            pk[i] += pk[i-1]
        # 轮盘方式随机选择主题
        u = random.random()*pk[self.T-1]
        for k in range(len(pk)):
            if pk[k]>=u:
                return k
            else:
                return random.randint(0,self.T-1)




    def predict(self, new_doc, isupdate=False):
        '''
            predict:new doc / comment
        '''
        # 对新文档进行切分等处理

        # 获取新文档中在word2id中存在的单词
        new_doc_word = list()
        for i ,doc in enumerate(new_doc):
            ii=[]
            for word in doc:
                if word in self.word2id.keys():
                    ii.append(word)
            new_doc_word.append(ii)
        print(new_doc_word)
        # 参数的设置  涉及到文档的矩阵需要重新设置一个新的，其余的不变
        n=len(new_doc_word)
        new_dtc = np.zeros([n, self.T])
        new_dc = 0
        new_twc = copy.deepcopy(self.topic_word_count)
        new_tc = copy.deepcopy(self.topic_count)

        new_z = np.zeros([n, self.V])


        # 参数的更新，和之前的过程类似
        for i ,doc in enumerate(new_doc_word):
            for j, word in enumerate(doc):
                print(word)
                word_id=int(self.word2id[word])
                topic = int(self.z[i, word_id])

                new_dtc[i,  topic] += 1
                new_dc += 1
                new_twc[ topic, word_id] += 1
                new_tc[topic] += 1

                new_z[i, word_id] = topic

        # 开始进行采样了
        for iter in range(0, self.interation):
            new_dtc, new_twc,new_tc, new_z = self._gibbsampling(new_doc_word,new_dtc,new_twc,new_tc,new_z)

            print(' 预测阶段 new_doc 第{0}次训练'.format(iter + 1))
                # 此时要输出LDA模型的评价标准

        if isupdate == True:
            self.topic_word_count = new_twc
            self.topic_count = new_tc
            self.doc_topic_count = np.r_[self.doc_topic_count, new_dtc]
            self.doc_count = np.r_[self.doc_count, new_dc]
            self.updateparam()
            print('加载new_doc之后选择更新参数，并更新完成')
        else:
            print('选择不更新参数')
        print('输出参数')
        return [new_dtc, new_twc, new_dc, new_tc]


    def get_top_word(self, topnums=20):
        '''打印出来每个主题与其概率最高词语的组合--等式
    将每一个topic的高频单词读取出来并保存'''
        with open('C:/Users/Administrator/Desktop/data/top_word.txt', 'w') as f:
            for i in range(0, self.T):
                print('输出主题{0}的top_word'.format(i))
                top_words = np.argsort(self.topic_word[i, :])[:topnums]
                #print('输出的是每个topic的top_word的下标',top_words)
                top_word = [self.id2word[j] for j in list(top_words)]
                print(top_word)
                res = 'topic{0}: \t {1}'.format(i, top_word)
                f.write(res + '\n')
                # print(res)


    def get_top_topic(self, topicnums=20, wordnums=20):
        with open('./concent/top_topic_word', 'w') as f:
            for doc in range(self.D):
                top_topic = np.argsort(self.doc_topic[doc, :])[:topicnums]
                res = 'doc:{0}\t'.format(doc)
                f.write(res)
                for theam in top_topic:
                    topword = np.argsort(self.topic_word[theam, :])[:wordnums]
                    topword = [self.id2word[j] for j in topword]
                    re = '\t'.join(topword)
                    res = 'topic:{0} \t {1}'.format(theam, re)
                    f.write(re + '\n')
        f.close()
        return


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
    path ='C:/Users/Administrator/Desktop/data/中文情感分析语料库/pda/cut_pda_all.txt'
    all_text = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
            all_text.append(lines)
        f.close()
    print(len(all_text))
    #comment_train, comment_test = train_test_split(all_text, test_size=0.1)
    comment_train=all_text[:100]
    M = ldamodel_nosent(20, 0.1,50,0.1,comment_train, 1000)
    M.createdictionary()
    M.initial()
    start = time.time()
    M.gibbssampling()
    end = time.time()
    print('gibbssampling stage use {0} second'.format(end - start))
    test0 = [all_text[434]]
    print(test0)
    new_dtc, new_twc, new_dc, new_tc=M.predict(test0)
    M.get_top_word()
    info = psutil.virtual_memory()
    print('没有运行Gibbs sampling 之前的内存使用情况')
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())


