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
    path_neg = 'C:/Users/Administrator/Desktop/data/corpus/train_neg_cup_corpus.txt'
    path_pos = 'C:/Users/Administrator/Desktop/data/corpus/train_pos_cup_corpus.txt'
    all_text_neg = []
    all_text_pos = []
    with open(path_neg, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
            ui=[]
            #print(lines)
            for word_flag in lines:
                word,flag=word_flag.strip().split('_')
                ui.append(word)
            all_text_neg.append(ui)
        f.close()
    with open(path_pos, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines = line.lstrip('\ufeff').rstrip('\n').split(' ')
            ui=[]
            for word_flag in lines:
                word,flag=word_flag.split('_')
                ui.append(word)
            all_text_pos.append(ui)
        f.close()
    all_text = all_text_neg + all_text_pos
    print(all_text[0])
    #comment_train, comment_test = train_test_split(all_text, test_size=0.1)
    comment_train, comment_test = train_test_split(all_text, test_size=0.15)
    M = ldamodel_nosent(10, 0.1,10000,0.1,comment_train, 120)
    M.createdictionary()
    M.initial()
    start = time.time()
    M.gibbssampling()
    end = time.time()

    #进行聚类
    from gensim import models,corpora
    from sklearn.cluster import KMeans
    sentences=[]
    for doc in comment_train:
        sentences.append(' '.join(doc))
    dictionary = corpora.Dictionary(comment_train)
    corpus = [dictionary.doc2bow(text) for text in comment_train]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    L=models.LdaModel(corpus=corpus,num_topics=150,id2word=dictionary)
    
    num_show_topic = 9  # 每个文档显示前几个主题
    print('下面，显示前9个文档的主题分布：')
    doc_topics = L.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    for i in range(9):
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        topic_idx = list(topic_distribute)
        print('第%d个文档的 %d 个主题分布概率分别为：' % (i, num_show_topic))
        print(topic_idx)

    num_show_term = 10   # 每个主题下显示几个词
    table=PrettyTable()
    for topic_id in range(num_topics):
        print('第%d个主题的词与概率如下：\t' % topic_id)
        term_distribute_all = L.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：\t', end='  ')
        for t in term_id:
            print(dictionary.id2token[t], end=' ')
        print('\n概率：\t', term_distribute[:, 1])
    from pylab import *
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    for i, k in enumerate(range(4)):
        ax = plt.subplot(2, 2, i+1)
        item_dis_all = L.get_topic_terms(topicid=k)
        item_dis = np.array(item_dis_all[:num_show_term])
        ax.plot(range(num_show_term), item_dis[:, 1], 'b*')
        item_word_id = item_dis[:, 0].astype(np.int)
        word = [dictionary.id2token[i] for i in item_word_id]
        ax.set_ylabel(u"概率")
        for j in range(num_show_term):
            ax.text(j, item_dis[j, 1], word[j], bbox=dict(facecolor='green',alpha=0.1))
    plt.suptitle(u'9个主题及其7个主要词的概率', fontsize=18)
    plt.show()


    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        doc_item = np.array(doc_topics[i])
        doc_item_id = np.array(doc_item[:, 0])
        doc_item_dis = np.array(doc_item[:, 1])
        ax.plot(doc_item_id, doc_item_dis, 'r*')
        for j in range(doc_item.shape[0]):
            ax.text(doc_item_id[j], doc_item_dis[j], '%.3f' % doc_item_dis[j])
    plt.suptitle(u'前4篇文档的主题分布图', fontsize=18)
    plt.show()
##data=pd.DataFrame(columns=['s0','s1','s2','s3'])
##>>> data['s0']=[0.943,0.002,0.001,0.001,0.904,0.001,0.827,0.003,0.002,0.002]
##>>> data['s1']=[0.056,0.993,0.932,0.996,0.048,0.998,0.173,0.99,0.994,0.994]
##>>> data['s2']=[0.001,0.002,0.001,0.001,0.048,0.001,0.0,0.003,0.002,0.002]
##>>> data['s3']=[0.001,0.002,0.067,0.001,0.0,0.001,0.0,0.003,0.002,0.002]
    i=0
    for row,do in data.iterrows():
        ax=plt.subplot(2,2,i+1)
        ax.plot(list(range(0,4)),do,'r*')
        for j in range(4):
            ax.text(j,do[j],'s{0}'.format(j))
        i+=1
        if i>4:
            break
    plt.suptitle(u'前4篇文档的情感分布图', fontsize=18)
    plt.savefig('C:/Users/Administrator/Desktop/论文/picture/jst_sentiment')
    #得到topic_word矩阵
    topic_word=np.zeros((L.num_topics,L.num_terms))
    for j in range(L.num_terms):
            ui=L.get_term_topics(j)
            for topic_par in ui:
                topic_word[topic_par[0]]=topic_par[1]
    #得到第47个主题的
    term_distribute_all = L.get_topic_terms(topicid=47)
    term_distribute = term_distribute_all[:num_show_term]
    term_distribute = np.array(term_distribute)
    term_id = term_distribute[:, 0].astype(np.int)
    print('词：\t', end='  ')
    for t in term_id:
        print(dictionary.id2token[t], end=' ')
    print('\n概率：\t', term_distribute[:, 1])
    
    km = KMeans(n_clusters=2, random_state = 666)   
    y_pre = km.fit_predict( topic_word)
    for j in range(len(y_pre)):
            ax.text(j, topic_word[j,1], f'topic{0}'.format(j), bbox=dict(facecolor='green',alpha=0.1))
    plt.axis('off')
    plt.show()
