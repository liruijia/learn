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



class ldamodel():
    def __init__(self,topic_num,sentiment_num,alpha,beta,gamma,corpus,interation):
        self.T=topic_num
        self.D=None
        self.S=sentiment_num
        self.V=None
        
        self.alpha=alpha if alpha else 0.1
        self.beta= beta if beta else 0.1
        self.gamma=gamma if gamma else 0.1 
        self.corpus=corpus
        self.interation= interation if interation else 1000

        #设置参数
        self.doc_sel_topic_count=None
        self.topic_sel_word_count=None
        self.doc_count=None
        self.topic_count=None
        self.sentiment_count=None
        self.topic_sentiment_count=None
        self.doc_sentiment_count=None
        

        self.doc_sel_topic=None
        self.topic_sel_word=None
        self.doc_sel=None
        
        self.z=None
        self.l=None


        

    def createdictionary(self,cut_corpus):
        word2id=dict()
        wordnum=0
        cut_doc_id = copy.deepcopy(cut_corpus)
        for i , doc in enumerate(cut_corpus):
            for j,word in enumerate(doc):
                wordnum+=1   #记录了所有单词的个数
                if word not in word2id.keys():
                    word2id[word]=len(word2id)
                cut_doc_id[i][j]=word2id[word]
        self.V=wordnum
        self.D=len(cut_corpus)
        return word2id,dict(zip(word2id.values(),word2id.keys())),cut_doc_id,wordnum 
    
      
  
    def initial(self,cut_doc_id):
        self.doc_sel_topic_count=np.zeros([self.D,self.S,self.T])
        self.topic_sel_word_count=np.zeros([self.S,self.T,self.V])
        self.doc_count=np.zeros(self.D)
        self.sentiment_count=np.zeros(self.S)  ##这个不是必须的
        self.topic_count=np.zeros(self.T)      ##这个不是必须的
        self.topic_sentiment_count=np.zeros([self.T,self.S])
        self.doc_sentiment_count=np.zeros([self.D,self.S])

        self.doc_sel_topic=np.ndarray([self.D,self.S,self.T])
        self.topic_sel_word=np.ndarray([self.S,self.T,self.V])
        self.doc_sel=np.ndarray([self.D,self.S])
        
        self.z=np.zeros([self.D,self.V])  #存放每个文档中每一个词的主题
        self.l=np.zeros([self.D,self.V])  #存放每个文档中每个词的情感极性

        for i,doc in enumerate(cut_doc_id):
            for j,word_id in enumerate(doc):
                topic=random.randint(0,self.T-1)
                sentiment=random.randint(0,self.S-1)
                
                self.doc_sel_topic_count[i,sentiment,topic]+=1
                self.topic_sel_word_count[sentiment,topic,word_id]+=1
                self.doc_count[i]+=1
                self.sentiment_count[sentiment]+=1
                self.topic_count[topic]+=1
                self.topic_sentiment_count[topic,sentiment]+=1
                self.doc_sentiment_count[i,sentiment]+=1
                
                
                self.z[i,word_id]=topic
                self.l[i,word_id]=sentiment

    


    def gibbssampling(self,cut_doc_id):
        for iter in range(self.interation):
            for i,doc in enumerate(cut_doc_id):
                for j,word_id in enumerate(doc):
                    topic=int(self.z[i,word_id])
                    sentiment=int(self.l[i,word_id])
              
                    n_jkd=self.doc_sel_topic_count[i,sentiment,topic]-1
                    n_jkw=self.topic_sel_word_count[sentiment,topic,word_id]-1
                    n_jk=self.topic_sentiment_count[topic,sentiment]-1
                    n_kd=self.doc_sentiment_count[i,sentiment]-1
                    n_d=self.doc_count[i]-1 
                                       
                    new_topic,new_sentiment=self.resampling(n_jkd,n_jkw,n_jk,n_kd,n_d)

                    self.z[i,word_id]=new_topic
                    self.l[i,word_id]=new_sentiment
              
                    self.doc_sel_topic_count[i,sentiment,topic]-=1
                    self.topic_sel_word_count[sentiment,topic,word_id]-=1
                    self.topic_sentiment_count[topic,sentiment]-=1
                    self.doc_sentiment_count[i,sentiment]-=1
                    self.topic_count[topic]-=1
                    self.sentiment_count[sentiment]-=1
              

                    self.doc_sel_topic_count[i,new_sentiment,new_topic]+=1
                    self.topic_sel_word_count[new_sentiment,new_topic,word_id]+=1
                    self.topic_sentiment_count[new_topic,new_sentiment]+=1
                    self.doc_sentiment_count[i,new_sentiment]+=1
                    self.topic_count[new_topic]+=1
                    self.sentiment_count[new_sentiment]+=1
            print('已经迭代到了第{0}次了'.format(iter))
            
        self.updateparam()
    ###验证模型


    def updateparam(self):
        for i in range(self.D):
            for j in range(self.S):
                self.doc_sel[i,j]=(self.doc_sel_count[i,j] + self.gamma)/(self.doc_count[i] + self.S * self.gamma)

        for i in range(self.S):
            for  j in range(self.T):
                for k in range(self.V):
                    self.topic_sel_word[i,j,k]=(self.topic_sel_word_count[i,j,k] + self.beta)/(
                        self.topic_sentiment_count[j,i] + self.beta * self.V)
        for i in range(self.D):
            for j in range(self.S):
                for k in range(self.T):
                    self.doc_sel_topic[i,j,k]=(self.doc_sel_topic_count[i,j,k] + self.alpha)/(
                        self.doc_sentiment_count[i,j] + self.T * self.alpha)
        print('参数更新完成******************* \n')
        return

    
                    
 
    def resampling(self,n_jkd,n_jkw,n_jk,n_kd,n_d):  
        pk = np.ndarray([self.T,self.S]) 
        for i in range(self.T):
            for j in range(self.S):
                pk[i,j] = float(n_jkd + self.alpha)*(n_jkw +self.beta)*(n_kd + self.gamma)/(
                  (n_kd + self.alpha*self.T)*(n_jk + self.beta*self.V)*(n_d + self.gamma * self.S))
                if i>0 and j>0:
                    pk[i,j]+=pk[i-1,j]
        # 轮盘方式随机选择主题
        u = random.random()*pk[self.T-1,self.S-1]
        for j in range(self.T):
            for  k in range(self.S):
                if pk[j,k]>=u:
                    return [j,k]

    def predict(self,new_doc):
        '''
            predict:new doc / comment
        '''
        #对新文档进行切分等处理
        new_doc_cut=jieba.lcut(new_doc)  

        #获取新文档中在word2id中存在的单词
        new_doc_id=list()
        for word in new_doc_cut:
            if word in self.word2id:
                new_doc_id.append(self.word2id[word])
        
        #参数的设置  涉及到文档的矩阵需要重新设置一个新的，其余的不变
        new_dstc=np.zeros([1,self.S,self.T]) 
        new_dsc=np.zeros([1,self.S])
        new_dc=0
        new_tswc=copy.deepcopy(self.topic_sel_word_count)
        new_sc=copy.deepcopy(self.setiment_count)
        new_tsc=copy.deepcopy(self.topic_sentiment_count)
        new_tc=copy.deepcopy(self.topic_count)
        
        new_z=np.zeros([1,self.V])
        new_l=np.zeros([1,self.S])

        #参数的更新，和之前的过程类似
        for i,word_id in enumerate(new_doc_id):
            topic=self.z[0,word_id]
            sentiment=self.l[0,word_id]
            
            new_dstc[0,sentiment,topic]+=1
            new_dsc[0,sentiment]+=1
            new_dc+=1
            new_tswc[sentiment,topic,word_id]+=1
            new_sc[sentiment]+=1
            new_tsc[topic,sentiment]+=1
            new_tc[topic]+=1

            new_z[0,word_id]=topic
            new_l[0,word_id]=sentiment
            
        
        resampling(self,n_jkd,n_jkw,n_jk,n_kd,n_d)    
        
        #开始进行采样了
        for iter in range(0,self.interation):
            for word_id in new_doc_id:
                topic=new_z[0,word_id]
                sentiment=new_l[0,word_id]
                
                n_jkd=new_dstc[0,sentiment,topic]-1
                n_jkw=new_tswc[sentiment,topic,word_id]-1
                n_jk=new_tsc[sentiment,topic]-1
                n_kd=new_dsc[0,sentiment]-1
                n_d=new_dc[0]-1

                #此处需要进行重新给每个该单词进行重新赋予主题
                new_topic,new_sentiment=self.resampling(self,n_jkd,n_jkw,n_jk,n_kd,n_d)   

                #更新旧的新的topic的值
                new_dstc[0,sentiment,topic]-=1 
                new_dsc[0,sentiment]-=1
                new_tswc[sentiment,topic,word_id]-=1
                new_sc[sentiment]-=1
                new_tsc[sentiment,topic]-=1
                new_tc[topic]-=1

                new_dstc[0,new_sentiment,new_topic]+=1 
                new_dsc[0,new_sentiment]+=1
                new_tswc[new_sentiment,new_topic,word_id]+=1
                new_sc[new_sentiment]+=1
                new_tsc[new_sentiment,new_topic]+=1
                new_tc[new_topic]+=1

                new_z[0,word_id]=new_topic
                new_l[0,word_id]=new_sentiment

            if (iter+1)%100==0:
                print('new_doc 第{0}次训练'.format(iter))
                #此时要输出LDA模型的评价标准
                  
        if isupdate==True:
            self.topic_sel_word_count=new_tswc
            self.sentiment_count=new_sc
            self.topic_sentiment_count=new_tsc
            self.topic_count=new_tc
            self.doc_sel_topic_count=np.r_[self.doc_sel_topic_count,new_dstc]
            self.doc_sentiment_count=np.r_[self.doc_sentiment_count,new_dsc]
            self.doc_count=np.r_[self.doc_count,new_dc]
            self.updateparam()
            print('加载new_doc之后选择更新参数，并更新完成')
        else:
            print('选择不更新参数')     
        return [new_dstc,new_tswc,new_dsc,new_dc,new_tc,new_tsc,new_sc]

    def get_top_word(self,topnums=20):
        '''打印出来每个主题与其概率最高词语的组合--等式
    将每一个topic的高频单词读取出来并保存'''
        with open('./content/top_word','w') as f:
          for i in range(0,self.K):
            top_words=np.argsort(self.topic_word[i,:])[:topnums]
            top_word=[self.id2word[j] for j in top_words]
            top_words = '\t'.join(top_words)
            res = 'topic{0}: \t {1}'.format(i, top_words)
            f.write(res+'\n')
            #print(res)
  
    def get_top_topic(self,topicnums=20,wordnums=20):
        with open('./concent/top_topic_word') as f:
            for doc in range(self.D):
                top_topic=np.argsort(self.doc_topic[doc,:])[:topicnums]
                res='doc:{0}\t'.format(doc)
                f.write(res)
                for theam in top_topic:
                  topword=np.argsort(self.topic_word[theam,:])[:wordnums]
                  topword=[self.id2word[j] for j in topword ]
                  re='\t'.join(topword)
                  res='topic:{0} \t {1}'.format(theam,re)
                  f.write(re+'\n')
        f.close()
        return 

    def print_topic_word(self,doc_id,topic_list=list(),word_nums=20):
        all_num=len(topic_list)
        table=PrettyTable()
        for i in topic_list:
          topword=np.argsort(self.topic_word[i,:])[:word_nums]
          table.add_column(i, [self.id2word[jj] for jj in topword])
        print(table)
        
        #打印出来该文档上的主题分布以及在每个主题上面的个数的图形
        doc_topic_count=self.doc_topic_count[doc_id,:]
        sns.stripplot(x=list(range(0,all_num-1)),y=doc_topic_count)
        for i in topic_list:
          sns.scatterplot(x=range(0,self.V-1),y=self.topic_word[i,:])
          plt.show()
          sns.countplot(x=range(0,self.V-1),hue=self.topic_word[i,:]) 
          plt.show()
    
  
if __name__=='__main__':
    stopwords_path='../论文/中文停用词/stopwords'
    path='C:/Users/Administrator/Desktop/data/评论/cut_comment.txt'
    all_text=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            lines=line.strip().split(' ')
            all_text.append(lines)
        f.close()
    comment_train, comment_test = train_test_split(all_text, test_size = 0.7)
    
    M=ldamodel(20,5,0.1,0.1,0.1,comment_train,100)
    word2id,id2word,cut_corpus_id,wordnum=M.createdictionary(comment_train)
    M.initial(cut_corpus_id)
    M.gibbssampling(cut_corpus_id)
    






        
        


        
