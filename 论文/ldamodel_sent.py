'''
需要注意的是情感大分类这个地方和我们模型中的情感的个数是不一致的，所有的词语都可以分为这7种类型
但是我们在进行情感分析的时候有的时候只需要其中的几类就可以比如说是，只研究情感极性
或者只研究喜怒哀乐这4个方面，在使用模型的时候还得看情况使用
大分类：按照文件来的
极性：褒义、贬义、中性、间有  0:中性，1：褒义，2：贬义，3：兼有
情感态度：正面 反面： 0：正面  1：反面
情感表达：喜怒哀乐  1--表示喜  2----表示怒，3------表示哀 4 ------表示乐

需要注意的是我们找到在训练LDA模型的时候，我们通过其情感大分类则可以知道其相应的情感大分类——表达
以及情感分类——态度   ，无须保留
这些值如何更新？？？？我们在训练的时候可以利用词典赋予新的情感标签（如果能找到的话则使用若找不到，则随机赋予），但是在采样过程中，
得到了新的情感标签的时候，我们可以要更新这个词在词库里的信息，最后进行保存，查看区别！！！！
对于那些没有的词我们也要保留其信息，进行后期验证
强度没有办法进行更新只能每次相应的修改
'''

import sys
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
from gensim.models  import word2vec
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append('G:/anconada/envs/py36/lib/site-packages')

''' 在写论文的时候可以使用两组数据查看模型的效果'''


class sentiment_dict():
    def __init__(self):
       print('开始处理情感词典')

    def get_data(self,path_list):
        final_data0=[]
        #final_corpus=[]
        data1 = pd.read_csv(path_list[1], engine='python')
        final_data0.append(data1['词语'].tolist())

        for path in path_list[2:]:
            final_data0.append(self._load_senti_dict(path))

        with open(path_list[0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                lines = line.strip().split(' ')
                final_data0.append(lines)
            f.close()

        with open('C:/Users/Administrator/Desktop/data/评论/final_corpus.txt', 'w', encoding='utf-8') as f:
            for line in final_data0:
                f.write(' '.join(line))
                f.write('\n')
            f.close()
        print('所有情感词典加载完毕*******************')
        return final_data0

    def _load_senti_dict(self,path):
        data=[]
        with open(path,'r',encoding='utf-8') as f :
            for line  in f.readlines():
                if line[0].isdigit():
                    continue
                else:
                    if line != '\n':
                        data.extend(line.strip().split(' '))
            f.close()
        data.pop(0)
        print(path[-15:]+'加载完毕**************')
        return data

    def load_amend_dict(self,data_1, final_data0):
        final_data = dict()
        dict_word = data_1['词语'].tolist()
        # print(data_2[:30])
        for i, doc in enumerate(final_data0):
            for j, word in enumerate(doc):
                if word in final_data.keys() or word == ' ' or word == '':
                    continue
                else:
                    final_data[word] = []
                    if word in dict_word:
                        index = dict_word.index(word)
                        final_data[word] = data_1[['情感大分类', '强度']].loc[index].tolist()
                    else:
                        # print('word :{0} not in vocabulary'.format(word))
                        sim_word = model.most_similar(word, topn=1)[0]
                        if sim_word in dict_word:
                            final_data[word] = final_data[sim_word]
                            # print('word:{0} can find similar word :{1}'.format(word,sim_word))
                        else:
                            # print('word:{0} can not find similar word'.format(word))
                            final_data[word] = [4, 3]
        print('词语词典建立完成*************')
        return final_data


class ldamodel():
    def __init__(self,topic_num,sentiment_num,alpha,beta,gamma,corpus,interation,final_info_sentword,df_info):
        self.T=topic_num
        self.D=None
        self.S=sentiment_num
        self.V=None

        #这两个词典用来指定词语的情感标签，若我们想要评论的喜怒哀乐（此时S=4）则使用df_info 中的情感大分类_表达作为该词的情感标签
        #若想考查评论的态度（即正面评论还是反面评论，此时S=2）,则使用df_info中的情感大分类_态度作为该词的情感标签

       self.sentiment_dict=final_info_sentword
        self.sentiment_map=df_info

        #超参数设置
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
            print('已经迭代到了第{0}次了'.format(iter+1))
            
        self.updateparam()
    ###验证模型


    def updateparam(self):
        for i in range(self.D):
            for j in range(self.S):
                self.doc_sel[i,j]=(self.doc_sentiment_count[i,j] + self.gamma)/(self.doc_count[i] + self.S * self.gamma)

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
                    pk[i,j]+=pk[i,j-1]
        # 轮盘方式随机选择主题
        u = random.random()*pk[self.T-1,self.S-1]
        for j in range(self.T):
            for  k in range(self.S):
                if pk[j,k]>=u:
                    #print('get the new topic {0} and new sentiment {1}'.format(j,k))
                    return j,k

    def predict(self,new_doc,word2id,isupdate=False):
        '''
            predict:new doc / comment
        '''
        #对新文档进行切分等处理

        #获取新文档中在word2id中存在的单词
        new_doc_id=list()
        for word in new_doc:
            if word in word2id:
                new_doc_id.append(word2id[word])
        
        #参数的设置  涉及到文档的矩阵需要重新设置一个新的，其余的不变
        new_dstc=np.zeros([1,self.S,self.T]) 
        new_dsc=np.zeros([1,self.S])
        new_dc=0
        new_tswc=copy.deepcopy(self.topic_sel_word_count)
        new_sc=copy.deepcopy(self.sentiment_count)
        new_tsc=copy.deepcopy(self.topic_sentiment_count)
        new_tc=copy.deepcopy(self.topic_count)
        
        new_z=np.zeros([1,self.V])
        new_l=np.zeros([1,self.V])

        #参数的更新，和之前的过程类似
        for i,word_id in enumerate(new_doc_id):
            topic=int(self.z[0,word_id])
            sentiment=int(self.l[0,word_id])
            
            new_dstc[0,sentiment,topic]+=1
            new_dsc[0,sentiment]+=1
            new_dc+=1
            new_tswc[sentiment,topic,word_id]+=1
            new_sc[sentiment]+=1
            new_tsc[topic,sentiment]+=1
            new_tc[topic]+=1

            new_z[0,word_id]=topic
            new_l[0,word_id]=sentiment
            
        
       
        
        #开始进行采样了
        for iter in range(0,self.interation):
            for word_id in new_doc_id:
                topic=int(new_z[0,word_id])
                sentiment=int(new_l[0,word_id])
                
                n_jkd=new_dstc[0,sentiment,topic]-1
                n_jkw=new_tswc[sentiment,topic,word_id]-1
                n_jk=new_tsc[sentiment,topic]-1
                n_kd=new_dsc[0,sentiment]-1
                n_d=new_dc-1

                #此处需要进行重新给每个该单词进行重新赋予主题
                new_topic,new_sentiment=self.resampling(n_jkd,n_jkw,n_jk,n_kd,n_d)   

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
                print('new_doc 第{0}次训练'.format(iter+1))
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
        print('输出参数')
        print(new_dstc)
        print(new_tswc)
        print(new_dsc)
        print(new_dc)
        print(new_tc)
        print(new_tsc)
        print(new_sc)
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
        with open('./concent/top_topic_word','w') as f:
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

    def print_topic_word(self,doc_id,topic_list,word_nums=20):
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
    path0 = 'C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
    path1 = 'C:/Users/Administrator/Desktop/data/情感词汇本体/情感词汇本体.csv'
    path2 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面情感词语（中文）.txt'
    path3 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面情感词语（中文）.txt'
    path4 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/程度级别词语（中文）.txt'
    path5 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面评价词语（中文）.txt'
    path6 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面评价词语（中文）.txt'

    path_list = [path0, path1, path2, path3, path4, path5, path6]

    # vector=CountVectorizer()
    # trans=TfidfTransformer()
    # tfidf = trans.fit_transform(vector.fit_transform(final_corpus))
    # word = vector.get_feature_names()  #
    # weight = tfidf.toarray()

    P = sentiment_dict()
    final_data0 = P.get_data(path_list)
    path01 = 'C:/Users/Administrator/Desktop/data/评论/final_corpus.txt'
    sentences = word2vec.Text8Corpus(path01)
    model = word2vec.Word2Vec(sentences, size=400, window=5, min_count=1)

    data1 = pd.read_csv(path1, engine='python')
    df_info = pd.DataFrame(columns=['情感分类'])
    df_info['情感分类'] = data1['情感分类'].unique().tolist()
    df_info['情感大分类'] = df_info['情感分类'].map({'PA': 1, 'PE': 1,
                                            'PD': 2, 'PH': 2, 'PG': 2, 'PB': 2, 'PK': 2,
                                            'NA': 3, 'NB': 4, 'NT': 4, 'NH': 4, 'PF': 4, 'NI': 5, 'NC': 5, 'NG': 5,
                                            'NE': 6, 'ND': 6, 'NN': 6, 'NK': 6, 'NL': 6, 'PC': 7})
    data1['情感大分类'] = data1['情感分类'].map({'PA': 1, 'PE': 1,
                                        'PD': 2, 'PH': 2, 'PG': 2, 'PB': 2, 'PK': 2,
                                        'NA': 3, 'NB': 4, 'NT': 4, 'NH': 4, 'PF': 4, 'NI': 5, 'NC': 5, 'NG': 5,
                                        'NE': 6, 'ND': 6, 'NN': 6, 'NK': 6, 'NL': 6, 'PC': 7})
    df_info['情感大分类_表达'] = df_info['情感大分类'].map({2: 1, 3: 2, 6: 2, 4: 3, 5: 3, 7: 3, 1: 4})
    df_info['情感大分类_态度'] = df_info['情感大分类'].map({1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})

    final_info_sentword = P.load_amend_dict(data1, final_data0)
    print(final_info_sentword['OPPO'])
    items = model.most_similar(u'好评', topn=20)
    print('“好评”一词的相似的词语')
    for word, sim_par in items:
        print(word, sim_par)


    stopwords_path='../论文/中文停用词/stopwords'
    path='C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
    all_text=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            lines=line.strip().split(' ')
            all_text.append(lines)
        f.close()
    comment_train, comment_test = train_test_split(all_text, test_size = 0.5)
    
    M=ldamodel(20,5,0.1,0.1,0.1,comment_train,100)
    word2id,id2word,cut_corpus_id,wordnum=M.createdictionary(comment_train)
    M.initial(cut_corpus_id)
    start=time.time()
    M.gibbssampling(cut_corpus_id)
    end=time.time()
    print('gibbssampling stage use {0} second'.format(end-start))
    test0=comment_test[0]
    M.predict(test0,word2id)






    

            






        
        


        
