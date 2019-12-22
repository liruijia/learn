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



class Loaddata():
    def __init__(self):
        print('开始处理文本数据')
        
    def _loadstopwords(self,stopwords_path):
        '''停用词：融合网络停用词、哈工大停用词、川大停用词'''
        stop_words = set()
        with open(stopwords_path + u'/中文停用词库.txt','r',encoding='gbk') as fr:
            for line in fr.readlines():
                item = line.strip().split(' ')
                for it in item:
                    stop_words.add(it)
            print('中文停用词已经加载完成！！！***********')
            fr.close()
        with open(stopwords_path + u'/哈工大停用词表.txt','r',encoding='gbk') as fr:
            for line in fr.readlines():
                item = line.strip().split(' ')
                for it in item:
                    stop_words.add(it)
            print('哈工大停用词已经加载完成！！！************')
            fr.close()
        with open(stopwords_path + u'/四川大学机器智能实验室停用词库.txt','r',encoding='gbk') as fr:
            for line in fr.readlines():
                item = line.strip().split(' ')
                for it in item:
                    stop_words.add(it)
            print('四川大学实验室停用词加载完成！！！***************')
            fr.close()
        with open(stopwords_path + u'/百度停用词列表.txt','r',encoding='utf-8') as fr:
            for line in fr.readlines():
                item = line.strip().split(' ')
                for it in item:
                    stop_words.add(it)
            print('百度停用词已经加载完成！！！******************')
        with open(stopwords_path + u'/网络停用词.txt','r',encoding='utf-8') as fr:
            for line in fr.readlines():
                item = line.strip().split(' ')
                for it in item:
                    stop_words.add(it)
            print('网络停用词加载完成！！！！！************')
            fr.close()
        stop_words.add('')
        stop_words.add(' ')
        stop_words.add(u'\u3000')
        stop_words.add(u'日')
        stop_words.add(u'月')
        stop_words.add(u'时')
        stop_words.add(u'分')
        stop_words.add(u'秒')
        stop_words.add(u'报道')
        stop_words.add(u'新闻')
        stop_words.add(u'本文')
        stop_words.add(u'网易')
        stop_words.add(u'记者')
        stop_words.add(u'来源')
        stop_words.add(u'责任编辑')
        stop_words.add(u'王晓易')
        stop_words.add(u'新华网')
        stop_words.add(u'NE00111')
        stop_words.add(u'真是太')
        stop_words.add(u'金木水火土')
        stop_words.add(u'上次')
        stop_words.add(u'始终认为')
        stop_words.add(u'评论')
        stop_words.add(u'外形外观')
        print('所有的停用词加载完成')
        return stop_words

    def load_news(self,all_path,type_content):
        sentences=[]
        root_path='C:/Users/Administrator/Desktop/data/新闻/'
        i=0
        for path in all_path:
            text=[]
            f=open(root_path+type_content+'/'+path,'r',encoding='utf-8')
            for line in f.readlines():
                 lineData=line.strip().split(' ')
                 text.extend(lineData)
            text.pop(0)
            text.pop(0)
            text.pop(0)
            text.pop(0)
            text.pop(-1)
            text.pop(-1)
            text.pop(-1)
            text.pop(-1)
            s=''.join(text)
            sentences.append(s)
            #print('第{0}篇文章'.format(i))
            i+=1
            f.close()
        print('{0}篇{1}文章加载完成'.format(i,type_content))
        return sentences
    
    def load_comment(self,stopwords_path,path):
        stop_words=list(self._loadstopwords(stopwords_path))
        comment_info=pd.read_csv(path,engine='python')
        data_comment=comment_info['comment'].tolist()
        corpus=[]
        for c in data_comment:
            new_c=re.sub(r'[%s]+'%punctuation,' ',c)
            cut_c=jieba.lcut(new_c)
            new_doc=[]
            for word in cut_c:
                if len(word)==1:
                    continue
                if word not in stop_words and not word.isdigit():
                    new_doc.append(word)
            corpus.append(new_doc)
        f=open('C:/Users/Administrator/Desktop/data/评论/cut_comment.txt','w',encoding='utf-8')
        for i in corpus:
            f.write(' '.join(i) )
            f.write('\n')
        f.close()
        
        print('已经加载完毕评论形成corpus***************')
        return corpus
        
    def _load_corpus(self,sentences,stopwords_path):
        '''得到的 corpus是一个双层列表'''
        stop_words=list(self._loadstopwords(stopwords_path))
        corpus=[]
        for s in sentences:
            new_s=re.sub(r'[%s]+'%punctuation, " ", s)
            cut_s=jieba.lcut(new_s)
            new_doc=[]
            #print('文章 \n',sentence_cut)
            for word in cut_s:
                if len(word)==1:
                    continue
                if word not in stop_words and not word.isdigit():
                    new_doc.append(word)
            corpus.append(new_doc)
        print('文本已经去掉停用词以及数字')
        return corpus
    def _count_num(self,corpus,top_k=50,low_k=50):
        '''
        count the number of every word in corpus

        return:

           print top-k  and low-k  （and print those number by table）

           word_count: is a dict,the key is word,the value is the number 
           
        we should use low-k filtration the low-frequency words
        
        '''
        word_count=dict()
        for i ,doc in enumerate(corpus):
            for j,word in enumerate(doc):
                if word not in word_count.keys():
                    word_count[word]=word_count.get(word,0)+1
                else:
                    word_count[word]+=1

        sort_word=list(sorted(word_count.items(),key=lambda x :x[1],reverse=True))
        top_word=sort_word[:top_k]
        low_word=sort_word[-low_k:]
        
        top_table=PrettyTable(['word','number'])
        low_table=PrettyTable(['word','number'])
        for i in top_word:
            top_table.add_row(i)
        for i in low_word:
            low_table.add_row(i)

        print('the number of top_{0}  word  \n'.format(top_k))
        print(top_table)
            
        print('the number of low_{0}  word  \n'.format(low_k))
        print(low_table)

        return word_count
    def load_data(self,sentences,stopwords_path,top_k,low_k):
        '''
        filter the low-frequency-word
        '''
        all_text=self._load_corpus(sentences,stopwords_path)
        word_count=self._count_num(all_text,top_k=50,low_k=50)
        corpus=[] 
        for i,doc in enumerate(all_text):
            corpus.append([])
            for j ,word in enumerate(doc):
                if word_count[word]==1:
                    continue
                else:
                    corpus[-1].append(word)
        return corpus,word_count
    def getwordcloud(self,word_count,img_path):
        '''
            get word-cloud of corpus

            word_count : is a matrix of word-frequency
            
        '''
        color_mask = imread(img_path) #读取背景图片，
        cloud = WordCloud(font_path="simsun.ttc",mask=color_mask,background_color='white',max_words=400,max_font_size=100,width=1000,\
                          height = 500,margin = 10,prefer_horizontal = 0.8)
        # background_color='black'
        wc = cloud.generate_from_frequencies(word_count)
        #mm=img_path.replace('.jpg','词云.jpg')
        #wc.to_file(mm)
        image_colors = ImageColorGenerator(color_mask)
        plt.imshow(wc)
        plt.axis('off')
        plt.show()
        return

          
     

        
        
        

class get_dagword():
    def __int__(self):
        print('得到词袋矩阵')

    def getdagword(self,all_text):
        '''
        return :
        
            word2id--------is a dict ,key is word ,the value is serial number
            
            id2word--------is a dict ,key is the serial number of word , the value is word

            corpus :is a double list,the element of list represent the serial number of the ith doc and the jth word

            wordnum: the size of corpus ,the number of total token

        '''
        word2id=dict()
        wordnum=0
        corpus=copy.deepcopy(all_text)
        for i ,doc in enumerate(all_text):
            for j, word in enumerate(doc):
                wordnum+=1
                if word not in word2id.keys():
                    word2id[word]=len(word2id)
                corpus[i][j]=word2id[word]
        print('词袋矩阵加载完成**********************')
        return word2id,dict(zip(word2id.values(),word2id.keys())),corpus,wordnum

    
    '''
    init:
         T:---the number of topic
         S:---the number of sentiment
         
         alpha:----the param of DIR in document---topic stage
         beta: ----the param of Dir in generating a word stage
         gamma:----the param of dir in sentiment stage
         
         interation:----the number of interation
         
         doc_sel_topic_count:----a 3 dim matrix , every elenment represent the topic number of in document=d and sentiment=j condition
         topic_sel_word_count:---a 3 dim matrix, every elenment represent the  word number of in topic=k and sentiment =j condition
         doc_sel_count:-----a 2 dim matrix , every element represent the sel word number of in document=d condition

         doc_sel_topic: a 3 dim matrix and parability matrix
         topic_sel_word: a 3 dim matrix and parabilty matrix
         doc_sel : a 2 matrix and parabiliy matrix 
    '''

    '''
    createdictionary:
        get word2id id2word and wordnum  and cut_doc_id

        word2id :is a dict ,the key is word and the value is numbering

        id2word :is a dict ,the key is a numbering of word ,the value is a word

        cut_doc_id : is a bi-list where every element represent a numbering of word  for document=d in corpus

        wordnum: is a value ,representing  the total token in corpus
    '''
    '''
    initial:
        initial all param by cut_doc_id 
    '''
    '''
    gibbsampling:

        按照Gibbs采样公式有如下：
            下标j : 表示topic
            下标k : 表示sentiment
            下标d ：表示document
            下标w : 表示word
        以下几个值都除去了word=t 

        n_jkd   、n_jkw、n_kd
        n_jk   、 n_kd 、n_d

        前面3个是分子上的值，后面3个是分母上的值
        
        
    '''
    '''
    updateparma
            updating the parability matrix 
           seta------- self.doc_sel_topic=np.ndarray([self.D,self.S,self.T])
           fei-------- self.topic_sel_word=np.ndarray([self.S,self.T,self.V])
           pei-------- self.doc_sel=np.ndarray([self.D,self.S])
    '''
    '''
    resampling:
        real resampling  by equal

        input:
            n_jkd , n_jkw, n_dk ,n_jk , n_kd , n_d
        
        return :new_topic,new_sentiment

    '''
    











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
                    topic=self.z[i,word_id]
                    sentiment=self.l[i,word_id]
              
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
            if (iter+1)%100==0:
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
    
  


'''

if __name__=='__main__':
    stopwords_path='../论文/中文停用词/stopwords'
    root_path='C:/Users/Administrator/Desktop/data/新闻/'
    ###现在要加载所有的新闻
    type_list=['公益新闻','旅游新闻','娱乐新闻','健康新闻','科技新闻']
    P=Loaddata()
    all_sentences=[]
    for type_content in type_list:
        all_path=os.listdir(root_path+type_content+'/')
        sentences=P.load_news(all_path,type_content)
        all_sentences.extend(sentences)
    all_text,word_count=P.load_data(all_sentences,stopwords_path,top_k=50,low_k=50)
    img_path='C:/Users/Administrator/Desktop/data/新闻/词云图/公益.jpg'
    P.getwordcloud(word_count,img_path)
    M=get_dagword()
    word2id,id2word,corpus,wordnum=M.getdagword(all_text)
'''
if __name__=='__main__':
    stopwords_path='../论文/中文停用词/stopwords'
    path='C:/Users/Administrator/Desktop/data/评论/comment_info_final.csv'
    P=Loaddata()
    all_text=P.load_comment(stopwords_path,path)
    comment_train, comment_test = train_test_split(all_text, test_size = 0.3)
    #img_path='C:/Users/Administrator/Desktop/data/新闻/词云图/公益.jpg'
    #P.getwordcloud(word_count,img_path)
    #M=get_dagword()
    #word2id,id2word,corpus,wordnum=M.getdagword(all_text)
    M=ldamodel(20,5,0.1,0.1,0.1,comment_train,100)
    word2id,id2word,cut_corpus_id,wordnum=M.createdictionary(comment_train)
    M.initial(cut_corpus_id)
    M.gibbssampling(cut_corpus_id)
    






        
        


        
