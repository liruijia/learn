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
        stop_words.add(u'ColorOS')
        stop_words.add(u'Aeno')
        stop_words.add(u'GPU')
        stop_words.add(u'gpu')
        stop_words.add(u'VO')
        stop_words.add(u' color OS MIUI ')
        stop_words.add(u'emui')
        stop_words.add(u' color')
        stop_words.add(u'OS')
        stop_words.add(u'nfc')
        stop_words.add(u'O')
        stop_words.add(u'hellip')
        stop_words.add(u'OTG')
        stop_words.add('NFC ')


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
            new_c=re.sub(r'[%s,\t,\\]+'%punctuation,' ',c)
            cut_c=jieba.lcut(new_c)
            new_doc=[]
            if '未填写' in cut_c or new_c=='' or new_c=='\n':
                continue
            else:
                for word in cut_c:
                    #print(word,word.isalpha())
                    if word not in stop_words:
                        if word.isalpha() is True :
                            if len(word) >=2:
                                new_doc.append(word)
                        #print(word)
            corpus.append(new_doc)
        f=open('C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt','w',encoding='utf-8')
        for i in corpus:
            f.write(' '.join(i) )
            f.write('\n')
        f.close()
        print('已经加载完毕评论形成corpus***************')
        return corpus
    def load_comment_zwqgfxylk(self,stopwords_path,path):
        root_path='C:/Users/Administrator/Desktop/data/中文情感分析语料库/'
        cut_corpus=[]
        stop_words = list(self._loadstopwords(stopwords_path))
        with open(path,'r',encoding='utf-8') as f:
            c=[]
            for line in f.readlines():
                new_c = re.sub(r'[%s,\t,\\]+' % punctuation, ' ', line)
                cut_c = jieba.lcut(new_c)
                for word in cut_c:
                    if word not in stop_words :
                        if  word.isalpha() is True:
                            c.append(word)
            cut_corpus.append(c)
            f.close()
        return cut_corpus
        
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
    P=Loaddata()
    stopwords_path='C:/Users/Administrator/Desktop/github/learn/learn/论文/中文停用词/stopwords/'
    path='C:/Users/Administrator/Desktop/data/评论/comment_info_final.csv'
    all_text=P.load_comment(stopwords_path,path)
    #path_list=os.listdir('C:\\Users\\Administrator\\Desktop\\data\\中文情感分析语料库')
    #load_comment_zwqgfxylk(stopwords_path, path)
 
