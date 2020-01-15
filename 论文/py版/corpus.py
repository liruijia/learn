import pandas as pd
import random
import collections
from gensim.models  import word2vec
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import  os
path1='C:/Users/Administrator/Desktop/data/中文情感分析语料库/'
lj1=os.listdir(path1)
all_path=[]
for line in lj1:
    op=os.listdir(path1+line)
    for ming in op:
        all_path.append(path1+line+'/'+ming)

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
    return data
all_corpus=[]
for path in all_path:
    data=getdata0(path)
    all_corpus.extend(data)
with open('C:/Users/Administrator/Desktop/data/corpus/all_corpus_3f.txt','w',encoding='utf-8') as f:
    for ui in all_corpus:
        f.write(ui)
        f.write('\n')
    f.close()
word_list={}
for corpus in all_corpus:
    for doc in corpus :
        for word in corpus:
            word_list[word]=word_list.get(word,0)+1

