import matplotlib.pyplot as plt     #数学绘图库
import jieba               #分词库
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import numpy as np
import pandas as pd
from zhon.hanzi import punctuation
import jieba
import re
import random

path='C:/Users/Administrator/Desktop/data/评论/comment_info_final.csv'

data=pd.read_csv(path,engine='python')
data['评价']=data['score'].map({5:'好评',4:'好评',3:'差评',2:'差评',1:'差评'})
dacou1=data[data['评价']=='差评']
dacou2=data[data['评价']=='好评']
daco1=dacou1['comment']
daco2=dacou2['comment']
stop_path='C:/Users/Administrator/Desktop/data/评论/stop_words.txt'
stop_words=[]
random_index=random.sample(list(range(0,11682)),len(daco1))
daco2_1=dacou2.loc[random_index,'comment']
with open(stop_path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        lines=line.strip('\n').split(' ')
        stop_words.extend(lines)
def get_comment_lx(dacop):
    comment=[]
    for ui in dacop:
        #print(ui)
        try:
            sp=ui.split('\n')
            lk=[]
            for oo in sp:
                kl=oo.split('：')
                lk.append(kl[-1])
            lk=','.join(lk)
            comment.append(lk)
        except Exception as result:
            print(result)
    return comment
comment1=get_comment_lx(daco1)
comment2=get_comment_lx(daco2_1)

comment3=get_comment_lx(daco2)


print(len(comment1),len(comment2))







comment0=comment1+comment2
comment=comment1+comment3
def get_corpus(comment_iu):
    corpus=[]
    for c in comment_iu:
        new_c=re.sub(r'[%s,\t,\\]+'%punctuation,' ',c)
        cut_c=jieba.cut(new_c)
        new_doc=[]
        for word in cut_c:
            #print(word,word.isalpha())
            if word not in stop_words:
                if word.isalpha() is True :
                    if len(word) >=2:
                        new_doc.append(word)
                    #print(word)
        corpus.append(new_doc)
    return corpus
corpus0=get_corpus(comment0)
corpus=get_corpus(comment)
path='C:/Users/Administrator/Desktop/data/corpus/oppo_comment_1.txt'
with open(path,'w',encoding='utf-8') as f:
    for lin in corpus:
        f.write(' '.join(lin))
        f.write('\n')
    f.close()

path2='C:/Users/Administrator/Desktop/data/corpus/oppo_comment_1_balance.txt'
with open(path2,'w',encoding='utf-8') as f:
    for lin in corpus0:
        f.write(' '.join(lin))
        f.write('\n')
    f.close()


path3='C:/Users/Administrator/Desktop/data/corpus/pda/cut_pda_all.txt'
image = Image.open(r'C:/Users/Administrator/Desktop/论文/picture/词云图背景.png')
my_mask = np.array(image)
c=[]
with open(path3,'r',encoding='utf-8') as f:
    sentences=f.read()
    f.close()
model=WordCloud(background_color='white',min_font_size=5,font_path='msyh.ttc',max_words=3000
                ,max_font_size=1000)
model.generate(text=sentences)
#image_color = ImageColorGenerator(my_mask)
model.to_file('C:/Users/Administrator/Desktop/论文/picture/pda_词云.png')
plt.figure("词云图")
plt.imshow(model)
plt.axis("off")
plt.show()
