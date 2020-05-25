### 获取其中一部分的数据
import json
path_senti='C:/Users/Administrator/Desktop/data/corpus/handle_corpus_train.txt'
path_no='C:/Users/Administrator/Desktop/data/corpus/handle_corpus_train2.txt'

data1=open(path_senti,encoding='utf-8').read()
data2=open(path_no,encoding='utf-8').read()
#print(data1[:10])
corpus1=json.loads(data1.lstrip('\ufeff'))
corpus2=json.loads(data2.lstrip('\ufeff'))

corpus1_senti=corpus1['正面']+corpus1['反面']
corpus2_no=corpus2['正面']+corpus2['反面']

i=0
j=0

write_senti='C:/Users/Administrator/Desktop/data/corpus/senti_10.txt'
write_no='C:/Users/Administrator/Desktop/data/corpus/no_10.txt'
with open(write_senti,'w',encoding='utf-8') as f:
    for doc in corpus1_senti:
        i+=1
        if i<11:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()
with open(write_no,'w',encoding='utf-8') as f:
    for doc in corpus2_no:
        j+=1
        if j<11:
            f.write(' '.join(doc))
            f.write('\n')
    f.close()
