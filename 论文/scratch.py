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



import pandas as  pd
from gensim.models  import word2vec
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

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
if __name__=='__main__':
    path0='C:/Users/Administrator/Desktop/data/评论/cut_comment_1.txt'
    path1 = 'C:/Users/Administrator/Desktop/data/情感词汇本体/情感词汇本体.csv'
    path2 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面情感词语（中文）.txt'
    path3 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面情感词语（中文）.txt'
    path4 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/程度级别词语（中文）.txt'
    path5 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/正面评价词语（中文）.txt'
    path6 = 'C:/Users/Administrator/Desktop/data/情感字典/知网Hownet情感词典/负面评价词语（中文）.txt'

    path_list = [path0,path1,path2, path3, path4, path5, path6]

    #vector=CountVectorizer()
    #trans=TfidfTransformer()
    #tfidf = trans.fit_transform(vector.fit_transform(final_corpus))
    #word = vector.get_feature_names()  #
    #weight = tfidf.toarray()

    P=sentiment_dict()
    final_data0=P.get_data(path_list)
    path01='C:/Users/Administrator/Desktop/data/评论/final_corpus.txt'
    sentences=word2vec.Text8Corpus(path01)
    model= word2vec.Word2Vec(sentences,size=400,window=5,min_count=1)

    data1=pd.read_csv(path1,engine='python')
    df_info=pd.DataFrame(columns=['情感分类'])
    df_info['情感分类']=data1['情感分类'].unique().tolist()
    df_info['情感大分类']=df_info['情感分类'].map({'PA':1,'PE':1,
                                             'PD':2, 'PH':2, 'PG':2, 'PB':2, 'PK':2,
                                        'NA':3,'NB':4,'NT':4,'NH':4,'PF':4,'NI':5,'NC':5,'NG':5,
                                            'NE':6,'ND':6,'NN':6,'NK':6,'NL':6,'PC':7})
    data1['情感大分类']=data1['情感分类'].map({'PA':1,'PE':1,
                                             'PD':2, 'PH':2, 'PG':2, 'PB':2, 'PK':2,
                                        'NA':3,'NB':4,'NT':4,'NH':4,'PF':4,'NI':5,'NC':5,'NG':5,
                                            'NE':6,'ND':6,'NN':6,'NK':6,'NL':6,'PC':7})
    df_info['情感大分类_表达']=df_info['情感大分类'].map({2:1,3:2,6:2,4:3,5:3,7:3,1:4})
    df_info['情感大分类_态度']=df_info['情感大分类'].map({1:0,2:0,3:1,4:1,5:1,6:1,7:1})

    final_info_sentword=P.load_amend_dict(data1,final_data0)
    print(final_info_sentword['OPPO'])
    items=model.most_similar(u'好评',topn=20)
    print('“好评”一词的相似的词语')
    for word ,sim_par in items:
        print(word,sim_par)















