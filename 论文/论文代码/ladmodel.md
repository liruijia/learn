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
    