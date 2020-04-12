#快手笔试 数学科学家 数据科学题目1
#求ROC曲线以及AUC的计算逻辑

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('C:/Users/Administrator/Desktop/label_score.csv')

#需要的值
FPR=[]
TPR=[]

#计算之前对于data按照score进行排序
da=data.sort_values(by='score',ascending=False)
n=len(da)
for i in range(n):
    ui=0
    tp=0
    fp=0
    fn=0
    tn=0
    for j ,row in da.iterrows():
        if ui<=i:
            if row['label']==1:
                tp+=1
            else:
                fp+=1
        else:
            if row['label']==1:
                fn+=1
            else:
                tn+=1
        ui+=1
    TPR.append(tp/(tp+tn))
    FPR.append(fp/(fp+fn))
sns.pointplot(x=FPR,y=TPR)
plt.xticks(rotation=270)
plt.show()

#计算AUC
#AUC表示了正例样本被分为正例的概率大于负利
#因此需要统计正例大于负利的概率
da1=da[da['label']==1]
da2=da[da['label']==-1]
m1=len(da1)
m2=len(da2)
kk=0
for i,row1 in da1.iterrows():
    for j ,row2 in da2.iterrows() :
        if row1['score']>row2['score']:
            kk+=1
AUC=kk/(m1*m2)

#如果要使用sklearn的话，则需要预测出来的label
#使用的包有：
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#confusion_matrix(y_true, y_pred)
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score#计算Roc的值


