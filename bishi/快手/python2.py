#快手春招B卷数据科学第二题
#使用python来验证中心极限定理
#中心极限定理：大量相互独立的随机分布，其均值的分布服从于均值为miu，
#方差为sigma^2/miu的正态分布，随机变量越多，其结果越明显

#用python来验证，使用任意一种分布，这里使用均匀分布，随机产生M个随机变量，
#求其均值，然后然后求均值的直方图即可验证，要么就是判断均值使用达到了
#原分布的均值

import matplotlib.pyplot as plt


import numpy as np

import numpy

iter_num=10000

Y=np.zeros((iter_num,10000))

for i in range(iter_num):
    Y[i,:]= numpy.random.uniform(-5,5,10000)

M=np.mean(Y,axis=1)

#画图，从直方图中可以发现是正态分布
plt.hist(M)
plt.show()

#均匀分布[-5,5]的均值为： （5-5）/2 =0 ，方差为：（b-a）^2 /12 /n= 100/12/n

miu=np.mean(M)
sigma=np.var(M)

mubiao_miu=0
mubiao_sigma=100/12/iter_num
print('最后得到的miu:{0},目标：{1}'.format(miu,mubiao_miu))
print('最后得到的sigma：{0}，目标：{1}'.format(sigma,mubiao_sigma))

