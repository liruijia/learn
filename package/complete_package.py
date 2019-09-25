#dp[i][j]表示装有i件物体且重量为j的背包中的最大价值
#dp[i][v]=max(dp[i-1][v-k*c_i]+k*w[i]) k的选择为v-k*c_i>=0  k<= v/c_i
#dp[i][j]=max(dp[i-1][j-k*c_i]+k*w[i])
import math
def complete_package1(c,w,v,n):
    dp=[[-1 for _ in range(v+1)] for i in range(n+1)]
    if n==1 and c[-1]<=v:
        return w[i]*v//c[-1]
    else:
        for i in range(v+1):
            dp[0][i]=0
        for i in range(1,n):
            for j in range(0,v+1):
                if c[i]>j:
                    dp[i][j]=dp[i-1][j]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-c[i]]+w[i])
    #print('--',dp)
    return dp[n-1][v]

def complete_package0(c,w,v,n):
    dp = [0 for i in range(v+1)]
    for i in range(n):
        for j in range(v,-1,-1): # 从后往前
            k = j//c[i]  # 能选多少次
            # 从这些次里面取最大
            dp[j] = max([dp[j- x* c[i]] + x * w[i] for x in range(k+1)])

    #print('--',dp)
    return dp[-1]


def complete_package2(c,w,v,n):
    new_c=[]
    new_w=[]
    #进行划分，然后使用普通的0-1背包问题
    for i in range(n):
        if c[i]>v:
            new_c.append(c[i])
            new_w.append(w[i])
        else:
            x=[c[i]*k for k in range(1,int(v/c[i])+1) if c[i]*k<=v]
            y=[w[i]*kk for kk in range(1,int(v/c[i])+1) if c[i]*kk<=v]
            new_c.extend(x)
            new_w.extend(y)
    print('new_c',new_c)
    print('new_w',new_w)
    lc=len(new_c)
    dp=[[-1 for _ in range(v+1)] for i in range(lc+1)]
    for  i in range(v+1):
        dp[0][i]=0
    for i in range(1,lc):
        for j in range(0,v+1):
            if new_c[i]>j:
                dp[i][j]=dp[i-1][j]
            else:
                dp[i][j]=max(dp[i-1][j],dp[i-1][j-new_c[i]])+new_w[i]
    #print('*'*10,dp)
    return dp[lc-1][v]


def complete_package3(c,w,v,n):
    dp=[[-1 for _ in range(v+1)] for i in range(n+1)]
    i=0
    while i<n:
        for j in range(n):
            if c[i]>=c[j] and w[i]<=w[j]:
                tmp=c[j]
                cmp=w[j]
                c[i]=tmp
                w[i]=cmp
        i+=1
    for i in range(v+1):
        dp[0][i]=0
    print('---c',c)
    print('---w',w)
    
    for i in range(0,n):
        for j in range(0,v+1):
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-c[i]]+w[i])
    print(dp)
    return dp[n-1][-1]

if __name__=='__main__':
    n=4
    c=[7 ,5 ,4 ,2]
    w=[9,3,7,10]
    v=10
    print(complete_package1(c,w,v,n))
    print(complete_package0(c,w,v,n))
    print('----前面两种是一种思路')
    print(complete_package2(c,w,v,n))
    print('----另外一种思路')
    print(complete_package3(c,w,v,n))
