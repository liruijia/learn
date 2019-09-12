#解码方法的总数
#长度为n的字符串的解码方法总数
#在解码的时候单个字符肯定实在A-Z的范围之类 ，所以不需要判断
#主要是看其和前一个 字符是否组成A-Z的数字，若组成了
    #dp[i]=dp[i-1]+dp[i-2]
#若没有则 dp[i]=dp[i-1]
#特殊的情况dp[1]=1,dp[2]=2(满足条件的时候)
#s=‘226’
#dp[1]=1  dp[2]=2 dp[3]=dp[2]+dp[1]=2+1=3
#但是有的时候如法解码，因此需要return 0
class Solution:
    def numDecodings(self, s):
        n=len(s)
        if n==1:
            if s[0]>=1:
                return 1
            else:
                return 0
        dp=[0 for _ in range(n+1)]
        if int(s[0])>=1:
            dp[1]=1
        else:
            dp[1]=0
        if n>1:
            if int(s[0])==0:
                if int(s[:2])<=26 and int(s[1])>=1:
                    dp[2]=1
                else:
                    return 0
            else:
                if  int(s[:2])<=26:
                    dp[2]=2
                else:
                    dp[2]=1
        for i in range(3,n+1): 
            if s[i] == '0':
                if s[i-1] == '0': 
                    return 0 
                else:
                    if int(s[i-1:i+1]) <= 26: 
                        dp[i] = dp[i-2]
                    else:
                        return 0
            else: 
                if s[i-1] == '0': 
                    dp[i] = dp[i-1]
                else: 
                    if int(s[i-1:i+1]) <= 26:
                        dp[i] = dp[i-1] + dp[i-2]
                    else:
                        dp[i] = dp[i-1]
        return dp[-1]
