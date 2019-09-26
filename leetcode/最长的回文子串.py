'''
为了改进暴力法，我们首先观察如何避免在验证回文时进行不必要的重复计算。考虑 “ababa” 这个示例。“bab” 是回文，很明显，
“ababa” 一定是回文，因为它的左首字母和右尾字母是相同的。

我们给出 P(i,j)P(i,j) 的定义如下：

P(i,j)={ true ---if s[i],...s[j]是回文 否则false,

P(i, j) = ( P(i+1, j-1) and  S_i == S_j )

基本示例如下：

P(i, i) = true
P(i,i)=true

P(i, i+1) = true ( S_i == S_{i+1} )
'''
class Solution:
    def longestPalindrome(self, s) :
        n=len(s)
        dp=[[False]*n for i in range(n)]
        largest=1
        for i in range(n):
            dp[i][i]=True
        for i in range(n-1):
            if s[i]==s[i+1]:
                dp[i][i+1]=True
                largest=2
        res=s[0]
        for r in range(1,n):
            for l in range(r):  #l为左边，r为右边
                if dp[l+1][r-1]==True and s[r]==s[l]:
                    dp[l][r]=True
                    cur=r-l+1
                    if cur>=largest:
                        largest=cur
                        res=s[l:r+1]
                        print('----res',res)
        return res
p=Solution()
s="babad"
a=p.longestPalindrome(s)
