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
            for l in range(i):  #j-i+1<n
                if (r-l<=2 or dp[l+1][r-1]==True) and s[r]==s[l]:
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
