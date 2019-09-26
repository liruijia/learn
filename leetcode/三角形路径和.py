#三角形路径和
#寻找自顶向下找最近的结点
#每一次移动只能移动到相邻的地方
#n是不确定的，上一行访问了第I个节点，下一行只能访问I，i+1这两个节点
#此时dp[i][j]记录访问到该行到每个元素的最小路径的和
#dp[i][j]=min(dp[i-1][j],dp[i-1][j+1])+triangle[i][j]
class Solution:
    '''
    #自底向下
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n=len(triangle)
        if n==1 or triangle==[]:
            return triangle[0][0]
        dp=[[0 for _ in range(n)] for _ in range(n)]
        for  i in range(n-1,-1,-1):
            for j in range(len(triangle[i])):
                if i==n-1:
                    dp[i][j]=triangle[i][j]
                else:
                    dp[i][j]=min(dp[i+1][j],dp[i+1][j+1]) + triangle[i][j]
        return dp[0][0]
    '''
    def  minimumTotal(self,triangle):
        #自顶向下
        n=len(triangle)
        if n==1 :
            return triangle[0][0]
        if triangle==[]:
            return 0
        dp=[[0]*n for _ in range(n)]
        dp[0][0]=triangle[0][0]
        pre=triangle[0][0]
        for i in range(1,n):
            now=triangle[i]
            for  j in range(len(now)):
                if j==0:
                    dp[i][j]=dp[i-1][j]+triangle[i][j]
                elif j==len(now)-1:
                    dp[i][j]=dp[i-1][j-1]+triangle[i][j]
                else:
                    dp[i][j]=min(dp[i-1][j],dp[i-1][j-1])+triangle[i][j]
            min0=min(dp[i][j] for j in range(len(now)))
            pre=min0
        return pre
            
    
        
