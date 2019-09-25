"input: n,v,c,w"

def solve_01_package(n,v,c,w):
  dp=[[-1 for i in range(v+1)]  for _ in range(n+1)]
  for k in range(v+1):
    dp[0][k]=0
  for i in range(1,n+1):
    for j in range(1,v+1):
      if c[i]>v:
        dp[i][j]=dp[i-1][j]
      else:
        dp[i][j]=max(dp[i-1][j],dp[i-1][j-c[i]]+w[i])
  return dp

if __name__=="main":
  n=int(input())
  v=int(input())
  c=list(map(int,input().split(' ')))
  w=list(map(int,input().split(' ')))
  res=solve_01_package(n,v,c,w)
  print(dp[n][v])
