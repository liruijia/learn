#coding:utf-8
#完全背包问题
import sys
 
 
def track(d, c, w):
    x = []
    while c > 0:                                                #因为可以重复，所以每次都需要重新寻找w
        for i in range(len(w), 1, -1):
            if d[i][c] != d[i - 1][c]:
                x.append(w[i - 1])
                c = c - w[i - 1]
                break
 
    if d[1][c] > 0:
        x.append(w[0])
    return x
 
if __name__ == '__main__':
    c = int(input())                                                 #输入一个限制条件，例如背包的体积为c
    w = sys.stdin.readline().strip().split(' ')                 #每个物品的体积
    w = list(map(int, w))
    v = sys.stdin.readline().strip().split(' ')                 #对应每个物体的价值
    v = list(map(int, v))
 
    dp = [[0 for _ in  range(c + 1)] for i in range(len(w)+1)]
 
    for i in range(1, len(w) + 1):
        for j in range(1, c + 1):
            if j >= w[i - 1]:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - w[i - 1]] + v[i - 1])
            else:
                dp[i][j] = dp[i - 1][j]
 
    print(max(dp[len(w)]))
 
    print(track(dp, c, w))
