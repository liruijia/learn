#昆仑万维
#1.走台阶 总共有n个台阶，每一次走台阶的可以走1步，2步...n阶，先求走N个台阶的方法
def solve_1():
    n=int(input())
    if n==1:
        print(1)
    elif n==2:
        print(2)
    elif n==3:
        print(4)
    else:
        dp=[0 for _ in  range(n)]
        dp[0]=1
        dp[1]=2
        dp[2]=4
        for i in range(3,n):
            dp[i]=sum(dp[:i])+1
        print(dp[-1])

#进行字符串的反转，单词内部不进行反转
# i am a student 反转结果为 student a am i
def solve_2():
    s=list(map(str,input().split(' ')))
    res=[]
    for i in range(len(s)):
        if s[i]!='':
            res.append(s[i])
    print(' '.join(res))

#3 求解连续子数组的和最大
def solve_3():
    a=list(map(int,input().split(' ')))
    n=len(a)
    if n==1:
        return a[0]
    else:
        dp=[0 for _ in range(n)]
        dp[0]=a[0]
        dp[1]=max(a[0]+a[1],a[1])
        res=dp[0]
        for i in range(2,n):
            dp[i]=max(dp[i-1]+a[i],a[i])
            res=max(res.dp[i])
        return res
        

        
