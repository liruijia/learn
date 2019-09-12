import sys
import bisect
lines=sys.stdin.readlines()
n=int(lines[0])
#需要知道的是，我们的变数最大为1e+5
res=[0 for _ in range(int(1e+5))]

res[3]=0.5
res[4]=2
x1=0.5
x2=1.5
for i in range(5,int(1e+5)):
    if i%4==1:
        res[i]=res[i-1]+x1
        continue
    if i%4==2:
        x1+=1
        res[i]=res[i-1]+x1
    if i%4==3:
        res[i]=res[i-1]+x2
    if i%4==0:
        x2+=1
        res[i]=res[i-1]+x2

for line in lines[1:]:
    x=int(line)
    index=bisect.bisect_left(res,x)
    print(index)
