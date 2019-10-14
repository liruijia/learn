import collections
import sys
class Solution():
    def maxPoints(self, p) :
        cnt = collections.Counter((x, y) for x, y in p)
        if len(cnt) <= 2:
            return len(p)
        ans = 0
        for _ in range(1, len(cnt)):
            (x1, y1), t1 = cnt.popitem()
            slp = collections.defaultdict(lambda: t1)
            for (x2, y2), t2 in cnt.items():
                s = (y2 - y1) / (x2 - x1) if x1 != x2 else float('inf')
                slp[s] += t2
            ans = max(ans, max(slp.values()))
        return ans



lines=sys.stdin.readlines()
a=[]
for line in lines:
    x,y=list(map(int,line.split(' ')))
    a.append([x,y])
p=Solution()
print(p.maxPoints(a))
        
