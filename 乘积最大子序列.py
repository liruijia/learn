#乘积最大的连续子序列
#连续
#大问题长度为N的序列中，连续子序列中的最大乘积
#子问题为长度为i序列中，连续子序列的最大值
#dp[k]---表示最大子序列k的最大乘积
#会用到down的原因是因为nums[i]<0的时候，会让最大值变成最小值，最小值变成最大值。
#从下面的解法也可以看见并没有直接使用dp,因为在寻找最大值，最小的的时候，只需要知道当前最大、当前最小、最终最大即可
#up[k]=max(up[k-1]*nums[k],nums[k])  ---这一前提是nums[k]>0
#up[k]=max(down[k-1]*nums[k],nums[k]) ----nums[k]<0
#down[k]=min(down[k-1]*nums,nums[k])---nums[k]>0
#down[k]=min(up[k-1]*nums,nums[k])  ----nums[k]<0
class Solution:
    def maxProduct(self, nums) :
        n=len(nums)
        if n==1:
            return nums[0]
        elif n<=2:
            return max(nums[0],nums[1],nums[0]*nums[1])
        max_0,min_0=nums[0],nums[0]
        final_max=nums[0]
        for n in nums[1:]:
            if n>0:
                max_0,min_0=max(max_0*n,n),min(min_0*n,n)
                print('n>0,max_0,min_0',max_0,min_0)
            else:
                max_0,min_0=max(min_0*n,n),min(max_0*n,n)
                print('n<0,max_0,min_0',max_0,min_0)
            final_max=max(max_0,final_max)
        return final_max
P=Solution()
nums=[2,3,-1,4]
b=P.maxProduct(nums)
