# insertion sort
# 卓望最后一道题目
# 插入排序：将一个数组看成两部分，一部分是已经排好序的，另外一部分是未排好序的，
# 排序的操作是，将未排好序的部分的元素和排好序的部分的元素进行比较，找到其合适的位置
# 若一个数组大部分是排好序的，则使用使用插入排序

class insertsort():
    def insertion_sort(self,A):
        n=len(A)
        for j in range(1,n):
            for i in range(j,0,-1):
                if A[i-1]>A[i]:
                    A[i-1],A[i]=A[i],A[i-1]
                else:
                    break
        return A

if __name__=='__main__':
    A=[4,5,6,0,2,9,3]
    P=insertsort()
    print(P.insertion_sort(A))
