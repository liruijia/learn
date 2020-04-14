#多种排序方法 以及 相应的算法复杂度
#简单选择排序方法

#冒泡排序
#冒泡排序：是不断地比较相邻两个元素的值，然后将最大值以冒泡的方式选择出来。
#算法复杂度：一般情况下n^2，最好的情况是已经排好序的N^2,最坏的情况逆序
#稳定性：稳定性指的是对于相同大小的元素其排序前后的相对位置并未发生变化
#冒泡排序是稳定的，其主要原因是每次冒泡的时候相邻位置的元素进行比较

class bubblesort():
    def bubble_sort(self,A):
        n=len(A)
        for i in range(n):
            for j in range(0,n-i-1):
                if A[j]>A[j+1]:
                    A[j],A[j+1]=A[j+1],A[j]

            print('第{0}次排序结果:{1}'.format(i,A))
        return A

#选择排序：选择排序，每次选择未排序的数组中选择最小值或者最大值放在已排好序的序列中。
#算法复杂度：N^2  最好的情况，已经排序序了不需要交换 但是复杂度仍然为n^2
#不稳定

class choicesort():
    def choice_sort(self,A):
        n=len(A)
        for i in range(n-1):
            min_ui=A[i]
            index=i
            for j in range(i+1,n):
                if A[j]<min_ui:
                    min_ui=A[j]
                    index=j
            A[i],A[index]=A[index],A[i]
            print('第{0}次排序结果:{1}'.format(i,A))
            #print('发生交换的位置:{0},{1}'.format(i,index))
        return A


if __name__=='__main__':
    A1=[56,23,2,4,67,46,23,45]
    print('*'*20,'bubble  sort','*'*20)
    P1=bubblesort()
    u1=P1.bubble_sort(A1)
    print('***最后的排序结果****',u1)
    print('\n')
    print('*'*20,'choice  sort','*'*20)
    A2=[34,23,5,32,56,34,9,3]
    P2=choicesort()
    u2=P2.choice_sort(A2)
    print('***最后的排序结果****',u2)
