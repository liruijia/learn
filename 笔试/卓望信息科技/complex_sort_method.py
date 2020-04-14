#比较复杂的排序方法
# 归并排序、快速排序、堆排序，其复杂度相对简单排序来说会小一点

#快速排序
#快速排序采用了分冶治法的方法来排序，将大问题转换成了小问题来解决，每一次选择出来一个povit值将数组划分成两部分，分别是小于povit的值和大于
#povit的值，不断地进行划分，最小的问题是只有两个元素的数组
#稳定性：不稳定
#还有随机化版本
#算法复杂度：n*lgn

class quicksort():
    def quick_sort(self,A,left_len,right_len):
        if left_len < right_len:
            qovit=self.get_divide_point(A,left_len,right_len)
            self.quick_sort(A,left_len,qovit)
            self.quick_sort(A,qovit+1,right_len)
        return A

    def get_divide_point(self,A,left_len,right_len):
        x=A[right_len-1]
        index_i=left_len-1
        #j是为了遍历找到比x更小元素的索引值，i指的则是当前已经找到的最后一个小于x的元素的索引位置
        for j in range(left_len,right_len-1):
            if A[j]<x:
                index_i+=1
                A[index_i],A[j]=A[j],A[index_i]
        A[index_i+1],A[right_len-1]=A[right_len-1],A[index_i+1]
        print('找到的当前的划分点:{0}'.format(index_i+1))
        print('进行了一次处理之后的A：',A)
        return index_i+1
'''
def quick_sort_standord(array,low,high):
    if low < high:
        key_index = partion(array,low,high)
        quick_sort_standord(array,low,key_index)
        quick_sort_standord(array,key_index+1,high)

def partion(array,low,high):
    key = array[low]
    while low < high:
        while low < high and array[high] >= key:
            high -= 1
        if low < high:
            array[low] = array[high]

        while low < high and array[low] < key:
            low += 1
        if low < high:
            array[high] = array[low]

    array[low] = key
    return low

if __name__ == '__main__':
    array2 = [9,3,2,1,4,6,7,0,5]

    print array2
    quick_sort_standord(array2,0,len(array2)-1)
    print array2

'''
#快速排序的另外一种写法

#堆排序
#堆排序，首先是简堆，其次进行维护堆的属性；堆排序的整个过程可以可视化成二叉树的形式
#节点之间索引之间的关系：父节点为i时，其左节点索引为2i,右节点的索引为2i+1
#堆排序主要做的几件事：建立最大堆、维护最大堆、堆排序
#算法复杂度：nlgn 和树的高度有关系
class heapsort():
    
    
    def MAX_Heapify(self,heap,HeapSize,root):
        '''维护第i个点的性质'''
        #最大堆的性质，即指父节点的值大于其左节点以及右节点,维护最大堆的属性
        left = 2*root + 1
        right = left + 1
        larger = root
        if left < HeapSize and heap[larger] < heap[left]:
            larger = left
        if right < HeapSize and heap[larger] < heap[right]:
            larger = right
        if larger != root:#如果做了堆调整则larger的值等于左节点或者右节点的，这个时候做对调值操作
            heap[larger],heap[root] = heap[root],heap[larger]
            self.MAX_Heapify(heap, HeapSize, larger)

            
    def build_max_heap(self,A):
        '''建立最大堆的过程就是维护所有内节点的最大堆性质，总共有N个节点，其最后一个内节点的索引为Int(N/2),
        维护的时候是从底向上'''
        n=len(A)
        for i in range((n-2)//2,-1,-1):
            self.MAX_Heapify(A,n,i)
            print('进行一次维护后的A：',A)
            

    def heap_sort(self,heap):
        '''开始基于最大堆进行排序，建立好最大堆之后，需要将结果按照升序进行输出，则需要每次将最大值和最后一个值进行交换，
           然后再维护性质，且从中删除最后一个节点（不是真的删除），然后维护第一个位置的最大堆属性'''
        self.build_max_heap(heap)
        for i in range(len(heap)-1,-1,-1):
            heap[0],heap[i] = heap[i],heap[0]
            self.MAX_Heapify(heap, i, 0)
        return heap

#归并排序
#归并排序是利用分冶的思想进行求解
        
if __name__=='__main__':
    A1=[67,54,34,23,35,64,23,56,78,68]
    print('*'*20,'快排','*'*20)
    left_len=0
    right_len=len(A1)
    P=quicksort()
    u=P.quick_sort(A1,left_len,right_len)
    print('最终的结果',u)
    A2=[67,54,34,23,35,64,23,56,78,68]
    print('\n')
    print('*'*20,'堆排序','*'*20)
    P=heapsort()
    u=P.heap_sort(A2)
    print('最终的结果',A2)
    
        

