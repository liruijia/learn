#将之前的所有可能的二叉搜索树打印出来
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def generateTrees(self, n):
        if not n :
            return []
        def generate_trees(start, end):
            #要求所有可能的数，则是要遍历掉所有树的可能性，每个节点都可以作为树的根，
            #选择根之后，需要查看左子树以及右子树的根
            if start > end:
                return [None,]
            all_trees = [] 
            for i in range(start, end + 1): 
                left_trees = generate_trees(start, i - 1)
                right_trees = generate_trees(i + 1, end)
                for l in left_trees:
                    for r in right_trees:
                        current_tree = TreeNode(i)
                        current_tree.left = l
                        current_tree.right = r
                        all_trees.append(current_tree)
                #print('---all_tree',all_trees)
            return all_trees
        
        def levelOrder( root):
            levels = []
            if not root:
                return levels
            def helper(node, level):
                if len(levels) == level:
                    levels.append([])
                if node is not None:
                    levels[level].append(node.val)
                else:
                    levels[level].append('null')
                if node.left is None or node.right is None :
                    levels.append([])
                if node.left:
                    helper(node.left, level + 1)
                else:
                    levels[level+1].append('null')
                if node.right:
                    helper(node.right, level + 1)
                else:
                    levels[level+1].append('null')
                
                
            helper(root, 0)
            return levels
                
        all_trees=generate_trees(1, n)
        res=[]
        for tree in all_trees:
            b=levelOrder(tree)
            b.pop(-1)
            a=[j for x in b for j in x]
            res.append(a)
        return res
        
P=Solution()
a=P.generateTrees(3)

