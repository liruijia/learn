# learn
## 动态规划

   最近做笔试被动态规划和贪心、图这些算法打击的有点深，明明之前都学过的东西，但是在做题的时候，就是不会，哎，现在要好好梳理一下这些内容，先从动态规划问题开始，0-1背包问题

   在这之前先复习一下：动态规划方法的思想，在求解某些问题的时候，我们如法直接求解原始问题的解，因此，需要讲大问题转换成一一的小问题进行求解，当然这些小问题是可求解的，在动态规划中，我们需要注意的是我们之关注当前状态以及影响该状态的一些状态，但是需要注意的是，对于原始问题来说，每个状态都无法直接影响到未来的结果。
   
   可以发现的是，可分治的思想有点相似，但是动态规划中，我们更关注状态，在分治里面我们是将序列进行缩小。
   
   能采用动态规划求解的问题的一般要具有3个性质：
   
   1.最优化原理：如果问题的最优解所包含的子问题的解也是最优的，就称该问题具有最优子结构，即满足最优化原理。
   
     比如说我们的大问题是求解一定条件下的最大值，则每个小问题也是一定条件下的最大值
     
   2.无后效性：即某阶段状态一旦确定，就不受这个状态以后决策的影响。也就是说，某状态以后的过程不会影响以前的状态，只与当前状态有关。
    
    
   3有重叠子问题：即子问题之间是不独立的，一个子问题在下一阶段决策中可能被多次使用到。（该性质并不是动态规划适用的必要条件，但是如果没有这条性质，动态规划算法同其他算法相比就不具备优势）    
   
   # 0-1背包问题

 将N个物品放到容量为V的背包中，每个物品有自己的重量以及价值，也就是说每向背包放1件物品，给背包带来w的价值
 
 现在有输入：
         N---物品数
         
         V---背包容量
         
         C数组---c1,c2....cN  表示每一件物品的重量
         
         w数组---w1,w2....wN  表示每一件物品的价值
         

现在求给背包放入一定物品之后使得背包的价值最大，求最大的价值

也就是要求，放入N件商品后背包达到的最大价值

使用动态规划的思想：---此时需要注意的是物品个数在变，背包中的空间在变

在放入的时候我们可以将重量进行排序

小问题：将前i件物品放入容量为j的背包之后的最大价值

dp表示二维数组

dp[i][j]----表示前i件放入容量为j的背包之后的最大价值

现在需要考虑dp[i][j]是从哪些状态转移而来的

 需要再次分析原来的问题，我们在将一件物品放入背包之前，我们会考虑
 
 1.我们现在的物品会不会使得背包超重
 
 2.我们的想法是放进去的东西价值大，体积小
 
 对于当前状态：dp[i][j]-----其可能来自哪些状态？？？
 
 当我们在分析状态的时候，我们需要知道原始问题主要的行为是什么，放或者不放，怎么的情况下放，怎么样是不放
 
 
 对于第i件物品（重量为Ci,价值为Wi），放入前i-1物品的小问题时，考虑到底放不放？？
 
 1.当第i件物品没有超重的时候--可以选择放，也可以不放
 
   放入问题i-1的小问题中，此时：
  
         dp[i][j] = dp[i-1][j-ci] + w[i]
 
 
   不放入
 
          dp[i][j] = dp[i-1][j]
  
 因此我们当前状态的最优值为：
   
         dp[i][j] = max(dp[i-1][j],  dp[i-1][j-ci]+w[i])----最优子结构
 
 2.当第i件物品超重的时候
 
         dp[i][j]=dp[i-1][j]
         
         
         
具体的代码见0-1package.py


