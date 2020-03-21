## ransac伪代码
### 变量解释
input:  
data---一组观测数据  
model---适应于数据的模型  
n---适用于模型的内点个数  
k---算法的迭代次数  
t---用于决定数据是否适应于模型的阈值
d---判定模型是否适用于数据集的数据数目  
output：  
best_model---跟数据最匹配的模型参数（如果没有找到好的模型，返回null）  
best_consensus_set---最后估计出模型的内点集
best_error---跟数据相关的估计出的模型的错误  
### 伪代码
初始化：  
iterations = 0   //迭代次数  
best_model = null   //初始最优模型  
best_consensus_set = null   //内点集  
best_error = 无穷大  
  
迭代操作：  
while(iterations < k):  
maybe_inliers = 从data中随机选择n个点作为内点集
maybe_model = 适合于maybe_inliers的模型参数  
consensus_set = maybe_inliers  
  
for dot in (非内点集中的点)：  
if(dot适合于maybe_model&错误小于t)：
consensus_set.add(dot)  

if(len(consensus_set)>d):  
已经找到了好的模型，现在测试该模型有多好  

better_model = 适合于consensus_set中所有点的模型参数  
this_error = better_model  
if (this_error < best_error):  
发现更好的模型，保存该模型直到更好的模型出现  

best_model = better_model  
best_consensus_set = consensus_set  
best_error = this_error
  
增加迭代次数  
return best_model,best_consensus_set,best_error


      