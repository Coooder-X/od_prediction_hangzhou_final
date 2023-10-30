# Package
torch==1.7.1+cu110  
tensorboardX==2.1  
numpy==1.16.6  
sklearn==0.23.2  
可能有所遗漏，具体的可以查看其中


#模型
1、model_ST_plus.py是模型文件，在该文件中需要将基本没有需要设置的参数。
其中需要的代码模块有：Convolution_block.py、IdentityBlock.py、MultiGraphCNN.py。其中MultiGraphCNN.py为其中的MGC操作。

2、训练模型：
```shell script
python train.py --lr 0.01
```
# 参数解释
'--use_adj':表示是否使用轨迹相似度矩阵。

'--use_vec'：目前没有用处

'--use_three_branch'：表示是否使用处理轨迹特征的分支。

'--self_naming'：为了避免结果发生命名重复发生覆盖，可以自命名一个后缀名

'--use_dynamic_graph'：表示是否使用动态的图

'--root':表示数据集的位置

'--save_folder'：表示训练结果的保存目录

'--len_trend'：表示使用前多少个周的相同时间的数据

'--len_period'：表示使用前多少天的相同时间的数据

'--len_closeness'：表示使用前多少小时的的数据

'--train_prop'、'--val_prop'：用于划分数据集

'--latent_dim'：表示图神经网络编码器得到的维度

'--latent_dim_l'：表示LSTM网络编码器得到的维度

'--activation'：表示使用的激活函数的类型

'--is_batch_normalization'：表示是否使用batch_normalization

'--network_structure_e'：设置图神经网络编码器具的结构

'--network_structure_l'：设置LSTM编码器具的结构

'--network_structure_d'：设置图神经网络解码器具的结构

'--lr'：学习率，很重要的参数，调整为0.0003-0.001范围内，基本影响不大。
  
以下参数会根据输入数据进行自动调整：

'--node_num'：表示节点数量
'--num_filters'：表示图的数量
