CharRNN_text_classification_tf：基于RNN（lstm,rnn,gru,nas）实现基于字符的语言模型
===============================================================================

Requirements:
==============
*Tensorflow1.8.0<br>
*numpy<br>

项目架构：
=========
data_helper.py:数据处理文件，此处在生成batch时利用了指针和next_batch();<br>
CharRNN_Model.py:构建的RNN网络模型；一般思路为：RNNCell -> dropout -> MultiRNNCell -> 重复time_steps步;<br>
train.py:模型训练文件；<br>
test.py:随便的一个测试文件。<br>

执行训练：
=========
默认情况下可执行如下命令：
```python
python train.py
```
但需要自己指定参数，可先查看参数配置：
```python
python train.py --help
```
然后修改自己想要指定的参数值,例如想将默认的lstm单元修改为rnn单元
```python
python train.py --model=rnn
```

训练结果：
=========
使用默认配置训练的loss如下图所示：
-------------------------------
从图中可以看到，在接近5000步时，loss就已经降到1.40左右
![训练过程loss变化图](https://github.com/Vincent131499/CharRNN_text_classification_tf/blob/master/train_loss.jpg "train_loss图")

使用tensorboard可视化的网络图如下所示：
------------------------------------
![网络结构图](https://github.com/Vincent131499/CharRNN_text_classification_tf/blob/master/model_graph.jpg "model_graph")

Finally,good luck!
===================
