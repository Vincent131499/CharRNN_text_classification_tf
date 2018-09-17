# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     CharRnn_Model
   Description :   构建RNN模型，一般思路为：RNNCell -> dropout -> MultiRNNCell -> 重复time_steps步
   Author :       Stephen
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
__author__ = 'Stephen'

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import numpy as np

class Model(object):
    def __init__(self, args, training = True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        #二层RNN，需要将rnn_size作为参数传入到rnn_cell中
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        #通过cells列表，构建多层RNN（MultiRNNCell函数只实现了一个时间步的多层rnn）
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        #占位符创建
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length])
        self.targets = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length])

        self.initial_state = cell.zero_state(args.batch_size, tf.float32) #将初始状态设为0

        #定义需要训练的权重和偏置，因为需要和[batch_size, rnn_size]大小的split片相乘
        #所以需要定义shape为[args.rnn_size, args.vocab_size]
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable('softmax_w',
                                        shape=[args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b',
                                        shape=[args.vocab_size])

        #嵌入层，随机进行初始化
        embedding = tf.get_variable('embedding', shape=[args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        #对输入数据进行dropout
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        #split函数，将每一个batch切割成seq_length个切片
        #即：[batch_size, seq_length, rnn_size] -> [batch_size, 1, rnn_size]

        """
        split(
                value,    #输入张量
                num_or_size_splits,  #每个分割后的张量的尺寸
                axis=0,  #被分张量的分割标准//当axis=0时按行分，当axis=1时，按列分。
                num=None,
                name='split'
            )

        """
        inputs = tf.split(inputs, args.seq_length, 1) #切割成seq_length个切片

        #squeeze()函数，将大小为1的维度去掉，因此每一片的维度从[batch_size, 1, rnn_size]变成[batch_size , rnn_size]
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        #loop函数连接num_steps步的rnn_cell，将h(t-1)的输出prev做变换然后传入h(t)作为输入
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))

            return  tf.nn.embedding_lookup(embedding, prev_symbol)

        """
        该函数实现了一个简单的多层rnn模型。上面的MultiRNNCell函数构造了一个时间步的多层rnn，
        本函数则实现将其循环num_steps个时间步。
        :param decoder_inputs：输入列表，是一个长度为num_steps的列表，
                            每个元素是[batch_size, input_size]的2-D维的tensor
        :param initial_state：初始化状态，2-D的tensor，shape为 [batch_size x cell.state_size].
        :param cell：RNNCell
        :param loop_function：如果不为空，则将该函数应用于第i个输出以得到第i+1个输入，
                此时decoder_inputs变量除了第一个元素之外其他元素会被忽略。其形式定义为：loop(prev, i)=next。
                prev是[batch_size x output_size]，i是表明第i步，next是[batch_size x input_size]。
        """
        #rnn_decoder生成outputs 和final state
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')

        out_concat = tf.concat(outputs, 1)
        output = tf.reshape(out_concat, [-1, args.rnn_size])

        #输出层
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        #计算loss
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                       [tf.reshape(self.targets, [-1])],
                                                       [tf.ones([args.batch_size * args.seq_length])])
        # self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables() #获取训练的变量

        #计算梯度，梯度截断
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

