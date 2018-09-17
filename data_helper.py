# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_helper
   Description :   数据处理文件
   Author :       Stephen
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
__author__ = 'Stephen'

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np

class TextLoader(object):
    def __init__(self, data_dir, batch_size, seq_length, encoding = 'utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, 'input.txt')
        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        tensor_file = os.path.join(data_dir, 'data.npy')

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            # 如果词库文件和数据文件不存在
            print('reading text file...')
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print('loading preprocessed files...')
            self.load_preprocessed(vocab_file, tensor_file)

        self.create_batches()
        self.reset_batch_pointer()


    #处理文本数据，生成data.npy和vocab.pkl（针对第一次训练）
    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, 'r', encoding=self.encoding) as f:
            data = f.read() #read()将一篇文档里的内容读取为一整个字符串
        print('len(data):', len(data)) #len(data):1115394
        counter = collections.Counter(data)  # 统计每个字符出现多少次
        # print('counter:', counter)
        # print('counter.items:', counter.items())

        count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 按照键值进行排序，实现了倒序排列

        # print('sorted:', count_pairs)
        #得到所有的字符
        self.chars, _ = zip(*count_pairs)
        # print('chars:', chars)
        self.vocab_size = len(self.chars)
        print('vocab_size:', self.vocab_size)

        #得到字符的索引
        self.vocab = dict(zip(self.chars, range(len(self.chars)))) #word2id

        #将字符写入文件
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        self.tensor = np.array(list(map(self.vocab.get, data)))  # 得到data各个字符对应的索引
        #使用numpy将numpy数组信息保存成本地文件*.npy
        np.save(tensor_file, self.tensor)

    """不是第一次训练，直接载入文件信息和词库信息"""
    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    """生成batch"""
    def create_batches(self):
        """
        tensor_size:1115000;
        batch_size:50;
        seq_length:100;
        num_batches:223
        """

        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length)) # num_batches=223

        if self.num_batches == 0:
            assert False, 'Not enough data, Make seq_length and batch_size small!'

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        #ydata为xdata的左循环移位，例如x:[1,2,3,4,5]，y就为[2,3,4,5,1]
        #y是x的下一个字符
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        #x_batches的shape就是223 * 50 * 100
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, axis=1)
        print('x_batches.shape:', np.array(self.x_batches).shape)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, axis=1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1

        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
