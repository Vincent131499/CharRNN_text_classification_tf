# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       Stephen
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
__author__ = 'Stephen'

import collections
import numpy as np

data_path = 'W:\\my_code\\python\\NLP\\My_NLP_Project\\CharRNN_text_classification_tf\data\\tinyshakespeare\\input.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read()

    data1 = f.read()

print('len(data):', len(data))
print(type(data))

print('len(data1):', len(data1))
print(type(data1))

counter = collections.Counter(data) #统计每个字符出现多少次
print('counter:', counter)
print('counter.items:', counter.items())

count_pairs= sorted(counter.items(), key=lambda x: -x[1]) #按照键值进行排序，实现了倒序排列

print('sorted:', count_pairs)

chars, _ = zip(*count_pairs)
print('chars:', chars)

print('len(vocab):', len(chars))

vocab = dict(zip(chars, range(len(chars))))
print('vocab:', vocab)

tensor = np.array(list(map(vocab.get, data))) #得到data各个字符对应的索引
print('tensor:', tensor)
print('tensor.shape:', tensor.shape)

# np.save('data.npy', tensor)

s = '我喜欢胡文坛，很喜欢很喜欢那种'

print('句子为：', s)
counter = collections.Counter(s) #统计每个字符出现多少次
print('counter:', counter)
print('counter.items:', counter.items())

count_pairs= sorted(counter.items(), key=lambda x: -x[1]) #按照键值进行排序，实现了倒序排列

print('sorted:', count_pairs)

chars, _ = zip(*count_pairs)
print('chars:', chars)

print('len(vocab):', len(chars))
