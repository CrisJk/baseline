# -*- coding:utf-8 -*-

"""

@Author: jun kuang
@Filename: embedding.py
@Date: 

"""
import os
import numpy as np
from collections import Counter

def load_wordVec(data_path,word_dim):
    wordMap = {}
    wordMap['PAD'] = len(wordMap)
    wordMap['UNK'] = len(wordMap)
    word_embed = []
    for line in open(os.path.join(data_path, 'word2vec.txt'), encoding='utf8'):
        content = line.strip().split()
        if len(content) != word_dim + 1:
            continue
        wordMap[content[0]] = len(wordMap)
        word_embed.append(np.asarray(content[1:], dtype=np.float32))

    word_embed = np.stack(word_embed)
    embed_mean, embed_std = word_embed.mean(), word_embed.std()

    pad_embed = np.random.normal(embed_mean, embed_std, (2, word_dim))
    word_embed = np.concatenate((pad_embed, word_embed), axis=0)
    word_embed = word_embed.astype(np.float32)
    return wordMap, word_embed

def load_wordMap(data_path,word_frequency):
    wordMap = {}
    wordMap['PAD'] = len(wordMap)
    wordMap['UNK'] = len(wordMap)
    all_content = []
    for line in open(os.path.join(data_path, 'sent_train.txt'),encoding='utf8'):
        all_content += line.strip().split('\t')[3].split()
    for item in Counter(all_content).most_common():
        if item[1] > word_frequency:
            wordMap[item[0]] = len(wordMap)
        else:
            break
    return wordMap
