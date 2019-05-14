# -*- coding:utf-8 -*-

from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import jieba
from tqdm import tqdm
import os
corpus_file_path = 'open_data/text.txt'

sentences = []

#jieba.load_userdict('dict.txt')

word_vec_file_name = 'word2vec.txt'

word2vec_path = 'word2vec.model'

def to_corpus_pattern(str):
    str = jieba.cut(str)
    str = ' '.join(str)

    return str

if(os.path.exists(word2vec_path)):
    model = word2vec.Word2Vec.load("word2vec.model")


else:
    with open(corpus_file_path,'r',encoding='utf8') as fr:
        for line in tqdm(fr.readlines()):
            str = to_corpus_pattern(line.strip())
            sentences.append(str)

    with open('corpus.txt','w',encoding='utf8') as fw:
        for sen in sentences:
            fw.write(sen+'\n')


    model = word2vec.Word2Vec(LineSentence('corpus.txt'), sg=1, size=300, window=5, min_count=10 ,negative=5, sample=1e-4, workers=7)
    model.save(word2vec_path)

with open(word_vec_file_name,'w',encoding='utf8') as fw:
        word_count = len(model.wv.vocab)
        word_vec_dim = 300
        fw.write(str(word_count)+" "+str(word_vec_dim)+'\n')
        for key in tqdm(model.wv.vocab):
            fw.write(key+" "+" ".join([str(x) for x in model.wv[key]])+'\n')

#model.wv.save_word2vec_format(word_vec_file_name,binary=False)




