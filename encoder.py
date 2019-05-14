# -*- coding:utf-8 -*-

import tensorflow as tf
"""

@Author: jun kuang
@Filename: encoder.py
@Date: 

"""


def CNN_encoder(word_embedding,input_word,pos_e1_embedding,pos_e2_embedding,input_pos_e1,input_pos_e2,window,word_dim,pos_dim,hidden_dim,sen_len,keep_prob):
    inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, input_word), \
                                               tf.nn.embedding_lookup(pos_e1_embedding, input_pos_e1), \
                                               tf.nn.embedding_lookup(pos_e2_embedding, input_pos_e2)])
    inputs_forward = tf.expand_dims(inputs_forward, -1)

    with tf.name_scope('conv-maxpool'):
        w = tf.get_variable(name='w', shape=[window, word_dim + 2 * pos_dim, 1, hidden_dim])
        b = tf.get_variable(name='b', shape=[hidden_dim])
        conv = tf.nn.conv2d(
            inputs_forward,
            w,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv')
        h = tf.nn.bias_add(conv, b)
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sen_len - window + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool')
    sen_reps = tf.tanh(tf.reshape(pooled, [-1, hidden_dim]))
    sen_reps = tf.nn.dropout(sen_reps, keep_prob)
    return sen_reps