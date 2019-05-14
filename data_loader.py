# -*- coding:utf-8 -*-

"""

@Author: jun kuang
@Filename: data_load.py
@Date: 

"""
import numpy as np
import os

class txt_file_data_loader():
    def __init__(self,sentence_dict,data_path,file_name,num_classes,batch_size,bag,padding = False,shuffle=True):
        self.sentence_dict = sentence_dict
        self.data_path = data_path
        self.file_name = file_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.bag = bag
        self.padding = padding
        self.shuffle = shuffle

    def data_loader(self):
        if self.bag:
            self.all_bags = []
            self.all_sents = []
            self.all_labels = []
            with open(os.path.join(self.data_path, self.file_name), 'r', encoding='utf8') as fr:
                for line in fr:
                    rel = [0] * self.num_classes
                    try:
                        bag_id, _, _, sents, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(
                                    type_list) > 1 and tp == '0':  # if a bag has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        bag_id, _, _, sents = line.strip().split('\t')

                    sent_list = []
                    for sent in sents.split():
                        sent_list.append(self.sentence_dict[sent])

                    self.all_bags.append(bag_id)
                    self.all_sents.append(np.concatenate(sent_list, axis=0))
                    self.all_labels.append(np.asarray(rel, dtype=np.float32))

            data_size = len(self.all_bags)
            datas = self.all_bags
        else:
            self.all_sent_ids = []
            self.all_sents = []
            self.all_labels = []
            with open(os.path.join(self.data_path, self.file_name), 'r', encoding='utf8') as fr:
                for line in fr:
                    rel = [0] * self.num_classes
                    try:
                        sent_id, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(
                                    type_list) > 1 and tp == '0':  # if a sentence has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        sent_id = line.strip()

                    self.all_sent_ids.append(sent_id)
                    self.all_sents.append(self.sentence_dict[sent_id])

                    self.all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes)))

            data_size = len(self.all_sent_ids)
            datas = self.all_sent_ids

            self.all_sents = np.concatenate(self.all_sents, axis=0)
            self.all_labels = np.concatenate(self.all_labels, axis=0)

        return data_size,datas

    def data_batcher(self):

        data_size,datas = self.data_loader()

        data_order = list(range(data_size))

        if self.bag:
            if self.shuffle:
                np.random.shuffle(data_order)
            if self.padding:
                if data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                total_sens = 0
                out_sents = []
                out_sent_nums = []
                out_labels = []
                for k in data_order[i * self.batch_size:(i + 1) * self.batch_size]:
                    out_sents.append(self.all_sents[k])
                    out_sent_nums.append(total_sens)
                    total_sens += self.all_sents[k].shape[0]
                    out_labels.append(self.all_labels[k])

                out_sents = np.concatenate(out_sents, axis=0)
                out_sent_nums.append(total_sens)
                out_sent_nums = np.asarray(out_sent_nums, dtype=np.int32)
                out_labels = np.stack(out_labels)

                yield out_sents, out_labels, out_sent_nums
        else:

            data_order = list(range(data_size))
            if self.shuffle:
                np.random.shuffle(data_order)
            if self.padding:
                if data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                idx = data_order[i * self.batch_size:(i + 1) * self.batch_size]
                yield self.all_sents[idx], self.all_labels[idx], None
