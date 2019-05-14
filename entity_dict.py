# -*- coding:utf-8 -*-

"""

@Author: jun kuang
@Filename: entity_dict.py
@Date: 

"""

train_file_name = 'open_data/bag_relation_train.txt'
test_file_name = 'open_data/bag_relation_test.txt'
dev_file_name = 'open_data/bag_relation_dev.txt'

file_name=[train_file_name,test_file_name,dev_file_name]

entity_list = []

for f in file_name:
    with open(f,'r',encoding='utf8') as fr:
        for line in fr:
            sample = line.split()
            entity_list.append(sample[1])
            entity_list.append(sample[2])


with open('dict.txt','w',encoding='utf8') as fw:
    for entity in entity_list:
        fw.write(entity+'\n')
