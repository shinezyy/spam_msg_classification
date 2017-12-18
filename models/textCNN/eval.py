#encoding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_prepare
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import jieba

#data parameters
tf.flags.DEFINE_string('unlabeled','./spam_data/unlabeled.txt','')

#eval parameters
tf.flags.DEFINE_integer('batch_size',2048,'')
tf.flags.DEFINE_string('checkpoint_dir','./check_points','')

#tensorflow parameters
tf.flags.DEFINE_boolean('allow_soft_placement',True,'')
tf.flags.DEFINE_boolean('log_device_placement',False,'')

FLAGS = tf.flags.FLAGS

#load test data
test_samples = list(open(FLAGS.unlabeled, 'r').readlines())
print(len(test_samples))
test_samples = [s.replace(' ','') for s in test_samples]
print("测试数据分词开始。。。")
test_samples = [list(jieba.cut(s, cut_all=False)) for s in test_samples]
test_samples = [' '.join(s[1:-1]) for s in test_samples]
print("测试数据分词结束。。。")

#map data into vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor.restore('./result')
x_test = np.array(list(vocab_processor.transform(test_samples)))
print(x_test.shape)
print('\n开始测试<(￣︶￣)>...\n')

#Evaluation
#========================================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)
    sess= tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        #get placeholders
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

        #output
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        all_predictions = []

        begin_ind = 0
        end_ind = begin_ind + FLAGS.batch_size
        counter = 1
        while(end_ind <= x_test.shape[0]):
            print("batch number: {}".format(counter))
            counter+=1
            test_batch = x_test[begin_ind:end_ind,:]
            batch_predictions = sess.run(predictions,{input_x:test_batch,dropout_keep_prob:1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

            begin_ind = end_ind
            end_ind = begin_ind + FLAGS.batch_size
            if end_ind > x_test.shape[0] and begin_ind < x_test.shape[0]:
                end_ind = x_test.shape[0]
            if begin_ind >= x_test.shape[0]:
                break
        
        all_predictions = np.row_stack((all_predictions))
        print('Saving evaluations....')
        with open('./results/predictions.csv' ,'w') as f:
            csv.writer(f).writerows(all_predictions)




