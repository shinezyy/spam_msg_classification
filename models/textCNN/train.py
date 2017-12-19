#encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_prepare
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "./spam_data/labeled.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 2048, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer('max_iter',40000,'')

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

#Data Preparation
# ========================================================

#load data
print("准备数据。。。")
start = time.time()
train_x_pos, train_y_pos, train_x_neg, train_y_neg = data_prepare.load_data_and_labels(FLAGS.data_file)

# Build vocabulary
max_document_length_pos = max([len(x.split(" ")) for x in train_x_pos])
max_document_length_neg = max([len(x.split(" ")) for x in train_x_neg])
max_document_length = max(max_document_length_pos, max_document_length_neg)
print("max doc length is {}".format(max_document_length))

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_pos = np.array(list(vocab_processor.fit_transform(train_x_pos)))
x_neg = np.array(list(vocab_processor.fit_transform(train_x_neg))
)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(train_y_pos)))
x_shuffled_pos = x_pos[shuffle_indices]
y_shuffled_pos = np.array(train_y_pos)[shuffle_indices]

shuffle_indices = np.random.permutation(np.arange(len(train_y_neg)))
x_shuffled_neg = x_neg[shuffle_indices]
y_shuffled_neg = np.array(train_y_neg)[shuffle_indices]


#split data_set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(train_y_pos)))
x_train_pos, x_dev_pos = x_shuffled_pos[:dev_sample_index], x_shuffled_pos[dev_sample_index:]
y_train_pos, y_dev_pos = y_shuffled_pos[:dev_sample_index], y_shuffled_pos[dev_sample_index:]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(train_y_neg)))
x_train_neg, x_dev_neg = x_shuffled_neg[:dev_sample_index], x_shuffled_neg[dev_sample_index:]
y_train_neg, y_dev_neg = y_shuffled_neg[:dev_sample_index], y_shuffled_neg[dev_sample_index:]

#del train_x, train_y,x_shuffled,y_shuffled
end = time.time()
print("time cost in prepare data is {}s".format((end-start)))

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train_pos.shape[1],
            num_classes=y_train_pos.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_writer = tf.summary.FileWriter('./summaries', sess.graph)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        #write vocab
        vocab_processor.save('./result')

        #init
        sess.run(tf.global_variables_initializer())

        #train
        for iter in range(FLAGS.max_iter):
            start = time.time()
            #ind = [np.random.randint(0,x_train.shape[0]) for _ in range(FLAGS.batch_size)]
            #x_batch = x_train[ind]
            #y_batch = y_train[ind]
            ind = [np.random.randint(0,x_train_pos.shape[0]) for _ in range(FLAGS.batch_size/2)]
            x_batch_pos = x_train_pos[ind]
            y_batch_pos = y_train_pos[ind]
            ind = [np.random.randint(0,x_train_neg.shape[0]) for _ in range(FLAGS.batch_size/2)]
            x_batch_neg = x_train_neg[ind]
            y_batch_neg = y_train_neg[ind]
            
            x_batch = np.concatenate((x_batch_pos,x_batch_neg))
            y_batch = np.concatenate((y_batch_pos,y_batch_neg))

            ind = np.random.permutation(np.arange(y_batch.shape[0]))
            x_batch = x_batch[ind]
            y_batch = y_batch[ind]

            feed_dict = {cnn.input_x:x_batch, cnn.input_y:y_batch, cnn.dropout_keep_prob:0.5}
            _,step,summaries,loss,accuracy = sess.run([train_op, global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dict)
            end = time.time()
            train_summary_writer.add_summary(summaries, step)
            print("========Time cost:{}s per batch========".format((end-start)))
	    print('loss:{}       accuracy:{}'.format(loss,accuracy))
            current_step = tf.train.global_step(sess, global_step)

            if iter % FLAGS.evaluate_every == 0:
                ind_eval = [np.random.randint(0,x_dev_pos.shape[0]) for _ in range(2048)]
                x_eval_batch_pos = x_dev_pos[ind_eval]
                y_eval_batch_pos = y_dev_pos[ind_eval]
                ind_eval = [np.random.randint(0,x_dev_neg.shape[0]) for _ in range(2048)]
                x_eval_batch_neg = x_dev_neg[ind_eval]
                y_eval_batch_neg = y_dev_neg[ind_eval]                

                feed_dict = {cnn.input_x:x_eval_batch_pos,
                             cnn.input_y:y_eval_batch_pos,
                             cnn.dropout_keep_prob:1.0}
                step,loss,accuracy = sess.run([global_step, cnn.loss, cnn.accuracy],feed_dict)
                print("\nall positive accuracy:{}".format(accuracy))
                
                feed_dict = {cnn.input_x:x_eval_batch_neg,
                             cnn.input_y:y_eval_batch_neg,
                             cnn.dropout_keep_prob:1.0}
                step,loss,accuracy = sess.run([global_step, cnn.loss, cnn.accuracy],feed_dict)
                print("all negative accuracy:{}\n".format(accuracy))

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, "./check_points/model", global_step=current_step)



