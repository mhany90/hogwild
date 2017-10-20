from __future__ import print_function
from sklearn import datasets, metrics
from sklearn import preprocessing, model_selection
import sys
import time
import skflow
import numpy as np
import csv
import tensorflow as tf
import pandas as pd
filename_queue = tf.train.string_input_producer(["EI-reg-En-anger-train.combined.train.csv"])

def read_my_file_format(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row = reader.read(filename_queue)
    record_defaults =  [['a'] for cl in range((445))]
    parsed_line = tf.decode_csv(csv_row,record_defaults = record_defaults )
    label = parsed_line[1] #the target
    
    # first element is the target
    del parsed_line[1] # Delete first element
    del parsed_line[0]
    feature = parsed_line
    print(label)
    return feature, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, 
                                                    num_epochs=num_epochs, 
                                                    shuffle=True)
    feature, label = read_my_file_format(filename_queue)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    print(label_batch)
    return feature_batch, label_batch

input_data = "EI-reg-En-anger-train.combined.train.csv"
total_lines = len(open("EI-reg-En-anger-train.combined.train.csv", 'r').readlines()) - 1

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    drop_out = tf.nn.dropout(layer_1, keep_prob)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(drop_out, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output fully connected layer with a neuron for each class
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.sigmoid(tf.matmul(layer_4, weights['out']) + biases['out'])
    return out_layer

# cluster specification
parameter_servers = ["127.0.0.1:2289"]
#workers = [ "127.0.0.1:2230", "127.0.0.1:2231"]
workers = [ "127.0.0.1:2290"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

#start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# Parameters
learning_rate = 0.001
training_epochs = 200
display_step = 1
batch_size = 10

# Network Parameters
n_hidden_1 = 250 # 1st layer number of neurons
n_hidden_2 = 125 # 2nd layer number of neurons
n_hidden_3 = 60
n_hidden_4 = 20
n_input = 443
n_classes = 1


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):   
        # count the number of updates
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            X = tf.placeholder("float", [None, n_input], name="x-input")
            # target 10 output classes
            Y = tf.placeholder("float", [None, n_classes], name = "y-input")

            # tf Graph input
            keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            weights = {
            'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
            'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes]))
            }
        with tf.name_scope("biases"):
            biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
            'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_classes]))
            }

#Build model
        with tf.name_scope("regression"):
            intensity = multilayer_perceptron(X)
       
# specify cost function
        with tf.name_scope('loss'):
            loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, intensity))


#used to report correlation
        with tf.name_scope('pearson'): 
            pearson = tf.contrib.metrics.streaming_pearson_correlation(intensity, Y, name="pearson")

        # specify optimizer
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
            train_op = optimizer.minimize(loss_op, global_step=global_step)
    
# Initializing the variables
        init = tf.global_variables_initializer()

    features, labels = input_pipeline([input_data], batch_size)
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init)
#init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    begin_time = time.time()
    print(begin_time)
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,
    device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
    with sv.prepare_or_wait_for_session(server.target, config=config) as sess:
    # start populating filename queue
    # Training cycle
         for epoch in range(training_epochs):
             avg_cost = 0.
             print(total_lines)
             total_batch = int(total_lines/batch_size)
             print(total_batch)
        # Loop over all batches
             for x in range(total_batch) :
            # Run optimization op (backprop) and cost op (to get loss value
                 feature_batch, label_batch = sess.run([features, labels])
                 batch_y = np.reshape(label_batch, (-1, 1))
                 batch_x =  [x for x in feature_batch]
                 batch_x = np.reshape(batch_x, (-1, 443))
                 _, c, p = sess.run([train_op, loss_op, pearson], feed_dict={X: batch_x,
                                                             Y: batch_y, keep_prob : 1.0})
#                 print(p[0])  
        # Compute average loss
                 avg_cost += c / total_batch
        # Display logs per epoch step
             if epoch % display_step == 0:
                 print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
         sv.request_stop()        
         print("Optimization Finished!")
         elapsed_time = time.time() - begin_time
         print(elapsed_time)



