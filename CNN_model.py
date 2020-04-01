#CNN model for MNIST 
#conv--pool--conv--pool--fc--fc


import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class CNN:
    def __init__(self, stepsize=1e-4):
       
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        # g is for apply gradient, com is for assign
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.W_conv1_g = tf.placeholder(tf.float32, shape=[5,5,1,32])
        self.W_conv1_com = tf.placeholder(tf.float32, shape=[5,5,1,32])
        
        self.b_conv1 = bias_variable([32])
        self.b_conv1_g = tf.placeholder(tf.float32, shape=[32])
        self.b_conv1_com = tf.placeholder(tf.float32, shape=[32])
        
        self.x_image = tf.reshape(self.x, [-1,28,28,1])
        
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)        
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.W_conv2_g = tf.placeholder(tf.float32, shape=[5, 5, 32, 64])
        self.W_conv2_com = tf.placeholder(tf.float32, shape=[5, 5, 32, 64])
        
        self.b_conv2 = bias_variable([64])
        self.b_conv2_g = tf.placeholder(tf.float32, shape=[64])
        self.b_conv2_com = tf.placeholder(tf.float32, shape=[64])
        
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)
        
        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.W_fc1_g = tf.placeholder(tf.float32, shape=[7*7*64, 1024])
        self.W_fc1_com = tf.placeholder(tf.float32, shape=[7*7*64, 1024])
        
        self.b_fc1 = bias_variable([1024])
        self.b_fc1_g = tf.placeholder(tf.float32, shape=[1024])
        self.b_fc1_com = tf.placeholder(tf.float32, shape=[1024])
        
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        
        self.W_fc2 = weight_variable([1024, 10])
        self.W_fc2_g = tf.placeholder(tf.float32, shape=[1024, 10])
        self.W_fc2_com = tf.placeholder(tf.float32, shape=[1024, 10])
        
        self.b_fc2 = bias_variable([10])
        self.b_fc2_g = tf.placeholder(tf.float32, shape=[10])
        self.b_fc2_com = tf.placeholder(tf.float32, shape=[10])
        
        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        self.cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        
        
        self.optimizer = tf.train.AdamOptimizer(stepsize)
        
        
        
        self.train_step = self.optimizer.minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))        
        
        #Collection of variables
        self.layers = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        self.calculated_gradient = [self.W_conv1_g, self.b_conv1_g, self.W_conv2_g, self.b_conv2_g, self.W_fc1_g, self.b_fc1_g, self.W_fc2_g, self.b_fc2_g]
        self.communication_ports = [self.W_conv1_com, self.b_conv1_com, self.W_conv2_com, self.b_conv2_com, self.W_fc1_com, self.b_fc1_com, self.W_fc2_com, self.b_fc2_com]
        
        #apply gradients from outside
        self.g_var_pair =  [[grad, vari] for grad, vari in zip(self.calculated_gradient, self.layers)]
        self.calculated_update = self.optimizer.apply_gradients(self.g_var_pair)
        
        #gradient step using local gradient
        self.gradient = self.optimizer.compute_gradients(loss = self.cross_entropy, var_list = self.layers)
        self.local_update = self.optimizer.apply_gradients(self.gradient)  
        
        #assign for communication
        self.communication = [var.assign(port) for var, port in zip(self.layers, self.communication_ports)]
        


    
    
    def weights(self):
        W_1 = self.W_conv1.eval()
        b_1 = self.b_conv1.eval()
        W_2 = self.W_conv2.eval()
        b_2 = self.b_conv2.eval()
        W_3 = self.W_fc1.eval()
        b_3 = self.b_fc1.eval()       
        W_4 = self.W_fc2.eval()
        b_4 = self.b_fc2.eval()
        weight = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]
        return weight
    
    def assign(self, weight, sess):
        for layer, op, port in zip(weight, self.communication, self.communication_ports):
            sess.run(op, feed_dict={port: layer})
        

        
        
        
        
        
        