#linear classifier for MNIST 



import tensorflow as tf
import numpy as np

def weight_variable(shape, std_dev):
    # Outputs a random matrix with mean 0 and stddev=0.1 from normal distribution
    # ensures all the random values are within 2 std dev from the mean
    initial = tf.truncated_normal(shape, stddev=std_dev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class linear_classifier:
    def __init__(self, stepsize=1e-4, sigma2 = 0.1):
        
        #Placeholder for our 784-dimensional data (MNIST data)
        self.x = tf.placeholder(tf.float32, shape=[None, 784])

        #Placeholder for the output of our linear classifier y=W*x+b
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
       
       # Creates the random weights matrix and bias vector
        self.W = weight_variable([784, 10])
        self.b = bias_variable([10])
       
        self.W_com = tf.placeholder(tf.float32, shape=[784, 10])
        self.b_com = tf.placeholder(tf.float32, shape=[10])
       
        self.y_inf = tf.matmul(self.x, self.W) + self.b

        #Cross entropy loss
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_inf))

        #L2 loss regularization parameters
        self.regularizer = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)

        #Objective/Loss function
        self.loss = self.cross_entropy + self.regularizer
        
        self.stepsize = tf.placeholder(tf.float32, shape=[])        
        self.optimizer = tf.train.GradientDescentOptimizer(self.stepsize)
        #self.optimizer = tf.train.AdamOptimizer(stepsize)      
        self.train_step = self.optimizer.minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.y_inf,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))        
       
        #Collection of variables
        self.layers = [self.W, self.b]    
        self.communication_ports = [self.W_com, self.b_com]
        
       
        #assign for communication
        self.communication = [var.assign(port) for var, port in zip(self.layers, self.communication_ports)]
       
        #separate for 1-d update
        self.gradient_w = self.optimizer.compute_gradients(loss = self.cross_entropy, var_list = [self.W])[0][0]
        self.gradient_b = self.optimizer.compute_gradients(loss = self.cross_entropy, var_list = [self.b])[0][0]
        self.gradient_port_w = tf.placeholder(tf.float32, shape=[784, 10])
        self.gradient_port_b = tf.placeholder(tf.float32, shape=[10])
        self.local_update_w = self.optimizer.apply_gradients([[self.gradient_port_w, self.W]])
        self.local_update_b = self.optimizer.apply_gradients([[self.gradient_port_b, self.b]])

   
    def weights(self):
        '''
        Return list containing weight matrix and offset vector [W,b]
        '''
        W = self.W.eval()
        b = self.b.eval()

        weight = [W, b]
        return weight
   
    def assign(self, weight, sess):
        for layer, op, port in zip(weight, self.communication, self.communication_ports):
            sess.run(op, feed_dict={port: layer})
       

       
       
       
       
       
       
