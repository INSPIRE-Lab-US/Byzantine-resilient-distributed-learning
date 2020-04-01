#  %%writefile dis_trimmed_mean_b4.py
# %%writefile Distributed_learning.py
# %load Distributed_learning.py
# Distributed learning

# Specify M agents, N local samples, dataset, failure type, screening type, stepsize, batchsize, T iterations,
# b Byzantine agents

import numpy as np
from pre_proc import data_prep
from CNN_model import CNN
import tensorflow as tf
import time
from screening_method import Byzantine_algs
from Byzantine_strategy import Byzantine_strategy
import pickle


class experiment_parameters:
    def __init__(self, agents, dataset, localsize_N, iteration, batchsize=0, stepsize=1, 
                 screen=False, b=0, Byzantine='random'):
        self.dataset = dataset
        self.M = agents
        self.T = iteration
        self.screen = screen
        self.b = b
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.N = localsize_N
        self.Byzantine = Byzantine
        if self.batchsize >  self.N:
            raise ValueError("Batch size is large than local datasize")


def data_distribution(dataset, nodes_M, sample_N):#samples will be distributed evenly to each node
    distributed_data, test_data, test_label = data_prep(dataset, nodes_M, nodes_M * sample_N, one_hot=True)
    return distributed_data, test_data, test_label

def initialization():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    return sess


def server_to_agent(server, nodes, sess):
    weights = server.weights()
    for node in nodes:
        node.assign(weights, sess)

def agent_calculation(node, models, sets, batchsize, sess):
    data, label = sets.next_batch(node, batchsize)
    w_local = models[node]
    grad = sess.run(w_local.gradient, feed_dict={w_local.x: data, 
                        w_local.y_: label, w_local.keep_prob: 0.5})
    # TensorFlow gradient is of the form [grad, variable] for each layer    
    local_gradient = []
    for pair in grad:
        local_gradient.append(pair[0])    
    return local_gradient

def Byzantine_gradient(strategy, node, models, sets, batchsize, sess):
    data, label = sets.next_batch(node, batchsize)
    w_local = models[node]
    Byz_gradient = Byzantine_strategy[strategy](data, label, w_local, sess)
    return Byz_gradient

def screen(grad, b, method):
    screened = Byzantine_algs[method](grad, b)
    return screened
    
def acc_test(model, t_data, t_label):
    acc = model.accuracy.eval(feed_dict={
            model.x:t_data, model.y_: t_label, model.keep_prob: 1.0})
    return acc
    
def server_update(model, gradient, sess, b, stepsize, screening=False):
    if screening:
        grad = screen(gradient, b, screening)
    else:
        grad = np.mean(gradient, axis = 0)
    g_var_pair =  [[grad, vari] for grad, vari in zip(grad, model.layers) ]
    sess.run(model.calculated_update, feed_dict={port: g for port, g in zip(model.calculated_gradient, grad)})

    return model


if __name__ == "__main__":
    for r in range(10):
        para = experiment_parameters(agents=20, dataset='MNIST', localsize_N=2000, iteration=10000,
                                     batchsize=50, stepsize=1e-4, screen='krum', b=9, Byzantine='random')
        local_set, test_data, test_label = data_distribution(para.dataset, para.M, para.N)
        #Initialization
        tf.reset_default_graph()
        w_server = CNN(stepsize = para.stepsize)
        w_nodes = []
        for node in range(para.M):
            w_nodes.append(CNN())

        sess = initialization()
        rec = []     
        for iteration in range(para.T):
            #Communication 
            server_to_agent(w_server, w_nodes, sess)
            #local gradient calculation
            gradient = []
                #Byzantine nodes are the first b nodes
            for node in range(para.b):    
                gradient.append(Byzantine_gradient(para.Byzantine, node, w_nodes, local_set, para.batchsize, sess))
            for node in range(para.b, para.M):
                gradient.append(agent_calculation(node, w_nodes, local_set, para.batchsize, sess))
            #Server update using Adam    
            server_update(w_server, gradient, sess, para.b, para.stepsize*iteration, screening = para.screen)
            if iteration%30 == 29:            
            #test over all test data
                accuracy = acc_test(w_server, test_data, test_label)
                rec.append(accuracy)
                print(accuracy)
                print(iteration)    
                with open('./result/result_krum_b9_%d.pickle'%r, 'wb') as handle:
                    pickle.dump(rec, handle)    
        sess.close()

