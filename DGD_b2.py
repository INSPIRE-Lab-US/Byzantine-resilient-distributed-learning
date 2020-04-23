# Specify M nodes, N local samples, dataset, failure type, screening type, stepsize, connection rate, T iterations,
# b Byzantine agents

import numpy as np
from pre_proc import data_prep
from linear_classifier import linear_classifier
import tensorflow as tf
import time
import pickle
import random
from screening_method import Byzantine_algs
import time
import sys

class experiment_parameters:
    def __init__(self, agents, dataset, localsize_N, iteration, stepsize=1e-4, 
                 screen=False, b=0, Byzantine='random', connection_rate=50):
        self.dataset = dataset
        self.M = agents
        self.T = iteration
        self.screen = screen
        self.b = b
        self.stepsize = stepsize
        self.N = localsize_N
        self.Byzantine = Byzantine
        self.con_rate = connection_rate

def gen_graph(nodes, con_rate, b=0, min_neigh = 0):
    '''
    Generates the adjacency matrix for a graph

    Args:
        nodes: Number of vertices in the graph
        con_rate: Connection rate of graph
        b: Number of Byzantine nodes 
    
    Returns:
        w : Relative weight of each edge
        graph: The adjacency matrix for the graph
    '''
    re = 1                      # regenerate if graph assumption not satisfied
    while re:
        graph = []
        for _ in range(nodes):
            graph.append([])
        for row in range(nodes):
            graph[row].append(1)
            for col in range(row + 1, nodes):
                d = random.randint(1, 100)
                if d < con_rate:
                    graph[row].append(1)     #form symmetric matrix row by row
                    graph[col].append(1)
                else:
                    graph[row].append(0)
                    graph[col].append(0)
        d_max = 0
        for row in graph:
            if sum(row) > d_max:
                d_max = sum(row)
        w = [row[:] for row in graph]
        for ind, row in enumerate(w):
            d = sum(row)
            w[ind] = [col/d_max for col in row]
            w[ind][ind] = 1 - (d - 1) / d_max
        if all([sum(row) >= min_neigh for row in graph]):
            re = 0
    return w, graph   

def get_neighbor(G):
    '''
    Returns a matrix where each row is a node and within each row the columns
    contain the respective node's neighbors

    Args:
        G: An adjacency matrix of a graph
    
    Returns:
        neighbor_list: A matrix where the ith row contains the neighboring nodes of the ith node
    '''
    neighbor_list = []
    for node in G:
        neighbor = []
        for ind, n in enumerate(node):
            if n == 1:
                neighbor.append(ind)
        neighbor_list.append(neighbor)
    return neighbor_list

def one_hot(label):
    '''
    Return ones hot encoding of label
    '''
    l_oh = []
    for i in label:
        new_l = [0] * 10
        new_l[i] = 1
        l_oh.append(new_l)
    return l_oh

def Byzantine(target, strategy='random'):
    '''
    Byzantine update method

    Arg:
        target: The parameter of the node undergoing Byzantine failure
        strategy: Method of failure
    
    Returns:
        fal: A numpy array with same dimensions as the target
    '''
    if strategy == 'random':
        #Creates a random array with values from a random uniform distribution [5,15)
        fal = np.random.random(target.shape)*10 + 5
    return fal


def initialization():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    return sess


def screen(grad, b, method):
    screened = Byzantine_algs[method](grad, b)
    return screened
    
def acc_test(model, t_data, t_label):
    acc = model.accuracy.eval(feed_dict={
            model.x:t_data, model.y_: t_label,})
    return acc
    
def communication(W, neighbor, sess, b=0, screen=False):
    '''
    Communicate the model (W,b) to all neighbors for each node

    Args:
        W: Nodes in our network
        neighbor: A matrix where the ith row contains an array of the neighbors for the ith node

    '''
    wb = [node.weights() for node in W]
    ave_w = []
    ave_b = []
    for neighbor_list in neighbor:
        neighborhood_w = [wb[n][0] for n in neighbor_list]
        neighborhood_b = [wb[n][1] for n in neighbor_list] 
        if screen:
            neighborhood_w = np.sort(neighborhood_w, axis = 0) 
            neighborhood_w = neighborhood_w[b : -b]
            neighborhood_b = np.sort(neighborhood_b, axis = 0)
            neighborhood_b = neighborhood_b[b : -b]
        neighborhood_w = np.mean(neighborhood_w, axis = 0)
        neighborhood_b = np.mean(neighborhood_b, axis = 0)
        ave_w.append(neighborhood_w)
        ave_b.append(neighborhood_b)

    for node, w, b in zip(W, ave_w, ave_b):
        node.assign([w, b], sess) 

def node_update(W, data, sess, stepsize=1e-3):        
    for model, sample, label in zip(W, data.dist_data, data.dist_label):
        sess.run(model.train_step, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
        


if __name__ == "__main__":

        #Setting random seed for reproducibility
        np.random.seed(30)
        random.seed(a=30)
        
        for ep in range(10):
            print(f'Monte Carlo {ep}')
            start = time.time()

            para = experiment_parameters(agents=20, dataset='MNIST', localsize_N=2000, iteration=100,
                                        screen=False, b=2, Byzantine='random', stepsize = 1e-1)
            #Generate the graph
            W_0, graph = gen_graph(para.M, para.con_rate, para.b, min_neigh=4*para.b + 1)    
            local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
            neighbors = get_neighbor(graph)

            #Initialization
            tf.reset_default_graph()

            #To ensure reproducibility 
            tf.set_random_seed(30+ep)

            w_nodes = [linear_classifier(stepsize = para.stepsize, sigma2=0.01) for node in range(para.M)]
            sess = initialization()
            rec = []

            for iteration in range(para.T):

                #Communication 
                communication(w_nodes, neighbors, sess, para.b, para.screen)

                #test over all test data
                accuracy = [acc_test(node, test_data, test_label) for node in w_nodes]
                rec.append(np.mean(accuracy))
                
                with open(f'./result/DGD/result_DGD_b2_{ep}.pickle', 'wb') as handle:
                    pickle.dump(rec, handle)
                
                #Node update using GD
                #Byzantine failed nodes assigned first
                for byzant in range(para.b):
                    W_byz = Byzantine(w_nodes[byzant].weights()[0])
                    b_byz = Byzantine(w_nodes[byzant].weights()[1])

                    w_nodes[byzant].assign([W_byz,b_byz],sess)
                
                #Gradient Descent update for the remaining nodes
                node_update(w_nodes[para.b:], local_set, sess, stepsize=para.stepsize/(iteration+1))

            print(f'Monte Carlo {ep} done!')
            end = time.time()
            print(f'Time Elapsed {end-start} sec')

            sess.close()

