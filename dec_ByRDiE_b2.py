# Decentralized learning with ByRDiE

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
        min_neigh : Minimum number of neighbors for each node

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
                # maximum number of neighbors for any single node 
                d_max = sum(row)

        # w is another graph matrix where the neighbors of each node are indicated by a 0 or 1/d_max
        w = [row[:] for row in graph]
        for ind, row in enumerate(w):
            d = sum(row)
            w[ind] = [col/d_max for col in row]
            w[ind][ind] = 1 - (d - 1) / d_max
            
        # Checks if the number of neighbors for each node is greater than 2b+1 
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
    Return one hot endcoding of labels
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
        #Creates a random array with values from a random uniform distribution [-1,0)
        fal = np.random.random(target.shape) - 1
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
   
def communication_w(W, neighbor, p, sess, b=0, screen=False):
    '''
    Communicate the dimension p of model W to all neighbors for each node, updates
    b nodes using Byzantine update

    Args:
        W: Nodes in our network
        neighbor: A matrix where the ith row contains an array of the neighbors for the ith node
        p: Dimension of W model
        sess: TensorFlow session
        b: Number of Byzantine nodes
        screen: Boolean to screen for Byzantines

    '''
    _w = [node.weights()[0].reshape([7840,1]) for node in W]
    _b = [node.weights()[1] for node in W]
    
    #Byzantine failure
    for node in range(b):
        #Set to a random number [-1,0)
        _w[node][p] = Byzantine(_w[node][p])

    ave = []
    for neighbor_list in neighbor:
        neighborhood = [_w[n][p] for n in neighbor_list]

        if screen:
            neighborhood = np.sort(neighborhood, axis = 0)
            neighborhood = neighborhood[b : -b]
        neighborhood = np.mean(neighborhood, axis = 0)
    #        print(neighborhood.shape)
        ave.append(neighborhood)
    for scalar, node in zip(ave, _w):
        node[p] = scalar

    for node, ww, bb in zip(W, _w, _b):
        node.assign([ww.reshape([784, 10]), bb], sess) 
        
def communication_b(W, neighbor, p, sess, b=0, screen=False):
    '''
    Communicate the model b to all neighbors for each node

    Args:
        W: Nodes in our network
        neighbor: A matrix where the ith row contains an array of the neighbors for the ith node

    '''
    _w = [node.weights()[0] for node in W]
    _b = [node.weights()[1] for node in W]
    
    #Byzantine failure
    for node in range(b):
        #Set to a random number [-1,0)
        _b[node][p] = Byzantine(_b[node][p])

    ave = []
    for neighbor_list in neighbor:
        neighborhood = [_b[n][p] for n in neighbor_list]

        if screen:
            neighborhood = np.sort(neighborhood, axis = 0)
            neighborhood = neighborhood[b : -b]
        neighborhood = np.mean(neighborhood, axis = 0)
    #        print(neighborhood.shape)
        ave.append(neighborhood)
    for scalar, node in zip(ave, _b):
        node[p] = scalar

    for node, ww, bb in zip(W, _w, _b):
        node.assign([ww, bb], sess) 

def node_update_w(W, data, p, sess, stepsize=1e-4):        
   for model, sample, label in zip(W, data.dist_data, data.dist_label):
       g = sess.run(model.gradient_w, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
       new_g = np.zeros([7840, 1])
       new_g[p] = g.reshape([7840, 1])[p]
       sess.run(model.local_update_w, feed_dict={model.gradient_port_w: new_g.reshape([784,10]), model.stepsize: stepsize})
       
def node_update_b(W, data, p, sess, stepsize=1e-4):        
   for model, sample, label in zip(W, data.dist_data, data.dist_label):
       g = sess.run(model.gradient_b, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
       new_g = np.zeros([10])
       new_g[p] = g[p]
       sess.run(model.local_update_b, feed_dict={model.gradient_port_b: new_g, model.stepsize: stepsize})

#checkpoint
class byrdie_checkpt:
    def __init__(self, weights, save, iteration):
        self.weights = weights
        self.save = save
        self.iteration = iteration
        
    def pickle(self, r):
        with open('./checkpt/ByRDiE_b4_checkpt_current_%d.pickle'%r, 'wb') as handle:
            pickle.dump(self, handle)
        
class byrdie_initialization:
    def __init__(self, exp_parameters, graph, neighbors, local_data, test_data, test_label):
        self.parameters = exp_parameters
        self.graph = graph
        self.neighbors = neighbors
        self.local_data = local_data
        self.test_data = test_data
        self.test_label = test_label
    def save(self, r):
        with open('./checkpt/ByRDiE_b2_checkpt_ini_%d.pickle'%r, 'wb') as handle:
            pickle.dump(self, handle)
        
def chekpt_read(r):
    with open('./checkpt/ByRDiE_b2_checkpt_ini_%d.pickle'%r, 'rb') as handle: 
        ini = pickle.load(handle)
    with open('./checkpt/ByRDiE_b2_checkpt_current_%d.pickle'%r, 'rb') as handle: 
        current = pickle.load(handle)
        
    return ini.parameters, ini.graph, ini.neighbors, ini.local_data, ini.test_data, ini.test_label, current.weights, current.save, current.iteration
    


if __name__ == "__main__":
    #Setting random seed for reproducibility
    random.seed(a=30)
    np.random.seed(30)

    for ep in range(10):        
        para = experiment_parameters(agents=20, dataset='MNIST', localsize_N=2000, iteration=100,
                                    screen=True, b=2, Byzantine='random', stepsize = 1e-1)
        #Generate the graph
        W_0, graph = gen_graph(para.M, para.con_rate, para.b, 4*para.b+1)    
        local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
        neighbors = get_neighbor(graph)
        rec = []
        t = 0

        #Initialization
        tf.reset_default_graph()

        #Ensure reproducibility
        tf.set_random_seed(30+ep)

        w_nodes = [linear_classifier(stepsize = para.stepsize) for node in range(para.M)]    
        sess = initialization()

        save = []
        for iteration in range(para.T):
            step = para.stepsize/(iteration+1)

            for p_w in range(7840):
                #Communication 
                communication_w(w_nodes, neighbors, p_w, sess, para.b, screen = para.screen)

                #node update using DGD    
                node_update_w(w_nodes, local_set, p_w, sess, stepsize=step)
            
            for p_b in range(10):
                communication_b(w_nodes, neighbors, p_b, sess, para.b, screen = para.screen)
                node_update_b(w_nodes, local_set, p_b, sess, stepsize=step)
            

            accuracy = [acc_test(node, test_data, test_label) for node in w_nodes]
            print(f'Accuracy for iteration {iteration} is {accuracy}')
            print(f'Total scalar broadcasts = {(iteration+1)*(p_w+p_b)}')
            rec.append(accuracy)

            with open('./result/ByRDiE/ByRDiE_b2_%d.pickle'%ep, 'wb') as handle:
                pickle.dump(rec, handle) 
        
        sess.close()
