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

def gen_graph(nodes, con_rate, b=0):
   '''
   Generates a graph with nodes (M), a specified connection rate (con_rate) and a number of Byzantine nodes (b)
   ''' 
   
   re = 1                      # regenerate if graph assumption not satisfied
   while re:
       # Generating the adjacency matrix
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
       if all([sum(row) >= 2 * b + 1 for row in graph]):
           re = 0
   return w, graph   

def get_neighbor(G):
   neighbor_list = []
   for node in G:
       neighbor = []
       for ind, n in enumerate(node):
           if n == 1:
               neighbor.append(ind)
       neighbor_list.append(neighbor)
   return neighbor_list

def one_hot(label):
   l_oh = []
   for i in label:
       new_l = [0] * 10
       new_l[i] = 1
       l_oh.append(new_l)
   return l_oh


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
   _w = [node.weights()[0].reshape([7840,1]) for node in W]
   _b = [node.weights()[1] for node in W]
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
   _w = [node.weights()[0] for node in W]
   _b = [node.weights()[1] for node in W]
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
        with open('./checkpt/ByRDiE_checkpt_current_%d.pickle'%r, 'wb') as handle:
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
        with open('./checkpt/ByRDiE_checkpt_ini_%d.pickle'%r, 'wb') as handle:
            pickle.dump(self, handle)


def chekpt_read(r):
    '''
    Attempts to read the saved initialization and checkpoint for the ByRDiE experiment and returns all parameters relevant to continuing the experiment
    '''
    with open('./checkpt/ByRDiE_checkpt_ini_%d.pickle'%r, 'rb') as handle: 
        ini = pickle.load(handle)
    with open('./checkpt/ByRDiE_checkpt_current_%d.pickle'%r, 'rb') as handle: 
        current = pickle.load(handle)
    
    return ini.parameters, ini.graph, ini.neighbors, ini.local_data, ini.test_data, ini.test_label, current.weights, current.save, current.iteration

if __name__ == "__main__":
    r = 1
    try:
        para, graph, neighbors, local_set, test_data, test_label, weights, save, t = chekpt_read(r)
        loaded = True
    except:        
        para = experiment_parameters(agents=20, dataset='MNIST', localsize_N=2000, iteration=100,
                                  screen=False, b=0, Byzantine='random', stepsize = 1e-1)
        loaded = False
        #Generate the graph
        W_0, graph = gen_graph(para.M, para.con_rate, para.b)
        local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
        neighbors = get_neighbor(graph)
        checkpt_ini = byrdie_initialization(para, graph, neighbors, local_set, test_data, test_label)
        save = []
        t = 0
        checkpt_ini.save(r)
    
    #Initialization
    tf.reset_default_graph()
    w_nodes = [linear_classifier(stepsize = para.stepsize) for node in range(para.M)]    
    sess = initialization()
    if loaded:
        for w, node in zip(weights, w_nodes):
            node.assign(w, sess)
    
    for iteration in range(t + 1, para.T):
        #Diminishing step size
        step = para.stepsize/(iteration+1)

        #Iterates over the first 784 dimensions of the weights
        for p in range(784):
           #Communication 
           communication_w(w_nodes, neighbors, p, sess, para.b, screen = para.screen)
           #node update step    
           node_update_w(w_nodes, local_set, p, sess, stepsize=step)
           if p%200 == 199:  
                
           #test over all test data
                accuracy = [acc_test(node, test_data, test_label) for node in w_nodes]
                print(accuracy)
                save.append(accuracy)
        #Iterates over the last 10 dimensions of the bias vector for our linear classifier
        for p in range(10):
            communication_b(w_nodes, neighbors, p, sess, para.b, screen = para.screen)
            node_update_b(w_nodes, local_set, p, sess, stepsize=step)
        accuracy = [acc_test(node, test_data, test_label) for node in w_nodes]
        print(accuracy)
        print(iteration)
        save.append(accuracy)
        with open('./result/ByRDiE_faultless_%d.pickle'%r, 'wb') as handle:
            pickle.dump(save, handle) 
        weights = [node.weights() for node in w_nodes]
        checkpt_current = byrdie_checkpt(weights, save, iteration)
        checkpt_current.pickle(r)
    sess.close()
