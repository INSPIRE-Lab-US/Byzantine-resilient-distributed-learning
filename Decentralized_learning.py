# %load Distributed_learning.py
# Distributed learning

# Specify M nodes, N local samples, dataset, failure type, screening type, stepsize, connection rate, T iterations,
# b Byzantine agents

import numpy as np
from pre_proc import data_prep
from linear_classifier import linear_classifier
import tensorflow as tf
import time
from screening_method import Byzantine_algs
from Byzantine_strategy import Byzantine_strategy
import pickle


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

def gen_graph(nodes, con_rate, b=0):          # connecting rate between 1-100
   
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
   
def communication(W, neighbor, sess, b=0, screen=False):
   wb = [node.weights() for node in W]
   

   ave = []
   for node in range(len(W)):
       neighborhood = [wb[n][layer] for n in neighbor[node]]

#         if screen:
#             neighborhood = np.sort(neighborhood, axis = 0)
#             neighborhood = neighborhood[b : -b]
       neighborhood = np.mean(neighborhood, axis = 0)
   #        print(neighborhood.shape)
       ave.append(neighborhood)
   for node, weight in zip(W,ave):
       node.assign(weight, sess) 

def node_update(W, data, sess):        
   for model, sample, label in zip(W, data.dist_data, dist_label):
       sess.run(node.train_step, feed_dict={model.x: sample, model.y_: label})
       


if __name__ == "__main__":
   para = experiment_parameters(agents=20, dataset='MNIST', localsize_N=2000, iteration=1000,
                                  screen=False, b=0, Byzantine='random')
   #Generate the graph
   W_0, graph = gen_graph(para.M, para.con_rate, para.b)    
   local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
   neighbors = get_neighbor(graph)
   #Initialization
   tf.reset_default_graph()
   w_nodes = [linear_classifier(stepsize = para.stepsize) for node in range(para.M)]    
   sess = initialization()

   for iteration in range(para.T):
       #Communication 
       w_nodes = communication(w_nodes, neighbors, para.b, para.screen)
       #node update using Adam    
       node_update(w_nodes, local_set, sess, para.b, screening = para.screen)
       if iteration%100 == 99:            
       #test over all test data
           accuracy = [acc_test(node, test_data, test_label) for node in w_nodes]

           print(accuracy)

   
   sess.close()
