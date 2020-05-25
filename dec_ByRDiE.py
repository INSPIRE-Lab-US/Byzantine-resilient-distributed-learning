'''
Decentralized Learning with ByRDiE

Run this script by specifying the number of Byzantine nodes and whether they should actually be faulty
Run this 10 times setting monte_trial to 0-9 to reproduce the results
'''
# Specify M nodes, N local samples, dataset, failure type, screening type, stepsize, connection rate, T iterations,
# b Byzantine agents

import numpy as np
from pre_proc import data_prep
from linear_classifier import linear_classifier
import tensorflow as tf
import time
import pickle
import random
import argparse
from DecLearning import DecLearning

parser = argparse.ArgumentParser()
parser.add_argument("monte_trial", help="Specify which monte carlo trial to run", type=int)
parser.add_argument("-b","--byzantine", help="Number of Byzantine nodes to defend against, if none defaults to 0", type=int)
parser.add_argument("-gb", "--goByzantine", help="Boolean to indicate if the specified number of Byzantine nodes actually send out faulty values"
                    , type=bool)

args = parser.parse_args()

monte_trial = args.monte_trial
if args.byzantine:
    b = args.byzantine
else:
    b = 0

if args.goByzantine:
    goByzantine = args.goByzantine
else:
    goByzantine = False

min_neighbor = 4*b+1

print(f'Starting Monte Carlo Trial {monte_trial}')
start = time.time()

#Reproducibility
np.random.seed(30+monte_trial)
random.seed(a=30+monte_trial)

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


r = 1
try:
    para, graph, neighbors, local_set, test_data, test_label, weights, save, t = chekpt_read(r)
    loaded = True
except:        
    para = DecLearning(dataset = 'MNIST', nodes=20, byzantine = b, local_samples=2000)
    loaded = False
    #Generate the graph
    para.gen_graph(min_neigh = min_neighbor)
    local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
    neighbors = para.get_neighbor()
    # checkpt_ini = byrdie_initialization(para, graph, neighbors, local_set, test_data, test_label)
    save = []
    t = 0
    # checkpt_ini.save(r)

#Initialization
tf.reset_default_graph()

#To ensure reproducibility 
tf.set_random_seed(30+monte_trial)

step_size=1e-1

w_nodes = [linear_classifier(stepsize = step_size) for node in range(para.M)]    
sess = para.initialization()

T = 100

if loaded:
    for w, node in zip(weights, w_nodes):
        node.assign(w, sess)

#ByRDiE Algorithm
for iteration in range(t + 1, T):


    #Iterates over the first 784 dimensions of the weights
    for p in range(7840):
        print(f'Dimension {p} of W')
        #Communication 
        para.communication_w(w_nodes, neighbors, p, sess, para.b, screen = True, goByzantine=goByzantine)
        #node update step    
        para.node_update_w(w_nodes, local_set, p, sess, stepsize=step_size/(iteration+1))
        if p%200 == 199: 
            #test over all test data
            accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
            print(f'Accuracy for iteration {iteration} is {accuracy}')
            save.append(accuracy)
    
    #Iterates over the last 10 dimensions of the bias vector for our linear classifier
    for p in range(10):
        para.communication_b(w_nodes, neighbors, p, sess, para.b, screen = True, goByzantine=goByzantine)
        para.node_update_b(w_nodes, local_set, p, sess, stepsize=step_size/(iteration+1))
    
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(accuracy)
    print(iteration)
    save.append(accuracy)


    
    # weights = [node.weights() for node in w_nodes]
    # checkpt_current = byrdie_checkpt(weights, save, iteration)
    # checkpt_current.pickle(r)
sess.close()

if b!=0 and goByzantine:
    filename = f'./result/ByRDiE/result_ByRDiE_b{b}_{monte_trial}.pickle'
else:
    filename = f'./result/ByRDiE/result_ByRDiE_b{b}_faultless_{monte_trial}.pickle'        

end = time.time()
print(f'Monte Carlo {monte_trial} Done!\n Time elapsed {end-start} seconds\n')

with open(filename, 'wb') as handle:
    pickle.dump(save, handle) 