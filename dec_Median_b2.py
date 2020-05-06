# Decentralized learning with BRIDGE using Median screening
# The BRIDGE Algo assumes 2 Byzantine nodes and 2 nodes in the network actually undergo Byzantine failure

import numpy as np
from pre_proc import data_prep
from linear_classifier import linear_classifier
import tensorflow as tf
import time
import pickle
import random
import sys
from DecLearning import DecLearning
from screening_method import Byzantine_algs
      
start = time.time()

monte_trial = int(sys.argv[1])
print(f'Starting Monte Carlo trial {monte_trial}', flush=True)

#Setting random seed for reproducibility
np.random.seed(30+monte_trial)
random.seed(a=30+monte_trial)

para = DecLearning(dataset = 'MNIST', nodes=20, iterations=100, byzantine=2, local_samples=2000, 
                              con_rate=50, stepsize = 1e-1)
#Generate the graph
para.gen_graph(min_neigh = 4*para.b+1)    
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
neighbors = para.get_neighbor()

save = []

#Initialization
tf.reset_default_graph()

#To ensure reproducibility 
tf.set_random_seed(30+monte_trial)

w_nodes = [linear_classifier(stepsize = para.stepsize) for node in range(para.M)]    
sess = para.initialization()

#BRIDGE Algorithm
for iteration in range(para.T):
    #Communication 
    para.communication(w_nodes, neighbors, sess, b = para.b, screen = True, goByzantine=True,
     screenMethod='Median')
    
        
    #test over all test data
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(f'Accuracy for iteration {iteration} is {accuracy}', flush=True)
    save.append(np.mean(accuracy))
    
    #node update using GD   
    para.node_update(w_nodes, local_set, sess, stepsize=para.stepsize/(iteration+1))


sess.close()
end = time.time()
print(f'Monte Carlo {monte_trial} Done!\n Time elapsed {end-start} seconds\n', flush=True)
with open(f'./result/Median/result_Median_b2_{monte_trial}.pickle', 'wb') as handle:
    pickle.dump(save, handle)
