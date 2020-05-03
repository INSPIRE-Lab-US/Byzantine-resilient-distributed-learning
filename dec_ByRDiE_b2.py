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
from DecLearning import DecLearning
import sys

start = time.time()

monte_trial = int(sys.argv[1])
print(f'Starting Monte Carlo trial {monte_trial}', flush=True)

#Decentralized Learning object
para = DecLearning(dataset='MNIST', nodes=20,  iterations=100, byzantine = 2, local_samples=2000, 
        con_rate = 50, stepsize= 1e-1)

#Setting random seed for reproducibility
random.seed(a=30+monte_trial)
np.random.seed(30+monte_trial)

#Generate the graph
para.gen_graph(min_neigh= (4*para.b+1))

#Split data
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)

neighbors = para.get_neighbor()

save = []

tf.reset_default_graph()

#Ensure reproducibility
tf.set_random_seed(30+monte_trial)

w_nodes = [linear_classifier(stepsize = para.stepsize) for node in range(para.M)]    
sess = para.initialization()


save = []
for iteration in range(para.T):
    step = para.stepsize/(iteration+1)
    
    #Iterate over dimensions of flattened W matrix
    for dim in range(7840):
        #Communication 
        para.communication_w(w_nodes, neighbors, dim, sess, b = para.b, screen = True, goByzantine=True)
        
        para.node_update_w(w_nodes, local_set, dim, sess, stepsize=step)
        
        if dim%200 == 199:            
            #test over all test data
            accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
            print(accuracy)
            save.append(accuracy)
    
    #Iterate over each dimension of bias vector
    for dim in range(10):
        para.communication_b(w_nodes, neighbors, dim, sess, para.b, screen = True, goByzantine = True)
        
        para.node_update_b(w_nodes, local_set, dim, sess, stepsize=step)
    
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(f'Accuracy for iteration {iteration} is {accuracy}', flush = True)
    save.append(accuracy)
    with open('./result/ByRDiE_b2_%d.pickle'%monte_trial, 'wb') as handle:
        pickle.dump(save, handle) 
    weights = [node.weights() for node in w_nodes]
sess.close()

end = time.time()
print(f'Time elapsed {end-start}', flush=True)
