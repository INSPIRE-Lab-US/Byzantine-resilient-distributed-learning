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

monte_trial = sys.argv[1]
print(f'Starting Monte Carlo trial {monte_trial}')

#Setting random seed for reproducibility
random.seed(a=30+monte_trial)
np.random.seed(30+monte_trial)


para = DecLearning(dataset = 'MNIST', nodes=20, iterations=100, byzantine=2, local_samples=2000, 
                              con_rate=50, stepsize = 1e-1)
#Generate the graph
W_0, graph = para.gen_graph(min_neigh = 4*para.b+1)    
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
neighbors = para.get_neighbor()

save = []

#Initialization
tf.reset_default_graph()

#Ensure reproducibility
tf.set_random_seed(30+monte_trial)

w_nodes = [linear_classifier(stepsize = para.stepsize) for node in range(para.M)]    
sess = para.initialization()

save = []
for iteration in range(para.T):
    step = para.stepsize/(iteration+1)

    for p_w in range(7840):
        #Communication 
        para.communication_w(w_nodes, neighbors, p_w, sess, para.b, screen = True, goByzantine=False)

        #node update using DGD    
        para.node_update_w(w_nodes, local_set, p_w, sess, stepsize=step)


        if p_w%200 == 199:            
            #test over all test data
            accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
            print(accuracy)
            save.append(accuracy)
    
    for p_b in range(10):
        para.communication_b(w_nodes, neighbors, p_b, sess, para.b, screen = True, goByzantine=False)
        para.node_update_b(w_nodes, local_set, p_b, sess, stepsize=step)
    

    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(f'Accuracy for iteration {iteration} is {accuracy}')
    print(f'Total scalar broadcasts = {(iteration+1)*(p_w+p_b)}')
    save.append(accuracy)

    with open('./result/ByRDiE/ByRDiE_b0_%d.pickle'%monte_trial, 'wb') as handle:
        pickle.dump(save, handle) 

sess.close()
end = time.time()
print(f'Time elapsed {end-start}')
