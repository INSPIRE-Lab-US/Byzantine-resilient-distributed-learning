'''
Decentralized Learning with ByRDiE

Run this script by specifying the number of Byzantine nodes and whether they should actually be faulty
Run this 10 times setting monte_trial argument from 0-9 to run ten independent trials
'''

import numpy as np
from dist_data import data_prep
from linear_classifier import linear_classifier
import tensorflow as tf
import time
import pickle
import random
import argparse
import os
from DecLearning import DecLearning

parser = argparse.ArgumentParser()
parser.add_argument("monte_trial", help="A number between 0 and 9 to indicate which Monte Carlo trial to run", type=int)
parser.add_argument("-b","--byzantine", help="Maximum number of Byzantine nodes to defend against, if none defaults to 0", type=int)
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

#Directory to store result
os.makedirs('./result/ByRDiE', exist_ok=True)

#Reproducibility
np.random.seed(30+monte_trial)
random.seed(a=30+monte_trial)


num_nodes = 20
para = DecLearning(dataset = 'MNIST', nodes=num_nodes, byzantine = b, local_samples=2000)
loaded = False
#Generate the graph
para.gen_graph(min_neigh = min_neighbor)
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.M*para.N, one_hot=True)
neighbors = para.get_neighbor()
save = []

#Initialization
tf.reset_default_graph()

#To ensure reproducibility 
tf.set_random_seed(30+monte_trial)

step_size=1e-1

w_nodes = [linear_classifier(stepsize = step_size) for node in range(para.M)]    
sess = para.initialization()

T = 100


#ByRDiE Algorithm
for iteration in range(T):


    #Iterates over the first 784 dimensions of the weights
    for p in range(7840):
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


sess.close()


if b!=0 and goByzantine:
    filename = f'./result/ByRDiE/result_{num_nodes}_nodes_{con_rate}%_b{b}_{monte_trial}.pickle'
else:
    filename = f'./result/ByRDiE/result_{num_nodes}_nodes_{con_rate}%_b{b}_faultless_{monte_trial}.pickle'        

end = time.time()
print(f'Monte Carlo {monte_trial} Done!\n Time elapsed {end-start} seconds\n')

with open(filename, 'wb') as handle:
    pickle.dump(save, handle) 
