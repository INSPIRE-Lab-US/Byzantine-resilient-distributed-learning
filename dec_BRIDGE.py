'''
Decentralized Learning with BRIDGE

Run this script by specifying the number of Byzantine nodes, whether they should actually be faulty, and the screening method to defend against them
Run this 10 times setting monte_trial to 0-9 to reproduce the results
'''

import numpy as np
from pre_proc import data_prep
from linear_classifier import linear_classifier
import tensorflow as tf
import time
import pickle
import random
import argparse
from DecLearning import DecLearning
from screening_method import Byzantine_algs
      
parser = argparse.ArgumentParser()
parser.add_argument("monte_trial", help="Specify which monte carlo trial to run", type=int)
parser.add_argument("-b","--byzantine", help="Number of Byzantine nodes to defend against, if none defaults to 0", type=int)
parser.add_argument("-gb", "--goByzantine", help="Boolean to indicate if the specified number of Byzantine nodes actually send out faulty values"
                    , type=bool)
parser.add_argument("-s","--screening", help="Screening method to use (BRIDGE,Median, Krum, Bulyan), default no screening is done regular gradient descent",
                     choices=['BRIDGE','Median','Krum', 'Bulyan'], type=str)

args = parser.parse_args()

monte_trial = args.monte_trial
if args.byzantine:
    b = args.byzantine
else:
    b = 0

if args.goByzantine:
    goByzantine=args.goByzantine
else:
    goByzantine = False

if args.screening:
    screen_method = args.screening
    dec_method = screen_method
    if screen_method == 'Bulyan':
        min_neighbor = 4*b+3
    else:
        min_neighbor = 2*b+3
else:
    screen_method = None
    dec_method = 'DGD'

print(f'Starting Monte Carlo trial {monte_trial}')
start = time.time()

#Setting random seed for reproducibility
np.random.seed(30+monte_trial)
random.seed(a=30+monte_trial)

para = DecLearning(dataset = 'MNIST', nodes=20, iterations=100, byzantine=b, local_samples=2000, 
                              con_rate=50, stepsize = 1e-1)
#Generate the graph
para.gen_graph(min_neigh = min_neighbor)    
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
    para.communication(w_nodes, neighbors, sess, b = para.b, 
                        goByzantine = goByzantine ,screenMethod=screen_method)
    
        
    #test over all test data
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(f'Accuracy for iteration {iteration} is {accuracy}')
    save.append(np.mean(accuracy))
    
    #node update using GD   
    para.node_update(w_nodes, local_set, sess, stepsize=para.stepsize/(iteration+1))

#Save final learned parameters
wb = [node.weights() for node in w_nodes]

sess.close()

end = time.time()
if b!=0 and goByzantine:
    filename = f'./result/{dec_method}/result_{dec_method}_b{b}_{monte_trial}.pickle'
else:
    filename = f'./result/{dec_method}/result_{dec_method}_b{b}_faultless_{monte_trial}.pickle'

print(f'Monte Carlo {monte_trial} Done!\n Time elapsed {end-start} seconds\n')

with open(filename, 'wb') as handle:
    pickle.dump(save, handle)
    pickle.dump(wb, handle)
