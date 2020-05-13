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
parser.add_argument("std_devInit", help="Specify stnd dev of W matrix", type=float)
parser.add_argument("-b","--byzantine", help="Number of Byzantine nodes to defend against, if none defaults to 0", type=int)
parser.add_argument("-gb", "--goByzantine", help="Boolean to indicate if the specified number of Byzantine nodes actually send out faulty values"
                    , type=bool)
parser.add_argument("-s","--screening", help="Screening method to use (BRIDGE,Median, Krum, Bulyan), default no screening is done regular gradient descent",
                     choices=['BRIDGE','Median','Krum', 'Bulyan'], type=str)

args = parser.parse_args()

std_devW = args.std_devInit

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
else:
    screen_method = None
    dec_method = 'DGD'

print(f'Starting with W matrix std_dev {std_devW}', flush=True)
start = time.time()

#Setting random seed for reproducibility
np.random.seed(30)
random.seed(a=30)

para = DecLearning(dataset = 'MNIST', nodes=20, iterations=100, byzantine=b, local_samples=2000, 
                              con_rate=50, stepsize = 1e-1)
#Generate the graph
para.gen_graph(min_neigh = 4*para.b+1)    
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
neighbors = para.get_neighbor()

save = []

#Initialization
tf.reset_default_graph()

#To ensure reproducibility 
tf.set_random_seed(30)

w_nodes = [linear_classifier(stepsize = para.stepsize, sigma2=std_devW) for node in range(para.M)]    
sess = para.initialization()

#BRIDGE Algorithm
for iteration in range(para.T):
    #Communication 
    para.communication(w_nodes, neighbors, sess, b = para.b, 
                        goByzantine = goByzantine ,screenMethod=screen_method)
    
        
    #test over all test data
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(f'Accuracy for iteration {iteration} is {accuracy}', flush=True)
    save.append(np.mean(accuracy))
    
    #node update using GD   
    para.node_update(w_nodes, local_set, sess, stepsize=para.stepsize/(iteration+1))

#Save final learned parameters
wb = [node.weights() for node in w_nodes]

sess.close()

end = time.time()
if b!=0 and goByzantine:
    filename = f'./result/{dec_method}/result_{dec_method}_b{b}_{std_devW}.pickle'
else:
    filename = f'./result/{dec_method}/result_{dec_method}_b{b}_faultless_{std_devW}.pickle'

print(f'W matrix with {std_devW} Done!\n Time elapsed {end-start} seconds\n', flush=True)

with open(filename, 'wb') as handle:
    pickle.dump(save, handle)
    pickle.dump(wb, handle)
