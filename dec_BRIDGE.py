'''
Decentralized Learning with BRIDGE

Run this script by specifying the number of Byzantine nodes, whether they should actually be faulty, and the screening method to defend against them
Run this 10 times setting monte_trial to 0-9 to run 10 independent trials
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
parser.add_argument("-b","--byzantine", help="Maximum number of Byzantine nodes to defend against; if none then it defaults to 0", type=int)
parser.add_argument("-gb", "--goByzantine", help="Boolean to indicate if the specified number of Byzantine nodes actually send out faulty values"
                    , type=bool)
parser.add_argument("-s","--screening", help="Screening method to use (BRIDGE,Median, Krum, Bulyan), default is distributed gradient descent without screening",
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

min_neighbor = 2*b+3

if args.screening:
    screen_method = args.screening
    dec_method = screen_method
    if screen_method == 'Bulyan':
        min_neighbor = 4*b+1
        
else:
    screen_method = None
    dec_method = 'DGD'

print(f'Starting Monte Carlo trial {monte_trial}')
start = time.time()
#Directory to make results
os.makedirs(f'./result/{dec_method}',exist_ok=True)

#Setting random seed for reproducibility
np.random.seed(30+monte_trial)
random.seed(a=30+monte_trial)

num_nodes = 20
para = DecLearning(dataset = 'MNIST', nodes=num_nodes, byzantine=b, local_samples=100)

#Generate the graph
con_rate = 50
para.gen_graph(min_neigh = min_neighbor,con_rate=con_rate)    
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.M*para.N, one_hot=True)
neighbors = para.get_neighbor()

save = []

#Initialization
tf.reset_default_graph()

#To ensure reproducibility 
tf.set_random_seed(30+monte_trial)

#Initial step size
step_size=1e-1

w_nodes = [linear_classifier(stepsize = step_size) for node in range(para.M)]    
sess = para.initialization()


#Iterations
T = 100

#BRIDGE Algorithm
for iteration in range(T):
    #Communication and screening
    para.communication(w_nodes, neighbors, sess, b = para.b, 
                        goByzantine = goByzantine ,screenMethod=screen_method)
    
        
    #test over all test data
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    print(f'Accuracy for iteration {iteration} is {accuracy}')
    save.append(np.mean(accuracy))
    

    #node update using GD   
    para.node_update(w_nodes, local_set, sess, stepsize=step_size/(iteration+1))

#Save final learned parameters
wb = [node.weights() for node in w_nodes]

sess.close()

end = time.time()


if b!=0 and goByzantine:
    filename = f'./result/{dec_method}/result_{num_nodes}_nodes_{con_rate}%_b{b}_{monte_trial}.pickle'
else:
    filename = f'./result/{dec_method}/result_{num_nodes}_nodes_{con_rate}%_b{b}_faultless_{monte_trial}.pickle'

print(f'Monte Carlo {monte_trial} Done!\n Time elapsed {end-start} seconds\n')

with open(filename, 'wb') as handle:
    pickle.dump(save, handle)
    pickle.dump(wb, handle)
