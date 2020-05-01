import random
import numpy as np
import tensorflow as tf
from screening_method import Byzantine_algs


class DecLearning:
    def __init__(self,dataset, nodes, iterations, byzantine, local_samples, con_rate):
        self.dataset = dataset
        self.M = nodes
        self.T = iterations
        self.b = byzantine
        self.N = local_samples
        self.con_rate = con_rate
        self.graph  = []
        self.edge_weight = []
    
    def gen_graph(self, min_neigh = 0):
        '''
        Generates the adjacency matrix for a graph

        Args:
            nodes: Number of vertices in the graph
            con_rate: Connection rate of graph
            b: Number of Byzantine nodes 
        
        Returns:
            w : Relative weight of each edge
            graph: The adjacency matrix for the graph
        '''
        re = 1                      # regenerate if graph assumption not satisfied
        while re:
            nodes = self.M
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
            if all([sum(row) >= min_neigh for row in graph]):
                re = 0
        self.graph = graph
        self.edge_weight = w
    
    def get_neighbor(self):
        '''
        Returns a matrix where each row is a node and within each row the columns
        contain the respective node's neighbors

        Args:
            G: An adjacency matrix of a graph
        
        Returns:
            neighbor_list: A matrix where the ith row contains the neighboring nodes of the ith node
        '''
        neighbor_list = []
        G = self.graph
        for node in G:
            neighbor = []
            for ind, n in enumerate(node):
                if n == 1:
                    neighbor.append(ind)
            neighbor_list.append(neighbor)
        return neighbor_list

    def one_hot(self,label):
        '''
        Return ones hot encoding of label
        '''
        l_oh = []
        for i in label:
            new_l = [0] * 10
            new_l[i] = 1
            l_oh.append(new_l)
        return l_oh

    def Byzantine(self, target, strategy='random'):
        '''
        Byzantine update method

        Arg:
            target: The parameter of the node undergoing Byzantine failure
            strategy: Method of failure
        
        Returns:
            fal: A numpy array with same dimensions as the target
        '''
        if strategy == 'random':
            #Creates a random array with values from a random uniform distribution [5,15)
            fal = np.random.random(target.shape)*20 + 5
        return fal

    def initialization(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        return sess
    

    def screen(self, grad, b, method):
        screened = Byzantine_algs[method](grad, b)
        return screened
    
    def acc_test(self, model, t_data, t_label):
        acc = model.accuracy.eval(feed_dict={
                model.x:t_data, model.y_: t_label,})
        return acc

    #Used for DGD and BRIDGE
    def communication(self, W, neighbor, sess, b=0, screen=False):
        '''
        Communicate the model (W,b) to all neighbors for each node

        Args:
            W: Nodes in our network
            neighbor: A matrix where the ith row contains an array of the neighbors for the ith node

        '''
        wb = [node.weights() for node in W]
        ave_w = []
        ave_b = []
        for neighbor_list in neighbor:
            neighborhood_w = [wb[n][0] for n in neighbor_list]
            neighborhood_b = [wb[n][1] for n in neighbor_list] 
            if screen:
                neighborhood_w = np.sort(neighborhood_w, axis = 0) 
                neighborhood_w = neighborhood_w[b : -b]
                neighborhood_b = np.sort(neighborhood_b, axis = 0)
                neighborhood_b = neighborhood_b[b : -b]
            neighborhood_w = np.mean(neighborhood_w, axis = 0)
            neighborhood_b = np.mean(neighborhood_b, axis = 0)
            ave_w.append(neighborhood_w)
            ave_b.append(neighborhood_b)

        for node, w, b in zip(W, ave_w, ave_b):
            node.assign([w, b], sess) 
    
    def communication_w(self, W, neighbor, p, sess, b=0, screen=False):
        '''
        Communicate the dimension p of model W to all neighbors for each node, updates
        b nodes using Byzantine update

        Args:
            W: Nodes in our network
            neighbor: A matrix where the ith row contains an array of the neighbors for the ith node
            p: Dimension of W model
            sess: TensorFlow session
            b: Number of Byzantine nodes
            screen: Boolean to screen for Byzantines

        '''

        #Weight matrix flattened to a vector from each node
        _w = [node.weights()[0].reshape([7840,1]) for node in W]

        #Bias vector from each node
        _b = [node.weights()[1] for node in W]
        
        #Byzantine failure
        for node in range(b):
            #Set to a random number [-1,0)
            _w[node][p] = Byzantine(_w[node][p])

        ave = []

        #Iterate over each node in 2D array of neighbors
        for neighbor_list in neighbor:
            #neighbor_list is an array that contains all neighbors of a node in the network
            
            #Collect weight value at dimension p for each node in the neighborhood of the current node in iteration
            #Step 6 of the algo in ByRDiE paper
            neighbors_wp = [_w[n][p] for n in neighbor_list]

            if screen:
                #Step 7 of ByRDiE algo
                neighbors_wp = np.sort(neighbors_wp, axis = 0)
                neighbors_wp = neighbors_wp[b : -b]
            
            #Half of step 8 (average of w across all nodes in neighborhood)
            neighbors_wp = np.mean(neighbors_wp, axis = 0)
            ave.append(neighbors_wp)
        
        #Half of step 8
        for scalar, node in zip(ave, _w):
            node[p] = scalar

        for node, ww, bb in zip(W, _w, _b):
            node.assign([ww.reshape([784, 10]), bb], sess) 


    def node_update(self, W, data, sess, stepsize=1e-3):        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            sess.run(model.train_step, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})

'''  
Member Variables:
graph Adjacency matrix, for generated graph
dataset
M (# of agents)
T (iterations)
screen
b (# of Byzantine nodes)
N (# of local data samples for each node)
con_rate (connection rate of the graph)


Methods 'private':
gen_graph(neigh_cond)- Generates the graph associated with this decentralized learning scheme and sets the adjacency matrix for this instance of decentralized learning, also accepts the restriction on # of neighbors for each node
get_neighbors() - Returns a 2D array with the neighbors for each node in the graph
one_hot() - Performs one hot encoding of the labels
Byzantine() - Sends a byzantine update

(For BRIDGE)
communication() - A method that communicates to ALL the neighbors of each node this is as in DGD and BRIDGE

(For ByRDiE)
communication_w(p) - A method that communicates a particular dimension of W to all the nodes
communication_b(p) - A method that communicates a particular dimension of b to all the nodes

Methods 'public':
ByRDiE(step_size) - Learns the model using the simulated ByRDiE algorithm
BRIDGE(screen_method, step_size) - Learns the model using the simulated BRIDGE algo
DGD(step_size) - Learns the model using regular DGD
'''