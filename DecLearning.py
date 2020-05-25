import random
import numpy as np
import tensorflow as tf
from screening_method import Byzantine_algs
from linear_classifier import linear_classifier


class DecLearning:
    def __init__(self,dataset = 'MNIST', nodes = 20, byzantine = 0, 
                    local_samples = 2000):
        self.dataset = dataset
        self.M = nodes
        self.b = byzantine
        self.N = local_samples
        self.graph  = []
        self.edge_weight = []
    
    def gen_graph(self, min_neigh = 0, con_rate=50):
        '''
        Generates the adjacency matrix for a graph

        Args:
            nodes: Number of vertices in the graph
            con_rate: Connection rate of graph (default: 50)
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

    def Byzantine(self, target, strategy='random', interval = (-1,0)):
        '''
        Byzantine update method

        Arg:
            target: The parameter of the node undergoing Byzantine failure
            strategy: Method of failure
        
        Returns:
            fal: A numpy array with same dimensions as the target
        '''
        if strategy == 'random':
            dist = interval[1] - interval[0]
            min = interval[0]
            #Creates a random array with values from a random uniform distribution
            fal = np.random.random(target.shape) * dist + min
        return fal

    def initialization(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        return sess
    

    def screen(self, grad, b, method):
        screened = Byzantine_algs[method](grad, b)
        return screened
    
    #The new grad W and grad b vectors for each node are set to the median of all their neighbors
    def Median(self, W, neighbor, wb, b):
        ave_w = []
        ave_b = []
        for neighbor_list in neighbor:
            neighborhood_w = [wb[n][0] for n in neighbor_list]
            neighborhood_b = [wb[n][1] for n in neighbor_list] 

            neighborhood_w = np.median(neighborhood_w, axis = 0)
            neighborhood_b = np.median(neighborhood_b, axis = 0)
            
            ave_w.append(neighborhood_w)
            ave_b.append(neighborhood_b)
               
        return ave_w, ave_b
    
    def Krum(self,neighbor, wb, b):
        '''
        Perform Krum screening

        Args:
            neighbor: 2D list of neighbors for each node
            wb: List with W matrix and b vector for each node
            b: Number of byzantine nodes

        Returns:
            new_w: List of W matrix for each node based on Krum screening
            new_b: List of b vector for each node based on Krum screening
        '''
        new_w = []
        new_b = []

        #Krum screening for each node
        for i,neighbor_list in enumerate(neighbor):
            score_w = []
            score_b = []

            neighborhood_w = [wb[n][0] for n in neighbor_list]
            neighborhood_b = [wb[n][1] for n in neighbor_list]

            #Iterate through all neighbors of the current node
            for g_w, g_b in zip(neighborhood_w, neighborhood_b):
                dist_w = [np.linalg.norm(other-g_w) for other in neighborhood_w]
                dist_b = [np.linalg.norm(other-g_b) for other in neighborhood_b]

                dist_w = np.sort(dist_w)
                dist_b = np.sort(dist_b)

                #Sum up closest n-b-2 vectors to g_w and g_b 
                score_w.append(np.sum(dist_w[:(len(neighborhood_w) - b - 2)]))
                score_b.append(np.sum(dist_b[:(len(neighborhood_b) - b - 2)]))
            
            print(f'Score W for node {i} is {score_w}')
            print(f'Score b for node {i} is {score_b}')
            
            ind_w = score_w.index(min(score_w))
            ind_b = score_b.index(min(score_b))

            new_w.append(neighborhood_w[ind_w])
            new_b.append(neighborhood_b[ind_b])
        
        return new_w, new_b

    def acc_test(self, model, t_data, t_label):
        acc = model.accuracy.eval(feed_dict={
                model.x:t_data, model.y_: t_label,})
        return acc

    #Used for DGD and BRIDGE
    def communication(self, W, neighbor, sess, b=0, 
                    goByzantine = False, screenMethod = None):
        '''
        Communicate the model (W,b) to all neighbors for each node

        Args:
            W: Nodes in our network
            neighbor: A matrix where the ith row contains an array of the neighbors for the ith node

        '''
        wb = [node.weights() for node in W]
        ave_w = []
        ave_b = []

        if goByzantine:
            # if screenMethod==None:
            #     byz_range = (5,25)
            # elif screenMethod=='Krum':
            #     byz_range = (-0.01, 0)
            # else:
            byz_range=(-1,0)
            #Byzantine failed nodes assigned first
            for byzant in range(b):
                wb[byzant][0] = self.Byzantine(wb[byzant][0], interval=byz_range)
                wb[byzant][1] = self.Byzantine(wb[byzant][1], interval=byz_range)

        if screenMethod == 'Median':
            ave_w, ave_b = self.Median(W, neighbor, wb, b)
        if screenMethod == 'Krum':
            ave_w, ave_b = self.Krum(neighbor, wb, b)
        else:
            for neighbor_list in neighbor:
                neighborhood_w = [wb[n][0] for n in neighbor_list]
                neighborhood_b = [wb[n][1] for n in neighbor_list] 
                
                #Perform vanilla BRIDGE screening
                if screenMethod == 'BRIDGE':
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
    
    #Used for ByRDiE
    def communication_w(self, W, neighbor, p, sess, b=0, screen=False, goByzantine = False):
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
        if goByzantine:
            for node in range(b):
                _w[node][p] = self.Byzantine(_w[node][p])

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


    def communication_b(self, W, neighbor, p, sess, b=0, screen=False, goByzantine=False):
        '''
        Communicate the model b to all neighbors for each node

        Args:
            W: Nodes in our network
            neighbor: A matrix where the ith row contains an array of the neighbors for the ith node

        '''
        _w = [node.weights()[0] for node in W]
        _b = [node.weights()[1] for node in W]
        
        #Byzantine failure
        if goByzantine:
            for node in range(b):
                _b[node][p] = self.Byzantine(_b[node][p])

        ave = []
        for neighbor_list in neighbor:
            neighborhood = [_b[n][p] for n in neighbor_list]

            if screen:
                neighborhood = np.sort(neighborhood, axis = 0)
                neighborhood = neighborhood[b : -b]
            neighborhood = np.mean(neighborhood, axis = 0)
        #        print(neighborhood.shape)
            ave.append(neighborhood)
        for scalar, node in zip(ave, _b):
            node[p] = scalar

        for node, ww, bb in zip(W, _w, _b):
            node.assign([ww, bb], sess) 

    def node_update(self, W, data, sess, stepsize=1e-3):        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            sess.run(model.train_step, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
    
    def node_update_w(self, W, data, p, sess, stepsize=1e-4):        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            g = sess.run(model.gradient_w, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
            new_g = np.zeros([7840, 1])
            new_g[p] = g.reshape([7840, 1])[p]
            sess.run(model.local_update_w, feed_dict={model.gradient_port_w: new_g.reshape([784,10]), model.stepsize: stepsize})
        
    def node_update_b(self, W, data, p, sess, stepsize=1e-4):        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            g = sess.run(model.gradient_b, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
            new_g = np.zeros([10])
            new_g[p] = g[p]
            sess.run(model.local_update_b, feed_dict={model.gradient_port_b: new_g, model.stepsize: stepsize})
