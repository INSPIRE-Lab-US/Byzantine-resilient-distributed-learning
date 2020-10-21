import random
import numpy as np
import tensorflow as tf
from linear_classifier import linear_classifier


class DecLearning:
    '''
    Encapsulates a simulated decentralized network
    '''
    def __init__(self,dataset = 'MNIST', nodes = 20, byzantine = 0, 
                    total_samples = 2000):
        self.dataset = dataset
        self.M = nodes
        self.b = byzantine
        self.N = total_samples
        self.graph  = []
        self.edge_weight = []
    
    def gen_graph(self, min_neigh = 0, con_rate=50):
        '''
        Generates the adjacency matrix for a graph

        Args:
            min_neigh: Minimum number of neighbors for each node (default: 0)
            con_rate: Connection rate of graph (default: 50)
        '''
        re = 1                      # regenerate if graph assumption not satisfied
        while re:
            nodes = self.M

            # Generate adjacency matrix
            graph = np.random.randint(1,high=100,size=(nodes,nodes))
            graph = (graph+graph.T)/2
            graph[graph<con_rate] = 1
            graph[graph>=con_rate] = 0
            np.fill_diagonal(graph,1)
            
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
        Return one hot encoding of label
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
            interval: Range of values that the failed Byzantine nodes take on
        
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
    
    
    def Median(self, neighbor, wb, b):
        '''
        Perform BRIDGE-Median screening

        Args:
            neighbor: 2D list of neighbors for each node
            wb: List of W matrix and bias vector for each node to be learned
            b: Number of byzantine nodes to defend against
        Return:
            ave_w: List of W matrix for each node based on BRIDGE-Median screening
            ave_b: List of b vector for each node based on BRIDGE-Median screening
        '''
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
        Perform BRIDGE-Krum screening

        Args:
            neighbor: Matrix of neighbors for each node
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

    def Bulyan(self, wb, b):
        '''
        Perform decentralized BRIDGE-Bulyan with Krum screening

        Args:
            wb: List with W matrix and b vector for each node
            b: Number of byzantine nodes

        Returns:
            new_w: List of W matrix for each node based on Bulyan screening
            new_b: List of b vector for each node based on Bulyan screening
        '''      
        new_w = []
        new_b = []


        #Bulyan screening for W matrix
        neighbor = self.get_neighbor()
        for i,neighbor_list in enumerate(neighbor):
            S_w = []
            #Part 1 of Bulyan using Krum screening to screen for W matrix
            M = len(neighbor_list)
            for _ in range(M - 2 * b):
                score_w = []

                neighborhood_w = [wb[n][0] for n in neighbor_list]

                for g_w in neighborhood_w:
                    dist_w = [np.abs(other - g_w) for other in neighborhood_w]

                    dist_w = np.sort(dist_w)

                    score_w.append(np.sum(dist_w[:(len(neighborhood_w) - b - 2)]))
                ind_w = score_w.index(min(score_w))

                S_w.append(neighborhood_w.pop(ind_w))


        #Part 2 of Bulyan screening for W matrix
        ave = []

        #Dimension of the gradient we are screening for
        grad_dim = len(S_w[0])

        for dim in range(grad_dim):
            m_i = [w[dim] for w in S_w]
            m_i = np.sort(m_i, axis = 0)
            m_i = m_i[b : -b]
            m_i = np.mean(m_i, axis = 0)
            ave.append(m_i)
        new_w.append(ave)

        #Bulyan screening for b vector
        neighbor = self.get_neighbor()
        for i,neighbor_list in enumerate(neighbor):
            S_b = []
            #Part 1 of Bulyan using Krum screening to screen for b vector
            M = len(neighbor_list)
            for _ in range(M - 2 * b):
                score_b = []

                neighborhood_b = [wb[n][1] for n in neighbor_list]

                for g_b in neighborhood_b:
                    dist_b = [np.abs(other - g_b) for other in neighborhood_b]

                    dist_b = np.sort(dist_b)

                    score_b.append(np.sum(dist_b[:(len(neighborhood_b) - b - 2)]))
                ind_b = score_b.index(min(score_b))

                S_b.append(neighborhood_b.pop(ind_b))

                del neighbor_list[ind_b]

        #Part 2 of Bulyan screening for W matrix
        ave = []

        #Dimension of the gradient we are screening for
        grad_dim = len(S_b[0])

        for i in range(grad_dim):
            m_i = [b[i] for b in S_b]
            m_i = np.sort(m_i, axis = 0)
            m_i = m_i[b : -b]
            m_i = np.mean(m_i, axis = 0)
            ave.append(m_i)
        new_b.append(ave)          


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
            neighbor: Matrix of neighbors for each node
            sess: TensforFlow session
            b: Number of Byzantine nodes to defend against
            goByzantine: Boolean to tell us whether b nodes actually undergo failure (default: False)
            screenMethod: Screening method to be used (default: None)
        '''
        wb = [node.weights() for node in W]
        ave_w = []
        ave_b = []

        if goByzantine:
            byz_range=(-1,0)
            #Byzantine failed nodes assigned first
            for byzant in range(b):
                wb[byzant][0] = self.Byzantine(wb[byzant][0], interval=byz_range)
                wb[byzant][1] = self.Byzantine(wb[byzant][1], interval=byz_range)

        if screenMethod == 'Median':
            ave_w, ave_b = self.Median(W, neighbor, wb, b)
        if screenMethod == 'Krum':
            ave_w, ave_b = self.Krum(neighbor, wb, b)
        if screenMethod == 'Bulyan':
            ave_w, ave_b = self.Bulyan(wb, b)
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
            neighbor: Matrix of neighbors for each node
            p: Dimension of W model
            sess: TensorFlow session
            b: Number of Byzantine nodes
            screen: Boolean to screen for Byzantines using ByRDiE

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

            neighbors_wp = [_w[n][p] for n in neighbor_list]

            if screen:
                neighbors_wp = np.sort(neighbors_wp, axis = 0)
                neighbors_wp = neighbors_wp[b : -b]
            
            neighbors_wp = np.mean(neighbors_wp, axis = 0)
            ave.append(neighbors_wp)
        
        for scalar, node in zip(ave, _w):
            node[p] = scalar

        for node, ww, bb in zip(W, _w, _b):
            node.assign([ww.reshape([784, 10]), bb], sess) 


    def communication_b(self, W, neighbor, p, sess, b=0, screen=False, goByzantine=False):
        '''
        Communicate the model b to all neighbors for each node

        Args:
            W: Nodes in our network
            neighbor: Matrix of neighbors for each node
            p: Dimension of bias vector
            sess: TensorFlow session
            b: Number of Byzantine nodes
            screen: Boolean to screen for Byzantines using ByRDiE

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
            ave.append(neighborhood)
        for scalar, node in zip(ave, _b):
            node[p] = scalar

        for node, ww, bb in zip(W, _w, _b):
            node.assign([ww, bb], sess) 

    def node_update(self, W, data, sess, stepsize=1e-3):
        '''
        Update model parameters for each node

        Args:
            W: Nodes in the network
            data: Training data
            sess: TensorFlow seassion
            stepsize: Stepsize for the optimization algorithm (default: 1e-3)
        '''        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            sess.run(model.train_step, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
    
    def node_update_w(self, W, data, p, sess, stepsize=1e-4):
        '''
        Update W matrix for each node used for coordinate descent with ByRDiE

        Args:
            W: Nodes in the network
            data: Training data
            p: Dimension being updated
            sess: TensorFlow seassion
            stepsize: Stepsize for the optimization algorithm (default: 1e-4)            
        '''        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            g = sess.run(model.gradient_w, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
            new_g = np.zeros([7840, 1])
            new_g[p] = g.reshape([7840, 1])[p]
            sess.run(model.local_update_w, feed_dict={model.gradient_port_w: new_g.reshape([784,10]), model.stepsize: stepsize})
        
    def node_update_b(self, W, data, p, sess, stepsize=1e-4):
        '''
        Update bias vector for each node used for coordinate descent with ByRDiE

        Args:
            W: Nodes in the network
            data: Training data
            p: Dimension being updated
            sess: TensorFlow seassion
            stepsize: Stepsize for the optimization algorithm (default: 1e-4) 
        '''        
        for model, sample, label in zip(W, data.dist_data, data.dist_label):
            g = sess.run(model.gradient_b, feed_dict={model.x: sample, model.y_: label, model.stepsize: stepsize})
            new_g = np.zeros([10])
            new_g[p] = g[p]
            sess.run(model.local_update_b, feed_dict={model.gradient_port_b: new_g, model.stepsize: stepsize})
