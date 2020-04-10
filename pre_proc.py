# read and preprocess data
import random
from MNIST_read import mnist_read
import pickle

class dis_data:
    '''
    Encapsulation of our training data split across multiple nodes
    '''
    def __init__(self, data, label, nodes, shuffle=False, index=None, one_hot=False):
        self.size = len(data)
        self.nodes = nodes
        self.all_data = data
        self.all_label = label
        if index:
            self.index = index
        else:
            self.index = list(range(self.size))
        if shuffle:
            self.shuffle()
        self.dist_data, self.dist_label = self.distribute(nodes)
        if one_hot:
            new_label = []
            for node in self.dist_label:
                new_label.append(_one_hot(node))
            self.dist_label = new_label
        
    def shuffle(self):
        '''
        Shuffle the training data and labels, updates the member variables and returns the shuffled data and labels vectors
        '''
        random.shuffle(self.index)
        new_data = []
        new_label = []
        for ind in self.index:
            new_data.append(self.all_data[ind])
            new_label.append(self.all_label[ind])
        self.all_data = new_data
        self.all_label = new_label
        return new_data, new_label
    
    def distribute(self, nodes):
        '''
        Evenly distribute the training data across the nodes
        '''
        remainder = self.size % nodes
        frac = int(self.size/nodes)
        dist_data = []
        dist_label = []
        for n in range(nodes):
            if n == 0:
                dist_data.append(self.all_data[0 : frac + remainder])
                dist_label.append(self.all_label[0 : frac + remainder])
            else:                
                dist_data.append(self.all_data[n * frac : (n + 1) * frac])
                dist_label.append(self.all_label[n * frac : (n + 1) * frac])
        return dist_data, dist_label
    def next_batch(self, node, size):
        '''
        Return next batch of distributed training samples with labels
        '''
        l = len(self.dist_data[node])
        sample = []
        label = []
        for _ in range(size):
            index = random.randint(0, l-1)
            sample.append(self.dist_data[node][index])
            label.append(self.dist_label[node][index])
        return sample, label

def data_prep(dataset, nodes, size=0, one_hot=False):
    '''
    Distribute training data across nodes and return test data w/ labels
    '''
    if dataset == 'MNIST':
        train_data, train_label, test_data, test_label = mnist_read()
        if one_hot:
            test_label = _one_hot(test_label)
    elif dataset == 'CIFAR':
        with open('cifar_dataset.pickle', 'rb') as handle:
            (train_data, train_label, test_data, test_label) = pickle.load(handle)
        train_data, test_data = train_data / 255.0, test_data / 255.0
        train_label = _one_hot(train_label)
        test_label = _one_hot(test_label)
    else:
        raise NameError("Cannot find %s dataset") % (dataset)
    
    if size:
        train_data = train_data[:size]
        train_label = train_label[:size]
        
    full_data = dis_data(train_data, train_label, nodes, shuffle = True, one_hot=one_hot)
    return full_data, test_data, test_label

def _one_hot(label):
    l_oh = []
    for i in label:
        new_l = [0] * 10
        new_l[int(i)] = 1
        l_oh.append(new_l)
    return l_oh