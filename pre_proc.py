# read and preprocess data
import random
from MNIST_read import mnist_read

class dis_data:
    def __init__(self, data, label, nodes, shuffle=False, index=None):
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
        
    def shuffle(self):
        random.shuffle(self.index)
        new_data = []
        new_label = []
        for ind in self.index:
            new_data.append(self.all_data[ind])
            new_label.append(self.all_label[ind])
        self.all_data = new_data
        self.all_label = new_label
        return new_data, new_label
    
    def distribute(self, nodes):  #Evenly distributed the data into nodes
        remainder = self.size % nodes
        frac = int(self.size/nodes)
        dist_data = []
        dist_label = []
        for n in range(nodes):
            if n == 0:
                dist_data.append(self.all_data[n * frac : (n + 1) * frac + remainder])
                dist_label.append(self.all_label[n * frac : (n + 1) * frac + remainder])
            else:                
                dist_data.append(self.all_data[n * frac : (n + 1) * frac])
                dist_label.append(self.all_label[n * frac : (n + 1) * frac])
        return dist_data, dist_label
    def next_batch(self, node, size):
        l = len(self.dist_data[node])
        sample = []
        label = []
        for _ in range(size):
            index = random.randint(0, l-1)
            sample.append(self.dist_data[node][index])
            label.append(self.dist_label[node][index])
        return sample, label

def data_prep(dataset, nodes, size=0):
    if dataset == 'MNIST':
        train_data, train_label, test_data, test_label = mnist_read()
    else:
        raise NameError("Cannot find %s dataset") % (dataset)
    
    if size:
        train_data = train_data[:size]
        train_label = train_label[:size]
        
    full_data = dis_data(train_data, train_label, nodes, shuffle = True)
    return full_data
