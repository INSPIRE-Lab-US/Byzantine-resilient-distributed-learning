# read MNIST dataset
#default path is '.'
#need the dataset to be downloaded
#http://yann.lecun.com/exdb/mnist/ this is the place to download it
#run with code: train_data, train_label, test_data, test_label = mnist_read()
import os
import struct
import numpy as np
import gzip
import shutil



def mnist_read(path = "."):
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f_in:
        with open('train-images-idx3-ubyte', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)    
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f_in:
        with open('train-labels-idx1-ubyte', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out) 
    with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f_in:
        with open('t10k-images-idx3-ubyte', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out) 
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f_in:
        with open('t10k-labels-idx1-ubyte', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out) 

            
    fname_img = os.path.join(path, 'train-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    trainingdata = []
    for i in range(len(lbl)):
        trainingdata.append(get_img(i))
    train_data = []
    train_label = []
    for sample in trainingdata:
        train_label.append(sample[0])
        image = []
        for row in sample[1]:
            image.extend(row)
        train_data.append(image)
    
    
    fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')    

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])    
    
    testdata = []
    for i in range(len(lbl)):
        testdata.append(get_img(i))
    test_data = []
    test_label = []
    for sample in testdata:
        test_label.append(sample[0])
        image = []
        for row in sample[1]:
            image.extend(row)
        test_data.append(image)
        
        
    return train_data, train_label, test_data, test_label

