
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical # for encoding data to one not form

# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import os


class Data():
    import numpy as np
    def __init__(self,stop_epoch=100,dataset='fashion-mnist'):
        self.samples = []
        self.labels = []
        self.test_samples = []
        self.test_labels = []
        self.pointer = 0
        self.data_index = []
        self.epoch = 0
        if dataset:
            self.initialize_data(dataset)
    
    def initialize_data(self,dataset):
        if dataset == 'fashion-mnist':
            print('now initizalize fashion mnist data')
            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            self.labels = self.oneHotEncode(y_train)
            self.test_labels = self.oneHotEncode(y_test)

            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.

            self.samples = x_train.reshape((len(x_train),784))
            self.test_samples = x_test.reshape(len(x_test),784)
            self.data_index = np.arange(len(self.samples))
            
        if dataset == 'cifar10':
            print('now initialize cifar10 data')
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            self.labels = self.oneHotEncode(y_train)
            self.test_labels = self.oneHotEncode(y_test)

            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.

            self.samples = x_train.reshape((len(x_train),32*32*3))
            self.test_samples = x_test.reshape(len(x_test),32*32*3)
            self.data_index = np.arange(len(self.samples))

            
    def oneHotEncode(self, data):
        print('Shape of data (BEFORE encode): %s' % str(data.shape))
        encoded = to_categorical(data)
        print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
        return encoded

    def next_batch(self,num):
        if self.pointer + num <= len(self.samples):
            index = self.data_index[self.pointer:self.pointer+num]
            batch_samples = self.samples[index]
            batch_labels = self.labels[index]
            self.pointer += num
            return batch_samples, batch_labels
        else:
            new_pointer = self.pointer + num - len(self.samples)
            index = np.concatenate((self.data_index[self.pointer:], self.data_index[:new_pointer]),axis=0)
            self.shuffle_data()
            batch_samples = self.samples[index]
            batch_labels = self.labels[index]
            self.pointer = new_pointer
            return batch_samples, batch_labels
     
    def shuffle_data(self):
        np.random.shuffle(self.data_index)
        self.epoch += 1
        #print('epoch: ', self.epoch)
        
    def reset(self):
        self.pointer = 0
        self.data_index = np.arange(len(self.samples))
        self.epoch = 0