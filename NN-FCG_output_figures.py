# coding=utf-8
import sys
import numpy as np
import math
import random
from support import *
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from confusion_matrix import ConfusionMatrix

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
random.seed(1024 * 1024)

num_feature = 28 * 28
num_label = 10

num_input = num_feature
num_hidden = 25
num_output = num_label

train_path = 'mnist.train'
test_path = 'mnist.test'

def read_dense_data(fp_data, num_feature):
    x = list()
    y = list()
    for index, line in enumerate(fp_data):
        line_arr = line.strip().split()
        x.append([0.0] * num_feature)
        y.append(int(line_arr[0]))
        for kv in line_arr[1: ]:
            k, v = kv.split(':')
            k = int(k) - 1
            v = float(v)
            x[index][k] = v
    x = np.matrix(x)
    y = np.matrix(y).T
    return x, y

class NeuralNet:
    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        
    def train(self, x, y, num_epoch=30, mini_batch=100, lambda_=0.01):

        
        num_params = self.num_hidden * (self.num_input + 1) + self.num_output * (self.num_hidden + 1)
        w = np.matrix(0.005 * np.random.random([num_params, 1]))
        data = np.column_stack([x, y])    

        eta_an = 0.05
        eta_inc = 0.8
        eta_dec = 1.01
        beta = 0.4 #"Armijo rule is applicable for beta less than 0.5"
 

        
        first_run = True
        # nestrov
        y_current  = w
        t_current = 1.0

        for epoch in range(num_epoch):
            np.random.shuffle(data)
            k = 0
            cost_array = list()
            
            while k < len(data):
                x = data[k: k + mini_batch, 0: -1]
                y = np.matrix(data[k: k + mini_batch, -1], dtype='int32')
                #Fast control gradient 
                # Armigo Stochastic gradient descent
                t_next = .5*(1 + np.sqrt(1 + 4*t_current**2))
                cost, grad = self.gradient(x, y, lambda_, y_current)
                w_next = y_current - eta_an * grad
                y_next = w_next + (t_current - 1.0)/(t_next)*(w_next - w)
                    
                cost, grad = self.gradient(x, y, lambda_, w_next)
                cost_next = cost#?!!! need cost of w
                grad_next = grad
                # restarting strategies nestrov
                if np.dot((y_current - w_next).T, w_next - w) > 0:
                    y_next = w_next
                    t_next = 1
                      
                w = w_next
                y_current = y_next
                t_current = t_next
                    
                k += mini_batch
                cost_array.append(cost)
                if first_run == True: first_run = False

            # Armijo condition
            if epoch > 0 and cost_next >= cost_old + beta * grad_next.T.dot(w_next - w_old):
                    eta_an *= eta_inc
            else:
                    eta_an *= eta_dec
                    
            w_old = w_next
            cost_old = cost_next  
                

        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1): ].reshape(self.num_output, self.num_hidden + 1)
        
        #confusion_mnist = self.predictm(x, y)
        #self.print_and_plot('Logistic Regression: CGD',  confusion_mnist)
        return w
        
    def gradient(self, x, y, lambda_, w):
        # x = data[:, 0: -1]
        # y = np.matrix(data[:, -1], dtype='int32')
        num_sample = len(x)

        w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        w2 = w[self.num_hidden * (self.num_input + 1): ].reshape(self.num_output, self.num_hidden + 1)
        b = np.matrix(np.ones([num_sample, 1]))
    
        a1 = np.column_stack([x, b])
        s2 = sigmoid(a1 * w1.T)
        a2 = np.column_stack([s2, b])
        a3 = sigmoid(a2 * w2.T)

        y_one_hot = np.matrix(np.zeros([num_sample, self.num_output]))
        y_one_hot[(np.matrix(range(num_sample)), y.T)] = 1
        
        cost = (1.0 / num_sample) * (- np.multiply(y_one_hot, np.log(a3)) - np.multiply(1.0 - y_one_hot, np.log(1.0 - a3))).sum()
        cost += (lambda_ / (2.0 * num_sample)) * (np.square(w1[:, 0: -1]).sum() + np.square(w2[:, 0: -1]).sum())

        delta3 = a3 - y_one_hot
        delta2 = np.multiply(delta3 * w2[:, 0: -1], np.multiply(s2, 1.0 - s2))
        l1_grad = delta2.T * a1
        l2_grad = delta3.T * a2
        
        r1_grad = np.column_stack([w1[:, 0: -1], np.matrix(np.zeros([self.num_hidden, 1]))])
        r2_grad = np.column_stack([w2[:, 0: -1], np.matrix(np.zeros([self.num_output, 1]))])

        w1_grad = (1.0 / num_sample) * l1_grad + (1.0 * lambda_ / num_sample) * r1_grad
        w2_grad = (1.0 / num_sample) * l2_grad + (1.0 * lambda_ / num_sample) * r2_grad
        w_grad = np.row_stack([w1_grad.reshape(-1, 1), w2_grad.reshape(-1, 1)])

        return cost, w_grad
     
    def predict(self, x):
        num_sample = len(x)
        b = np.matrix(np.ones([num_sample, 1]))
        h1 = sigmoid(np.column_stack([x, b]) * self.w1.T)
        h2 = sigmoid(np.column_stack([h1, b]) * self.w2.T)
        return np.argmax(h2, 1) 
        
    def test(self, x, y):
        num_sample = len(x)
        y_pred = self.predict(x)
        y_one_hot = np.matrix(np.zeros(y.shape))
        y_one_hot[np.where(y_pred == y)] = 1
        acc = 1.0 * y_one_hot.sum() / num_sample
        return acc
    
    def predictm(self, x, y): 

        self.yhat = self.predict(x)

        #self.yhat = np.argmax(self.yhat, axis=1)
        confusion = self.find_confusion_matrix(y, self.yhat)
        return confusion
     
    # Calculate Softmax
    @staticmethod
    def softmax(z):
        """
        :param z:
        :return: softmax
        """
        return (np.exp(z) / np.sum(np.exp(z), axis=0)).T
    
    @staticmethod
    def find_confusion_matrix(y, yhat):
        """
        :param y: actual label
        :param yhat: predicted label
        :return: confusion matrix
        """
        return confusion_matrix.calculate(y, yhat)
    
def print_and_plot(model, confusion_mnist):
    print('\n\n----------' + model + '----------')
    print('On MNIST data')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot(model + ' on MNIST dataset', 
                          confusion_mnist, confusion_mnist.shape[0])
    
if __name__ == '__main__': 
    
    x_train, y_train = read_dense_data(open(train_path), num_feature)
    train = x_train, y_train
    #train = np.concatenate(x_train, y_train, axis=1)
    showImages(train, np.random.randint(len(train[0]), size=9))    
    # z_score_normalize
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_train = 1.0 * (x_train - mean) / (std + 0.0001)
    x_test, y_test = read_dense_data(open(test_path), num_feature)
    x_test = 1.0 * (x_test - mean) / (std + 0.0001)
    
    confusion_matrix = ConfusionMatrix()
    clf = NeuralNet(num_input ,num_hidden, num_output)
    clf.train(x_train, y_train)
    confusion_mnist = clf.predictm( x_test, y_test) 
    print_and_plot('Neural network: FGD',  confusion_mnist)
    



