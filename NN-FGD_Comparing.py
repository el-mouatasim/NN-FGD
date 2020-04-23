# coding=utf-8
import sys
import numpy as np
import math
import random
from datetime import datetime

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
plt_color_array = ['blue',  'green', 'red', 'cyan', 'magenta' ]
plt_dict = dict()
plt_dict_time = dict()

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
random.seed(1024 * 1024)

num_feature = 28 * 28
num_label = 10

num_input = num_feature
num_hidden = 25
num_output = num_label

train_path = 'data/mnist.train'
test_path = 'data/mnist.test'
opt_algo_set = ['FGD','SGD','NAG', 'Momentum', 'Adadelta']

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
        
    def train(self, x, y, opt_algo, num_epoch=30, mini_batch=500, lambda_=0.02):
        if not opt_algo in opt_algo_set:
            print(   'opt_algo not in %s' % opt_algo_set)
            return
        print(   'optimization with [%s]' % opt_algo)
        
        num_params = self.num_hidden * (self.num_input + 1) + self.num_output * (self.num_hidden + 1)
        w = np.matrix(0.005 * np.random.random([num_params, 1]))
        data = np.column_stack([x, y])    

        if opt_algo == 'FGD':
            eta_an = 0.05
            eta_inc = 0.8
            eta_dec = 1.01
            beta = 0.4 #"Armijo rule is applicable for beta less than 0.5"

        gamma = 0.9
        epsilon = 1e-8

        
        eta = 0.05

        v = np.matrix(np.zeros(w.shape))
        m = np.matrix(np.zeros(w.shape))
       

        
        # Adadelta & RMSprop params
        grad_expect = np.matrix(np.zeros(w.shape))
        delta_expect = np.matrix(np.zeros(w.shape))
        

        
        first_run = True
        # nestrov
        y_current  = w
        t_current = 1.0
        loss_array = list()
        
        #timings_collector.append(0)
        timings_collector=[]
        start = datetime.now()
        timing=0
        
        for epoch in range(num_epoch):
            timing=(datetime.now() - start).total_seconds()
            timings_collector.append(timing)
            
            #
            np.random.shuffle(data)
            k = 0
            cost_array = list()
            
            while k < len(data):
                
                #
                x = data[k: k + mini_batch, 0: -1]
                y = np.matrix(data[k: k + mini_batch, -1], dtype='int32')
                if opt_algo == 'SGD':
                    # Stochastic gradient descent
                    cost, grad = self.gradient(x, y, lambda_, w)
                    w = w - eta * grad

                elif opt_algo == 'Momentum':
                    # Momentum
                    cost, grad = self.gradient(x, y, lambda_, w)
                    v = gamma * v + eta * grad
                    w = w - v

                elif opt_algo == 'NAG':
                    # Nesterov accelerated gradient
                    cost, grad = self.gradient(x, y, lambda_, w - gamma * v)
                    v = gamma * v + eta * grad
                    w = w - v


                elif opt_algo == 'Adadelta':
                    # Adadelta
                    cost, grad = self.gradient(x, y, lambda_, w)
                    grad_expect = gamma * grad_expect + (1.0 - gamma) * np.square(grad)
                    # when first run, use sgd
                    if first_run == True:
                        delta = - eta * grad 
                    else:    
                        delta = - np.multiply(np.sqrt(delta_expect + epsilon) / np.sqrt(grad_expect + epsilon),  grad)
                    w = w + delta
                    delta_expect = gamma * delta_expect + (1.0 - gamma) * np.square(delta)

                    
                elif opt_algo == 'FGD':
                    #Fast gradient descent
                    # Armigo Stochastic gradient descent
                    t_next = .5*(1 + np.sqrt(1 + 4*t_current**2))
                    cost, grad = self.gradient(x, y, lambda_, y_current)
                    w_next = y_current - eta_an * grad
                    y_next = w_next + (t_current - 1.0)/(t_next)*(w_next - w)
                    
                    #cost, grad = self.gradient(x, y, lambda_, w_next)
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

            if opt_algo == 'FGD':
                # Armijo condition
                if epoch > 0 and cost_next >= cost_old + beta * grad_next.T.dot(w_next - w_old):
                    eta_an *= eta_inc
                else:
                    eta_an *= eta_dec
                    
                w_old = w_next
                cost_old = cost_next  
                
            if not opt_algo in plt_dict:
                plt_dict[opt_algo] = list()
            #loss = sum(cost_array) / len(cost_array)
            loss_array.append(cost)
            #plt_dict[opt_algo].extend(loss_array)
            #print( 'epoch: [%04d], cost: [%08.4f]' % (epoch, loss))
            plt_dict[opt_algo].extend(cost_array)
            
            
            

        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1): ].reshape(self.num_output, self.num_hidden + 1)
        
        return timings_collector, loss_array
        
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

if __name__ == '__main__':
    
    x_train, y_train = read_dense_data(open(train_path), num_feature)
    
    # z_score_normalize
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_train = 1.0 * (x_train - mean) / (std + 0.0001)
    x_test, y_test = read_dense_data(open(test_path), num_feature)
    x_test = 1.0 * (x_test - mean) / (std + 0.0001)
    
    clf = NeuralNet(num_input ,num_hidden, num_output)
    
    for opt_algo in opt_algo_set:
        if opt_algo == 'FGD':
            num_epoch=10
        else:
            num_epoch=15
        timings_collector, cost_array = clf.train(x_train, y_train, opt_algo,num_epoch) 
        acc_train = clf.test(x_train, y_train)
        print(  'Training accuracy for Neural Network : %lf%%' % (100.0 * acc_train))
        acc_test = clf.test(x_test, y_test)
        print( 'Test accuracy for Neural Network : %lf%%' % (100.0 * acc_test))
        if opt_algo == 'FGD':
           timings_collector_fcg, cost_array_fcg = timings_collector, cost_array
        elif opt_algo == 'SGD':
           timings_collector_sgd, cost_array_sgd = timings_collector, cost_array
        elif opt_algo == 'NAG':
           timings_collector_nag, cost_array_nag = timings_collector, cost_array
        elif opt_algo == 'Momentum':
           timings_collector_mom, cost_array_mom = timings_collector, cost_array
        elif opt_algo == 'Adadelta':
           timings_collector_ada, cost_array_ada = timings_collector, cost_array
           
    plt.subplot(111)
    #plt.title('Comparing with different Gradient Descent Optimization')
    plt.xlabel('# of epoch')
    plt.ylabel('cost')
    #plt.xlim(0.0, 1000.0)# set axis limits
    plt.ylim(0.0, 3.5)
    
    
    proxy = list()
    legend_array = list()
    for index, (opt_algo, epoch_cost) in enumerate(plt_dict.items()):
        selected_color = plt_color_array[index % len(plt_color_array)]
        plt.plot(range(len(epoch_cost)), epoch_cost, '-%s' % selected_color[0])
        proxy.append(Rectangle((0,0), 0,0, facecolor=selected_color))
        legend_array.append(opt_algo)
    plt.legend(proxy, legend_array)
    plt.savefig("output_images/comparing.png",fmt="png")
    plt.show()
    
#time
    plt.subplot(111)
    plt.plot(timings_collector_fcg, cost_array_fcg,'blue')
    plt.plot(timings_collector_sgd, cost_array_sgd,'green')
    plt.plot(timings_collector_nag, cost_array_nag,'red')
    plt.plot(timings_collector_mom, cost_array_mom,'cyan')
    plt.plot(timings_collector_ada, cost_array_ada,'magenta')
    #plt.title('Comparing with different Gradient Descent Optimization')
    plt.xlabel('CPU time in secounds')
    plt.ylabel('cost')
    plt.ylim(0.0, 3.5)
    plt.legend(['FGD', 'SGD', 'NAG', 'Momentum', 'Adadelta'], fontsize=12, loc=1)
    plt.savefig("output_images/comparing_time.png",fmt="png")
    plt.show()
    print('Training results:')
    print('FGD') 
    print("CPU time = {}".format(timings_collector_fcg[-1]))
    print("Cost = {}".format(cost_array_fcg[-1]))
    print('SGD') 
    print("CPU time = {}".format(timings_collector_sgd[-1]))
    print("Cost = {}".format(cost_array_sgd[-1]))
    print('NAG') 
    print("CPU time = {}".format(timings_collector_nag[-1]))
    print("Cost = {}".format(cost_array_nag[-1]))
    print('FMomentum') 
    print("CPU time = {}".format(timings_collector_mom[-1]))
    print("Cost = {}".format(cost_array_mom[-1]))
    print('Adadelta') 
    print("CPU time = {}".format(timings_collector_ada[-1]))
    print("Cost = {}".format(cost_array_ada[-1]))