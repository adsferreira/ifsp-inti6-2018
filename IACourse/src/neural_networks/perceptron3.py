# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:46:32 2014

@author: adriano
"""

import random
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D

def readMatFile(fileName):
    return sio.loadmat(fileName)    

def sign(u):
    if u <= 0:
        return -1
    else:
        return 1

def Perceptron(X, D, alfa, min_err, max_it):
    #w = [random.uniform(-0.5, 0.5) for _ in range(3)]
    #w = [0.1, 0.5, 0.1]
    w = [-0.3, 0.15, -0.2]
    #w = [0.1, 0.0002, -0.0003]
    print "initial w: " , w
    #bias = random.uniform(-0.5, 0.5)
    bias = 0.25    
    #bias = 0.05
    #bias = -0.5
    print "initial bias: ", bias
    print "\n"
    eqm = 100    
    epoch = 1    
    error_per_epoch = []
    epoch_vector = []    
    
    while (eqm > min_err) and (epoch <= max_it):
        error = 0        
        sum_squared_error = 0
       
        for j in range(len(X)):
            #print X[j]
            u = np.dot(w, X[j]) + bias
            y = sign(u)
            
            #print "d - y: ", (D[j] - y) 
            if (D[j] - y) != 0:
                error = error + 1            
                
            # update weights and bias            
            w = w + np.dot(alfa * (D[j] - y), X[j])
            #print "w: " , w
            bias = bias + (alfa * (D[j] - y))
            #print "bias: ", bias
            sum_squared_error = sum_squared_error + ((D[j] - y) ** 2)
            #print "sum_squared_error: ", sum_squared_error
            
        eqm = float(sum_squared_error) / len(D)
        error_per_epoch.append(error) 
        epoch_vector.append(epoch)
        
        print "error: ", error
        print "epoch: ", epoch        
        print "eqm: ", eqm
        print "\n"        
        
        epoch += 1
    
    print error_per_epoch    
    plt.plot(epoch_vector, error_per_epoch, "r-")
    plt.grid(True,  which="both")
    plt.title("Erros por epoca")
    plt.ylabel("numero de erros")
    plt.xlabel("epoca")
    #mpl.axis([-2, 11, -2, 11])
    plt.show()
    
    return (w, bias)

def getPatternsAndTargets(dataSet):
    x = [[elem[0], elem[1], elem[2]] for elem in dataSet['x']]
    d = [elem[0] for elem in dataSet['desejado']]
    
    return (x, d)
            
def plotData(x, d):
    x_d = [x[idx] + [d[idx]] for idx in range(len(x))]
    
    x_positive = [elem for elem in x_d if elem[3] == 1]
    x_negative = [elem for elem in x_d if elem[3] == -1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([elem[0] for elem in x_positive], [elem[1] for elem in x_positive], [elem[2] for elem in x_positive], c = 'b', marker = 'o')
    ax.scatter([elem[0] for elem in x_negative], [elem[1] for elem in x_negative], [elem[2] for elem in x_negative], c = 'r', marker = '^')
    plt.show()
    
def plotData2(patterns):
    "positive class: class +1"
    "Negative class: class -1"
    
    mask = patterns[:,-1] > 0
        
    positive_class = patterns[mask]
    negative_class = patterns[mask]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positive_class[:,0], positive_class[:,1], positive_class[:,2], c = 'b', marker = 'o')
    ax.scatter(negative_class[:,0], negative_class[:,1], negative_class[:,2], c = 'r', marker = '^')
    plt.show()

def plotPlane(w, b, x, d):
    x_d = [x[idx] + [d[idx]] for idx in range(len(x))]
    
    x_positive = [elem for elem in x_d if elem[3] == 1]
    x_negative = [elem for elem in x_d if elem[3] == -1]    
    
    xx, yy = np.meshgrid(range(-5, 5), range(-5, 5))
    print "\n", xx    
    print "\n", yy
    # create x,y
    # calculate corresponding z
    z = (-w[0] * xx - w[1] * yy - b) * 1. / w[2]
    #print z
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.scatter([elem[0] for elem in x_positive], [elem[1] for elem in x_positive], [elem[2] for elem in x_positive], c = 'b', marker = 'o')
    plt3d.scatter([elem[0] for elem in x_negative], [elem[1] for elem in x_negative], [elem[2] for elem in x_negative], c = 'r', marker = '^')
    plt3d.plot_surface(xx, yy, z)  
    plt.xlabel("x")
    plt.xlabel("y")
    plt.xlabel("z")    
    plt.show()
    

#dataSet = readMatFile('dados3.mat')   
#x, d = getPatternsAndTargets(dataSet)
    
    
with open('dados3.csv', 'rb') as myfile:
    reader = csv.reader(myfile)
    patterns = np.array(list(reader),dtype=float)
    

    
X = patterns[:,:-1]
D = patterns[:,-1]


    
#print x[0]
#print np.dot(0.1, x[0])
#plotData2(patterns)
#w, bias = Perceptron(X, D, 0.001, 0.0, 40)
#plotPlane(w, bias, X, D)



