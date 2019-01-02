# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from numpy import int64

def add_noise(input_patterns, noise_percent):
    new_input_patterns = []
    changable_ids = list(range(0, len(input_patterns[0])))
    
    for input_pattern in input_patterns:
        changing_ids = random.sample(changable_ids, noise_percent)
        
        for idx in changing_ids:
            if input_pattern[idx] == 0:
                input_pattern[idx] = 1
            else:
                input_pattern[idx] = 0
            
        new_input_patterns.append(input_pattern)
            
    return new_input_patterns


def sign(u):
    y = []
    for _u in u:
        if _u <= 0:
            y.append(0)
        else:
            y.append(1)
    return y

def run_slp(W, b, X):
    outs = []
    
    for i in range(len(X)):
        # propagaço do sinal 
        u = np.dot(W, X[i]) + b
        y = sign(u)
        outs.append(y)
    
    return np.array(outs)

def slp(max_it, alpha, nr_neurons, X, D):
    W = [[random.uniform(-0.001, 0.001) for _ in range(len(X[0]))] for _ in range(nr_neurons)]
    b = [random.uniform(-0.001, 0.001) for _ in range(nr_neurons)]
    
    eqm = 100    
    epoch = 1    
    error_per_epoch = []
    epoch_vector = []
    
    while (eqm > 0) and (epoch <= max_it):
        sum_squared_error = 0
        
        for j in range(len(X)):
            u = np.dot(W, X[j]) + b
            y = sign(u)
            e = D[j] - y
            W = W + np.outer(alpha * e, X[j])
            #print "w: " , w
            b = b + (alpha * e)
            sum_squared_error = sum_squared_error + sum(e ** 2)
            
        eqm = float(sum_squared_error) / len(D)
        
        error_per_epoch.append(eqm) 
        epoch_vector.append(epoch)
        
        #print "error: ", error
        print "epoch: ", epoch        
        print "eqm: ", eqm
        print "\n"     
        epoch += 1
              
    plt.plot(epoch_vector, error_per_epoch, "r-")
    plt.grid(True,  which="both")
    plt.title("Erros por epoca")
    plt.ylabel("numero de erros")
    plt.xlabel("epoca")
    #mpl.axis([-2, 11, -2, 11])
    plt.show()
    
    return (W, b)
            
        
numbers = ['zero', 'one', 'two', 'three', 'four', 'six', 'non_num', 'nine']
input_patterns = []

with open('number_characters.csv', 'rb') as myfile:
    reader = csv.reader(myfile)
    input_patterns = np.array(list(reader), dtype=int)
    
X = input_patterns[:,0:-1]
D = input_patterns[:, -1]
  
noisy_inputs = add_noise(input_patterns, 50)
D = np.diag(np.ones(len(input_patterns), dtype=int64))

W, b = slp(50, 0.01, len(D), noisy_inputs, D)
outputs = run_slp(W, b, noisy_inputs)
for i in range(len(outputs)):
        if (outputs[i] == D[i]).all():
            print("Correct classification of the character " + numbers[i] + ".")
        else:
            print("Wrong classification! Consider retraining your neural network.")