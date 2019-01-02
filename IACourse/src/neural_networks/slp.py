# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from numpy import int64


def gen_char_images(imgs, f_names):
    for i in range(len(imgs)):
        f_name = f_names[i]
        plt.imsave(str(f_name) + '.pdf', imgs[i], cmap='gray')
        
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

def slp(max_it, alpha, nr_neurons, X, D):
    W = [[random.uniform(-0.001, 0.001) for _ in range(len(X[0]))] for _ in range(nr_neurons)]
    b = [random.uniform(-0.001, 0.001) for _ in range(nr_neurons)]
    
    eqm = 100    
    epoch = 1    
    error_per_epoch = []
    epoch_vector = []    
    
    while (eqm > 0) and (epoch <= max_it):
        #error = 0        
        sum_squared_error = 0
       
        for j in range(len(X)):
            #print len(W[j])
            u = np.dot(W, X[j]) + b
            y = sign(u)
            e = D[j] - y          
            # update weights and bias       
            W = W + np.outer(alpha * e, X[j])
            #print "w: " , w
            b = b + (alpha * e)
            #print "bias: ", bias
            sum_squared_error = sum_squared_error + sum(e ** 2)
            #print "sum_squared_error: ", sum_squared_error
            
        eqm = float(sum_squared_error) / len(D)
        error_per_epoch.append(eqm) 
        epoch_vector.append(epoch)
        
        #print "error: ", error
        print "epoch: ", epoch        
        print "eqm: ", eqm
        print "\n"        
        
        epoch += 1
    
    #print error_per_epoch    
    plt.plot(epoch_vector, error_per_epoch, "r-")
    plt.grid(True,  which="both")
    plt.title("Erros por epoca")
    plt.ylabel("numero de erros")
    plt.xlabel("epoca")
    #mpl.axis([-2, 11, -2, 11])
    plt.show()
    
    return (W, b)

def run_slp(W, b, X):
    outs = []
    
    for i in range(len(X)):
        # propagaço do sinal 
        u = np.dot(W, X[i]) + b
        y = sign(u)
        outs.append(y)
    
    return np.array(outs)
    

numbers = ['zero', 'one', 'two', 'three', 'four', 'six', 'non_num', 'nine']
input_patterns = []

with open('number_characters.csv', 'rb') as myfile:
    reader = csv.reader(myfile)
    input_patterns = np.array(map(np.int64, list(reader)))
    
# deserved values
D = np.diag(np.ones(len(input_patterns), dtype=int64))
# add noise to the images
noisy_inputs = add_noise(input_patterns, 25)
train = 1

if train:
    W, b = slp(50, 0.01, len(D), input_patterns, D)
    outputs = run_slp(W, b, noisy_inputs)
    
    for i in range(len(outputs)):
        if (outputs[i] == D[i]).all():
            print("Correct classification of the character " + numbers[i] + ".")
        else:
            print("Wrong classification! Consider retraining your neural network.")