# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D

import random

def dot(w, x):
    result = 0
    for i in range(len(x)):
        result += (w[i] * x[i])
    
    return result

def sign(u):
    if u >= 0:
        return 1
    else:
        return 0
    
def atualiza_pesos(w, alfa, e, x):
    for i in range(len(x)):
        w[i] = w[i] + alfa * e * x[i]
    
    return w 

def run_perceptron(w, b, X):
    y = []
    for i in range(len(X)):
        u = dot(w, X[i]) + b
        y.append(sign(u))
    
    return y   
        
def perceptron(max_it, alfa, X, d):
    # pesos sinápticos
    w = [random.uniform(-0.5, 0.5) for _ in range(2)]
    # bias
    b = random.uniform(-0.5, 0.5)
    nr_padroes = len(X)
    t = 0
    E = 1
    eqm = 0
    eqm_por_epoca = []
    epocas = []
    
    while(t < max_it) and (E > 0):
        E = 0
        
        for i in range(nr_padroes):
            u = dot(w, X[i]) + b
            y = sign(u)
            e = d[i] - y
            # atualiza pesos e bias
            w = atualiza_pesos(w, alfa, e, X[i])
            b = b + alfa * e
            E = E + (e) ** 2
        
        
        eqm = float(E) / len(X)
        eqm_por_epoca.append(eqm) 
        epocas.append(t)
        #print "erro quadrático médio: ", eqm
        #print "epoch: ", t
        t = t + 1
        
    plt.plot(epocas, eqm_por_epoca, "r-")
    plt.grid(True,  which="both")
    plt.title("Erros por epoca")
    plt.ylabel("numero de erros")
    plt.xlabel("epoca")
    plt.show()
        
    return w, b

# X e d correspondem à função booleana AND
# padroes de entrada
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
# valores desejados
d = [0, 0, 0, 1]
# número máximo de iterações
max_it = 50
# taxa de aprendizado
alfa = 0.1
w, b = perceptron(max_it, alfa, X, d)
# resposta do perceptron após treinamento
y = run_perceptron(w, b, X)
print("\nresposta do perceptron para funcao AND")
print(y)