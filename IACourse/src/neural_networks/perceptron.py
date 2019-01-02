# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt

def plotData(patterns):
    "positive class: class +1"
    "Negative class: class -1"
    mask = patterns[:,-1] > 0
    positive_class = patterns[mask]
    negative_class = patterns[-mask]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positive_class[:,0], positive_class[:,1], positive_class[:,2], c = 'b', marker = 'o')
    ax.scatter(negative_class[:,0], negative_class[:,1], negative_class[:,2], c = 'r', marker = '^')
    plt.show()

def dot(w, x):
    resultado = 0
    print(w)
    print(x)
    
    for i in range(len(x)):
        resultado += w[i] * x[i]
        
    return resultado

def sign(u):
    if u >= 0:
        return 1
    else:
        return 0
    
def atualiza_pesos(w, alfa, e, x):
    for i in range(len(w)):
        w[i] = w[i] + alfa * e * x[i]
        
    return w

def roda_perceptron(w, b, X):
    resposta = []
    
    for i in range(len(X)):
        # propagação do sinal 
        u = dot(w, X[i]) + b
        y = sign(u)
        resposta.append(y)
    
    return resposta

def perceptron(max_it, alfa, X, d):
    w = [random.uniform(-0.5, 0.5) for _ in range(2)]
    b = random.uniform(-0.5, 0.5)
    
    t = 1
    E = 1
    
    eqm_por_epoca = []
    epocas = []
    
    while(t <= max_it) and (E > 0):
        E = 0
        
        for i in range(len(X)):
            # propagação do sinal 
            u = dot(w, X[i]) + b
            y = sign(u)
            # calcula o erro
            e = d[i] - y
            # atualiza os pesos
            w = atualiza_pesos(w, alfa, e, X[i])
            b = b + alfa * e
            E = E + (e) ** 2
        
        
        eqm = float(E) / len(X)
        eqm_por_epoca.append(eqm) 
        epocas.append(t)
        
        t = t + 1
        
    plt.plot(epocas, eqm_por_epoca, "r-")
    plt.grid(True,  which="both")
    plt.title("Erros por epoca")
    plt.ylabel("numero de erros")
    plt.xlabel("epoca")
    plt.show()
    
    return w, b
        
# inicio do programa        
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

d = [0, 0, 0, 1]

alfa = 0.001
max_it = 50
w, b = perceptron(max_it, alfa, X, d)
resposta = roda_perceptron(w, b, X)
print("resposta do perceptron:")
print(resposta)
