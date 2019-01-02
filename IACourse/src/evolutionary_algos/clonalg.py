# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plot_function(x, y, plot_type):
    plt.plot(x, y, plot_type)
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y) + 0.1)
    plt.grid()
    plt.show()

class CLONALG:
    def __init__(self, nr_gen, nr_indiv, nr_atr, nr_clones, x_bounds):
        self.nr_gen = nr_gen
        self.nr_indiv = nr_indiv
        self.nr_atr = nr_atr
        self.nr_clones = nr_clones
        self.P = np.random.rand(nr_indiv, nr_atr)
        #self.P = np.random.randint(100, size=(nr_indiv, nr_atr))
        self.beta = 1
        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]
        self.P = self.init_indiv()#np.random.rand(nr_indiv, nr_atr)
        self.fit = self.fitness(self.P) # fitness inicial
        
    def init_indiv(self):
        P = np.empty((self.nr_indiv, self.nr_atr))
        for i in range(self.nr_indiv):
        #    while True:
            P[i] = np.random.random_integers(1, 6)
        #        if (self.is_inside_bounds(aux_indiv)):
        #P[i] = np.random.random_integers(1, 6)
        #            break
        return P       
        
    def mutar(self, indiv_pai, mut):
        individuo_clone = indiv_pai + mut
        
        return individuo_clone
        
    def clone(self):
        # matriz com indivíduos pais e respectivos clones
        P_c = np.empty([len(self.P), self.nr_clones + 1, self.nr_atr])
        # melhores clones de cada indivíduo
        C = np.empty([len(self.P), self.nr_atr])
        # fitness dos clones de um indivíduo
        fit_clones = np.empty([self.nr_clones + 1])
        # armazena fitness dos clones mais adaptados
        melhor_fit_clones = np.empty([len(self.P)])
        
        for i in range(len(self.P)):
            # primeiro indivíduo é o pai
            P_c[i][0] = self.P[i]
            
            # gerando clones do cada indivídio i
            for k in range(1, self.nr_clones + 1):
                while True:
                    mut = np.random.randn(self.nr_atr) / self.beta * np.exp(-(self.fit[i] / max(self.fit)))
                    aux_C = self.mutar(P_c[i][0], mut)
                    
                    if (self.is_inside_bounds(aux_C)):
                        P_c[i][k] = aux_C
                        break
            
            # calculando o fitness dos clones do indivíduo i
            fit_clones = self.fitness(P_c[i])
            # encontra clone com melhor fitness
            melhor_fit_clones_id = np.argmin(fit_clones)
            # armazena melhor fitness
            melhor_fit_clones[i] = fit_clones[melhor_fit_clones_id]                
            # armazena clone do indivíduo i com melhor fitness
            C[i] = P_c[i][melhor_fit_clones_id]
        
        return (C, melhor_fit_clones)
    
    def is_inside_bounds(self, x):
        if (x >= self.x_min and x <= self.x_max):
            return True
        else:
            return False
        
    def busca(self):
        for i in range(self.nr_gen):
            #print("geração %d" % (i + 1))
            self.P, self.fit = self.clone()
            #print(self.P)                    
        
    def fitness(self, individuos):  
        fit = np.empty([len(individuos)])
        
        for i in range(len(individuos)):
            x = individuos[i]
            #fit[i] = 2 ** ((-2 * (x - 0.1) / 0.9) ** 2) * ((np.sin(5 * np.pi * x)) ** 6)
            #fit[i] = x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1
            #fit[i] = (60 * x[0]) + (100 * x[1]) - (1.5 * x[0] ** 2) - (1.5 * x[1] ** 2) - (x[0] * x[1]) 
            #fit[i] = 120 * x[0] - 3 * x[0] ** 2 + 200 * x[1] - 3 * x[1] ** 2 - 2 * x[0] * x[1]
            #fit[i] = 4500 - 5 * x[0] ** 2 - 5 * x[1] ** 2 + 160 * x[0] + 205 * x[1] - 2 * x[0] - x[1]
            fit[i] = x ** 3 - 10.5 * x ** 2 + 30 * x + 20
            
        return fit
    
# executando o CLONALG
# número de gerações
nr_gen = 200
# número de indivíduos
nr_individuos = 10
# número de atributos de cada indivíduo
nr_atr = 1
# número de clones para cada indivíduo
nr_clones = 5
# limites no eixo x
x_bounds = [1, 6]

clonalg = CLONALG(nr_gen, nr_individuos, nr_atr, nr_clones, x_bounds)
clonalg.busca()
print(clonalg.P)
print("\n\n")
print(clonalg.fit)
# x = [15.8, 20.4]
# z = 4500 - 5 * x[0] ** 2 - 5 * x[1] ** 2 + 160 * x[0] + 205 * x[1] - 2 * x[0] - x[1]
# print(z)
#x = np.arange(0, 1, 0.005)
#y = (2 ** ((-2 * (x - 0.1) / 0.9) ** 2)) * ((np.sin(5 * np.pi * x)) ** 6)
# plt.plot(x, y, 'r-')
# plt.plot(clonalg.P, clonalg.fit, 'bo')
# plt.xlim(min(x), max(x))
# plt.ylim(min(y), max(y) + 0.1)
# plt.grid()
# plt.show()

# plot searching space and initial population
# _x = np.arange(-2, 2.05, 0.1)
# _y = np.arange(-2, 2.05, 0.1)
# xx, yy = np.meshgrid(_x, _y)
# z = xx * np.sin(4 * np.pi * xx) - yy * np.sin(4 * np.pi * yy + np.pi) + 1
# print(xx)
# print(yy)
# print(z)
# 
# ax = plt.figure().gca(projection='3d')
# surf = ax.scatter(xx, yy, z)#, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)  
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# plt.xlabel("x")
# plt.ylabel("y")    
# #ax.colorbar(surf, shrink = 0.5, aspect = 5)
# ax.scatter(clonalg.P[:,0], clonalg.P[:,1], clonalg.fit, s=120, c = 'k', marker = 'o')
# ax.view_init(elev = 48., azim = -101)
# #plt.ion()
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()
