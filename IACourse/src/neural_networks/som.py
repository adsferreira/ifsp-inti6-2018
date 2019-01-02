import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

def plotData(patterns, w1, w2):
    "positive class: class +1"
    "Negative class: class -1"
    mask = patterns[:,-1] > 0
    positive_class = patterns[mask]
    negative_class = patterns[-mask]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positive_class[:,0], positive_class[:,1], positive_class[:,2], c = 'b', marker = 'o')
    ax.scatter(negative_class[:,0], negative_class[:,1], negative_class[:,2], c = 'r', marker = '^')
    ax.scatter(w1[0], w1[1], w1[2], c = 'k', marker = 's',s=40)
    ax.scatter(w2[0], w2[1], w2[2], c = 'y', marker = 's',s=40)
    plt.show()

def euclidean_dist(x, w):
    return np.linalg.norm(x - w, axis=1)

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

with open('dados1.csv', 'rb') as myfile:
    reader = csv.reader(myfile)
    patterns = np.array(list(reader),dtype=float)
    
X = patterns[:,:-1]
alpha = 0.01
nr_neurons = 10
max_it = 100
  
W = np.array([[random.uniform(-0.1, 0.1) for _ in range(len(X[0]))] for _ in range(nr_neurons)])
dists = np.empty(len(W))
X = np.random.permutation(X)

t = 0
while t < max_it:
    for x in X:
        dists = euclidean_dist(x, W)
        winner_idx = np.argmin(dists)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, nr_neurons//10, W.shape[0])
        vra = gaussian[:,np.newaxis]
        
        W += gaussian[:,np.newaxis] * alpha * (x - W)
      
    alpha = 0.1 * np.exp(-t / 1000)  
    X = np.random.permutation(X)
    t += 1

plotData(patterns, W[0], W[1])