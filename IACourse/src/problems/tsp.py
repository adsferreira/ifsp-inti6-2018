import numpy as np

city_coords = np.array([[0, 0], [-1, 2], [3, 4]])
dist = np.sqrt(np.sum((city_coords[0] - city_coords[3]) ** 2))
print("Distance from the city %d to the city %d is %.2f." % (0, 2, dist))
cities = [0, 1, 2, 3]

# como criar todas as rotas?
#for i in cities:
    