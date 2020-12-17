import numpy as np
import random 

import starter_code

class Network:

    """
    Creates a network instance for testing. When instatiating, a network size is required and turn probability and weight matrices are optional. 

    If initialized without turn probabilities and weights, a random network is generated in which each node has 2-5 neighbors and each link has a weight uniformly chosen from 0-20. 
    """

    probs = None
    weights = None
    def __init__(self, size, p = [], w = []): 
        if p: 
            assert (size, size) == p.shape
        else:
            probs = np.zeros((size, size))
            weights = np.zeros((size, size))
            for i in range(size):
                num_neighbors = random.choice([2, 3, 4, 5])
                neighbors = random.sample(list(range(i)) + list(range(i+1, size)), num_neighbors)

                p = np.random.rand(num_neighbors)
                p /= np.sum(p)

                for j in range(num_neighbors):
                    probs[i, neighbors[j]] = p[j]
                    weights[i, neighbors[j]] = np.random.rand() * 20
            self.probs = probs
            self.weights = weights 

        weights = 0


# Sample usage
# n = Network(20)
# starter_code.print_stationary(starter_code.stationary_dist(n.probs))