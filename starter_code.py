#############################################################################################
# Code for encoding graph as transition matrix, Markov chain implementation + random sampling
# 11/11/20
# Brendon Gu, Emilia French, Alden Pritchard
#############################################################################################


import os
import numpy as np
from tqdm import tqdm
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import seaborn as sns

def read_network(filename):
    with open(filename, 'r') as doc:
        data = doc.read().strip().split('\n')

    # Fill transition and  weight matrices using values provided in file
    P = np.zeros((len(data), len(data)))
    W = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        assert i+1 == int(data[i].split(':')[0])
        
        # Read destination, probabilities, weights
        dests = np.array(list(map(int, data[i].split(':')[1][1:-1].split(',')))) - 1
        probs = np.array(list(map(float, data[i].split(':')[2][1:-1].split(','))))
        weights = np.array(list(map(float, data[i].split(':')[3][1:-1].split(','))))
        
        # Fill in matrices
        P[i, dests] = probs
        W[i, dests] = weights

    # Check for correctness: rows of transition matrix should sum to 1, weights should be symmetric unless weights are different in opposite directions 
    for i in range(len(data)):
        for j in range(len(data)):
            assert abs(W[i, j] - W[j, i]) < 10e-4
        assert (np.sum(P[i]) - 1.0) < 10e-4

    return P, W


# Returns approximation of stationary distribution for Markov chain (p^100)
def stationary_dist(p, verbose=False):
    # Compute stationary distribution for Markov chain
    p_inf = p
    for i in range(100):
        p_inf = np.matmul(p_inf, p)

    if verbose:
        print('Stationary Distribution for Network')
        for row in p_inf:
            print(list(map(lambda x: round(x, 3), row.tolist())))

    return p_inf


# Returns a list of vertices visited in order of visitation, not allowed to visit the same vertex twice
def random_route(start_vtx, n_steps, p, verbose=False):
    path = [start_vtx+1]
    visited = {start_vtx}
    for i in range(n_steps):
        # Generate list of possible destinations and associated probabilities
        dests, probs = [], []
        for j in range(p.shape[1]):
            if verbose: print(p[start_vtx, j], end='||')
            if p[start_vtx, j] > 10e-4 and j not in visited:
                dests.append(j)
                probs.append(p[start_vtx, j])
        
        # Exit early if there's no valid more turns we can make
        if probs == []:
            if verbose: print('Early exit')
            return path

        # Compute turn probabilities adjusted to include no repeat visits
        probs = np.array(probs) / np.array(np.sum(probs))
        for j in range(probs.shape[0]):
            probs[j] = probs[j] + np.sum(probs[:j])

        # Generate random sample between 0 and 1 to use for deciding which way to turn
        turn = np.random.uniform(0, 1)
        if verbose: print(turn, probs)
        for k in range(len(probs)):
            if turn <= probs[k] + 10e-4:
                next_vtx = dests[k]
                break

        if verbose: print(start_vtx+1, '->', next_vtx+1)
        assert p[start_vtx, next_vtx] > 10e-4
        path.append(next_vtx + 1)
        visited.add(next_vtx)
        start_vtx = next_vtx

    return path


# Simulates a bunch of random routes around the network
def generate_random_routes(p, num_routes, route_min_len, route_max_len):
    num_steps = np.random.uniform(route_min_len, route_max_len, num_routes)
    start_vtxs = np.random.uniform(0, p.shape[0], num_routes)
    paths = []
    for i in tqdm(range(num_routes), desc='computing paths', ncols=80):
        paths.append(random_route(int(start_vtxs[i]), int(num_steps[i]), p))
    return paths


def estProbs(routes, nodes):
    """

    Given a set of routes, estimate the turning probabilities associated with the network.
    Can be used to find the estimated stationary distribution of the network.
    """
    
    probs = np.zeros((nodes, nodes))
    for route in routes:
        for i in range(len(route) - 1):
            probs[route[i] - 1, route[i + 1] - 1] += 1

    # normalize rows
    row_sums = probs.sum(axis = 1)
    probs /= row_sums[:, None] 
    print(probs)

    return probs

# Creates plot of stationary distribution
def print_stationary(data):
    plot = sns.barplot(x = np.arange(1, len(data[0])+ 1), y = data[0])
    plot.set(xlabel = "Node", ylabel = "Probability", title = "Stationary Distribution")
    
    plt.show()

# Sample program: read in sioux_falls network, generate 100,000 random paths, see which vertices have the most visits
def main():
    p, w = read_network('sioux_falls.txt')
    pi = stationary_dist(p, verbose=False)
    counts = {}
    for i in range(p.shape[0]):
        counts[i+1] = 0

    n_paths = 10**5
    for path in generate_random_routes(p, n_paths, 5, 9):
        # print(path)
        for vtx in path:
            counts[vtx] += 1

    total = sum(list(map(lambda x: counts[x], counts)))
    print('Vertex\tPercent of Total Visits')
    for i in range(p.shape[0]):
        print(i+1, '\t', round(100 * counts[i+1] / total, 1), '%', sep='')


if __name__ == '__main__':
    main()