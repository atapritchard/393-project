#############################################################################################
# Code for encoding graph as transition matrix, Markov chain implementation + random sampling
# 11/11/20
# Brendon Gu, Emilia French, Alden Pritchard
#############################################################################################


import os
import numpy as np
from tqdm import tqdm
from pdb import set_trace as debug


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
    for i in range(1000):
        p_inf_old = p_inf
        p_inf = np.matmul(p_inf, p)
        if np.max(np.abs(p_inf_old - p_inf)) < 10e-5:
            print('Convergence to tolerance 10e-5 at iteration', i+1)
            break

    if verbose:
        print('Stationary Distribution for Network')
        for row in p_inf:
            print(list(map(lambda x: round(x, 3), row.tolist())))

    return p_inf[0]


# Returns a list of vertices visited in order of visitation, not allowed to visit the same vertex twice
def random_route(start_vtx, n_steps, p, verbose=False):
    path = [start_vtx]
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
        path.append(next_vtx)
        visited.add(next_vtx)
        start_vtx = next_vtx

    return path


# Simulates a bunch of random routes around the network
def generate_random_routes(p, num_routes, route_min_len, route_max_len, sources=None, sinks=None):
    num_steps = np.random.uniform(route_min_len, route_max_len, num_routes)
    start_vtxs = []
    for i in range(num_routes):
        start_vtxs.append(sources[np.floor(np.random.uniform(0, len(sources))).astype(int)])
    paths = []
    for i in tqdm(range(num_routes), desc='computing paths', ncols=80):
        # debug()
        if start_vtxs[i] not in sources:
            continue
        paths.append(random_route(start_vtxs[i], int(num_steps[i]), p))
    paths1 = list(filter(lambda path: path[0] in sources and path[-1] in sinks, paths))
    if not paths1:
        debug()
    return paths


def add_station(p, vtx_idx, alpha=1.25):
    # vtx_idx = vtx - 1
    for i in range(p.shape[0]):
        edge_prob = p[i][vtx_idx]
        if edge_prob < 10e-5:
            continue
        beta = (1 - edge_prob * alpha) / (1 - edge_prob)
        p[i] = p[i] * beta
        p[i, vtx_idx] = p[i, vtx_idx] / beta * alpha

    if np.max(p) > 1:
        print('Error: chosen alpha produces invalid matrix')
        raise ValueError
    for i in range(p.shape[0]):
        try:
            assert (np.sum(p[i]) - 1.0) < 10e-4
        except:
            print('Invalid row', i)
            print(alpha, beta)
            debug()

    return p


def add_stations(p, vtxs, alpha=1.25):
    for v in vtxs:
        p = add_station(p, v, alpha=alpha)
    return p


def display(transition_matrix):
    print('', ''.join(list(map(lambda x: str(x) + '  ', list(range(transition_matrix.shape[0]))))))
    for i in range(transition_matrix.shape[0]):
        row = transition_matrix[i]
        print(i+1, ' '.join(list(map(str, row.tolist()))))


def path_experiment(p_mat, sources=None, sinks=None, verbose=False):
    counts = {}
    for i in range(p_mat.shape[0]):
        counts[i + 1] = 0

    n_paths = 10 ** 4
    paths = generate_random_routes(p_mat, n_paths, 1, 12, sources=sources, sinks=sinks)
    for path in paths:
        for vtx in path:
            counts[vtx + 1] += 1

    total = sum(list(map(lambda x: counts[x], counts)))

    pcts = []
    if total == 0:
        debug()
    for i in range(p_mat.shape[0]):
        pcts.append((i+1, round(100 * counts[i + 1] / total, 2)))
    if verbose:
        print('Vertex\tPercent of Total Visits')
        for row in pcts:
            print(row)
    else:
        return np.array(pcts)


def norm(vec, p=2):
    if p == 2:
        return np.sum(vec**2)
    elif p == 1:
        return np.sum(np.abs(vec))
    elif p == np.float('inf'):
        return np.max(vec)
    else:
        raise NotImplemented


# Sample program: read in sioux_falls network, generate 100,000 random paths, see which vertices have the most visits
def main():
    p, w = read_network('sioux_falls.txt')
    pi0 = stationary_dist(p, verbose=False)

    p_new = add_stations(p, [3, 6, 9, 18, 20, 23], alpha=1.5)
    pi1 = stationary_dist(p_new, verbose=False)

    # for i in range(len(pi0)):
    #     print(pi0[i], '\t', pi1[i])
    print(norm(100*(pi0-pi1), p=2))

    ps0 = path_experiment(p)
    ps1 = path_experiment(p_new)
    print(norm(np.array(ps1)-np.array(ps0), p=2))
    # for i in range(len(ps0)):
    #     print(ps0[i], '\t', ps1[i])


if __name__ == '__main__':
    main()
