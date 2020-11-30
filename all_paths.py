import numpy as np
from tqdm import tqdm
import math
from collections import defaultdict
from collections.abc import Iterable
import copy

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

def flatten(L, result = []):
    for x in L:
        if isinstance(x, list):
            flatten(x, result)
        else:
            result.append(x)
    return result

def BFS(G, start, end):
    visited = []
    queue = [[start]]
    fullPath = []
    first = True
    if start == end:
        return []
    while queue:
        path = queue.pop(0)
        v = path[-1]
        #print(v)
        if v not in visited:
            neighbors = G[v]
            for neighbor in neighbors:
                newPath = [path]
                newPath.append(neighbor)
                queue.append(newPath)
                if neighbor == end:
                    fullPath = flatten(newPath)
                    return fullPath
            visited.append(v)
    return None


def minDistance(dist,queue): 
    minimum = float("Inf") 
    min_index = -1
        
    for i in range(len(dist)): 
        if dist[i] < minimum and i in queue: 
            minimum = dist[i] 
            min_index = i 
    return min_index 

def storePath(parent, j, res): 
    #Base Case : If j is source 
    if parent[j] == -1 :  
        res.append(j)
        return res
    res.append(j)
    storePath(parent , parent[j], res) 
    return res

def storeSolution(dist, parent): 
    src = 0
    paths = []
    for i in range(1, len(dist)): 
        path = storePath(parent,i, [])
        revPath = list(reversed(path))
        paths.append((revPath, round(dist[i], 4)))
    return paths

#half stolen from https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
def dijkstra(G, start): 
    rows = len(G) 
    cols = len(G[0]) 
    dists = [100000] * rows 
    parent = [-1] * rows 
    dists[start] = 0

    queue = [] 
    for i in range(rows): 
        queue.append(i) 

    while queue: 
        u = minDistance(dists, queue)  
        queue.remove(u) 

        for i in range(cols): 
            if ((G[u][i]) and (i in queue)): 
                if (dists[u] + G[u][i] < dists[i]): 
                    dists[i] = dists[u] + G[u][i] 
                    parent[i] = u 
    return(storeSolution(dists, parent))

#make adj list from adj matrix        
def makeGraph(P):
    adjList = defaultdict(list)
    for i in range(len(P)):
        for j in range(len(P[0])):
            if P[i][j] > 0:
                adjList[i].append(j)
    return adjList

sinks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
sources = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 23]
vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# all paths from vertices 1-12 to 12-24
def filter1(allPaths):
    newPaths = []
    for paths in allPaths:
        for path in paths:
            end = path[0][-1]
            if end in sources:
                newPaths.append(path)
    return newPaths

# all paths of length (not weight) greater than k
def filter2(allPaths, k):
    newPaths = []
    for paths in allPaths:
        for path in paths:
            p = path[0]
            if (len(p) > k):
                newPaths.append(path)
    return newPaths

# if sinks and sources are empty, find paths longer than k
def doDijkstra(G, sinks, sources, k):
    allPaths = []
    if len(sinks) != 0:
        for sink in sinks:
            allPaths.append(dijkstra(G, sink))
        return filter1(allPaths)
    else:
        for v in vertices:
            allPaths.append(dijkstra(G, v))
        return filter2(allPaths, k)

def main():
    p, w = read_network('sioux_falls.txt')
    G = copy.deepcopy(w)
    paths1 = doDijkstra(G, sinks, sources, 0)
    paths2 = doDijkstra(G, [], [], 5)
    #print(paths2)

if __name__ == '__main__':
    main()
