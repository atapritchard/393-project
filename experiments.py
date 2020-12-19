import csv
import copy
import numpy as np
from tqdm import tqdm
from all_paths import *
from starter_code import *


# Measure impact of adding charging stations at specified vtxs - stationary dist metric
def prob_measure(base_transition_matrix, stations, sources, sinks):
    # Ideal traffic flow should distribute traffic evenly across roads
    # In this way one way to rank traffic flows is to rank by max flow through any edge - smallest max is best flow
    new_transition_matrix = add_stations(base_transition_matrix, stations, alpha=1.5)
    base_pcts = path_experiment(base_transition_matrix, sources=sources, sinks=sinks, verbose=False)
    new_pcts = path_experiment(new_transition_matrix, sources=sources, sinks=sinks, verbose=False)
    e1, e2, e3 = np.sum(base_pcts[:, 1] - new_pcts[:, 1])**2, max(base_pcts[:, 1]), max(new_pcts[:, 1])
    return e1, e2, e3


# Measure impact of adding charging stations at specified vtxs - path weights metric
def path_measure(G, stations, sources, sinks):
    stationPaths = dijkstraStations(G, stations, sources, sinks, 0)
    return stationPaths[-1]


def pickn(n, n_max):
    nums = set()
    while len(nums) < n:
        nums.add(int(np.random.uniform(0, n_max)))
    return nums


def main():
    p, w = read_network('sioux_falls.txt')
    G = copy.deepcopy(w)
    source_sink_sets = [  # All vtxs are -1 to account for CPU counting from 0
        ([1, 6, 17], [13]),
        ([19, 20], [0, 2]),
        ([5, 6], [11, 12]),
        ([16, 18], [2, 3]),
        ([11, 12], [6, 17]),
    ]

    n = 4
    station_placements = {
        # Random vtx
        'Random1': pickn(n, p.shape[0]),
        'Random2': pickn(n, p.shape[0]),
        'Random3': pickn(n, p.shape[0]),

        # Greedy 1: pick vtxs from outermost edges
        'Greedy1': [0, 7, 19, 12],

        # Greedy 2: s-t cut on vtxs furthest apart (2 & 24), pick n from somewhere where cut straight across size n
        'Greedy2': [0, 4, 7, 17],

        # Greedy 3: any set of n vtxs where each vtx is at least 3 edges away from another vtx
        'Greedy3': [3, 7, 18, 23],

        # Heuristic 1: n vtxs each 1 removed from outermost edges
        'Heuristic1': [3, 22, 7, 19],

        # Heuristic 2: s-t cut on vtxs furthest apart (2 & 24), pick n from where equidistant
        'Heuristic2': [8, 10, 14, 15],

        # Heuristic 3: any set of n vtxs where each vtx is at least 2 edges away from another vtx while avoiding outer
        'Heuristic3': [4, 10, 16, 21]
    }

    print('Vertex #s below are adjusted to reflect true numbers and not cpu references')
    dataset = [['Sources', 'Sinks', 'Stations', 'Algo', 'PathLength', 'PathWeight', 'L2_Diff', 'MaxFlowBase',
                'MaxFlowNew']]
    for sources, sinks in tqdm(source_sink_sets, ncols=80, desc='outer loop'):
        for station_key in tqdm(station_placements, ncols=80, desc='inner loop'):
            p, w = read_network('sioux_falls.txt')
            G = copy.deepcopy(w)

            stations = station_placements[station_key]
            path_stuff = path_measure(G, stations, sources, sinks)
            path_len = len(path_stuff[0])
            path_weight = path_stuff[1]

            prob_stuff = prob_measure(p, stations, sources, sinks)
            l2_diff, max_base_traffic, max_new_traffic = prob_stuff

            dataset.append([sources, sinks, stations, station_key, path_len, path_weight, l2_diff, max_base_traffic,
                            max_new_traffic])

    with open('results.csv', 'w', newline='') as file:
        csv.writer(file, delimiter=',').writerows(dataset)


if __name__ == '__main__':
    main()
