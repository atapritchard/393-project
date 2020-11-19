import gurobipy as gp
from gurobipy import GRB

# Set Cover - universe is set of routes and sets are nodes 
def optimize(routes, num_nodes, threshold = 0):
    """
        routes : array of all route/cost tuples in the network
        num_nodes : total number of nodes
        threshold : check that any route with cost >= threshold has a station along it

        TODO: add parameter for max distance to travel in route before recharging

        Returns a dictionary mapping each node(station) to 0 or 1
    """
    model = gp.Model("Station Optimization")
    model.setParam("OutputFlag", 0)

    node_names = []
    for i in range(num_nodes):
        node_names.append("Station " + str(i))
    stations = model.addVars(num_nodes, vtype = GRB.BINARY, name = node_names)
    
    # route = (nodes, cost), nodes is a list of the nodes the shortest path passes through
    for route in routes: 
        if(route[1] >= threshold):
            model.addConstr(sum(stations[i] for i in route[0]) >= 1)

    model.setObjective(stations.sum(), GRB.MINIMIZE)
    model.optimize()

    # Debug
    # print(model.display())
    # model.printAttr('X')
    
    sol = {}
    for _, var in stations.items():
        sol[var.varName] = int(var.X)
    
    return sol

# Small case for testing; 4 nodes, and edges from 0 to 1, 1 to 2, 2 to 3, and 3 to 0 with weights 3, 2, 10, 3
# optimize(routes, 4) should return Station 0 and Station 1
# optimize(routes, 4, threshold = 3) should return Station 0
def test():

    routes = [((0, 1), 3), ((0, 1, 2), 5), ((0, 3), 3), ((1, 2), 2), ((1, 0, 3), 6), ((2, 1, 0, 3), 8)]
    print(optimize(routes, 4))

test()