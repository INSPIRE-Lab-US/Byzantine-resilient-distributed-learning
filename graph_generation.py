# graph generation
import random


def random_connected(nodes, con_rate, b=0): 
# random connected graph with connecting rate between 1-100
    graph = []
    re = 1                      # regenerate if graph assumption not satisfied
    while re:
        for _ in range(nodes):
            graph.append([])
        for row in range(nodes):
            graph[row].append(1)
            for col in range(row + 1, nodes):
                d = random.randint(1, 100)
                if d > con_rate:
                    graph[row].append(1)     #form symmetric matrix row by row
                    graph[col].append(1)
                else:
                    graph[row].append(0)
                    graph[col].append(0)
        d_max = 0
        for row in graph:
            if sum(row) > d_max:
                d_max = sum(row)
        w = [row[:] for row in graph]
        for ind, row in enumerate(w):
            d = sum(row)
            w[ind] = [col/d_max for col in row]
            w[ind][ind] = 1 - (d - 1) / d_max
        if all([sum(row) >= 2 * b + 1 for row in graph]):
            re = 0    #if any neighborhood has degree less than 2b+1, regenerate
    return w, graph   
    pass

