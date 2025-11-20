import networkx as nx
import math
import matplotlib.pyplot as plt
from pyvis.network import Network

def Bellman_Ford(G, source):

    #Initializing maps for nodes, to specify weights and edges
    distance = {v: math.inf for v in G.nodes()}
    predecessor = {v: None for v in G.nodes()}

    distance[source] = 0

    #The actual algorithm running
    for _ in range (len(G.nodes()) - 1):
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1)
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                predecessor[v] = u
            if not G.is_directed(): #Also relaxing in the opposite direction in the case of an undirected graph
                if distance[v] + w < distance[u]:
                    distance[u] = distance[v] + w
                    predecessor[u] = v

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        if distance[u] + w < distance[v]:
            raise  ValueError("Negative weight cycle detected")

    return distance, predecessor

def main():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (1, 2, 4),
        (1, 3, 2),
        (5, 2, 3),
        (4, 1, 1),
        (2, 4, 2),
        (5, 1, 2),
        (4, 3, 5)
    ])

   # nx.draw(G, with_labels=True)
    nx.draw(G, with_labels=True, node_color='#BD4B83', edge_color='#FCC5E1', node_size=800)
    plt.savefig("graph.png")
    plt.close()
    net = Network(notebook=True)
    distance, predecessor = Bellman_Ford(G, 1)
    net.show("graph.html")

    print(distance)
    print(predecessor)

if __name__ == "__main__":
    main()
