import networkx as nx
import math
import matplotlib.pyplot as plt
from pyvis.network import Network
import heapq
def Dijkstra (G, source):

    #Initializing maps for nodes, to specify weights and edges
    distance = {v: math.inf for v in G.nodes()}
    predecessor = {v: None for v in G.nodes()}
    pq = [(0, source)]
    distance[source] = 0

    #The actual algorithm running

    # TODO : Fix this stupid Dijkstra
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > distance[current_node]:
            continue
        else:
            for neighbor in G.get_neighbors(current_node, source):
                distance = current_dist + weight
                distance[neighbor] = current_dist + weight
                heapq.heappush(pq, (distance, neighbor))
                predecessor[neighbor] = current_node

    return distance, predecessor

def main():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ('S', 'A', 2),
        ('S', 'C', -4),
        ('C', 'A', 3),
        ('A', 'B', 2)
    ])

   # nx.draw(G, with_labels=True)
    nx.draw(G, with_labels=True, node_color='#BD4B83', edge_color='#FCC5E1', node_size=800)
    plt.savefig("graph.png")
    plt.close()

    distance, predecessor = Dijkstra(G, 'S')
    print(distance)
    print(predecessor)

if __name__ == "__main__":
    main()
