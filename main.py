import networkx as nx 
import heapq
import time
import random
from collections import defaultdict, deque, namedtuple

G = nx.erdos_renyi_graph(n=20, p=0.2)  

for u, v in G.edges():
    G[u][v]['weight'] = random.randint(1, 10)  

for n in G.nodes():
    G.nodes[n]['node_weight'] = random.randint(1, 5)

print(f"This is the weight of node number 1: {G.nodes[1]}")


def func(u, v, d):
    node_u_wt = G.nodes[u].get("node_weight", 1)
    node_v_wt = G.nodes[v].get("node_weight", 1)
    edges_wt = d.get("weight", 1)
    return node_u_wt / 2 + node_v_wt / 2 + edges_wt



def main():
    edge_attr = G.get_edge_data(1, 2)
    print (func( 1, 2, edge_attr))
    print("shahd")

u, v = random.choice(list(G.edges()))
edge_attr = G.get_edge_data(u, v)
print(f"Edge ({u},{v}) cost:", func(u, v, edge_attr))
