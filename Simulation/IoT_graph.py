import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# 1. Load the dataset
# This will use the 'data/Cora' folder you already created
dataset = Planetoid(root='data/Cora', name='Cora') #To be checked (Our dataset won't have features)
data = dataset[0]

print("=== Cora Dataset Tensors ===")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Node features shape (x): {data.x.shape}") #To be checked
print(f"Labels shape (y): {data.y.shape}") #To be checked
print("==============================")

# 2. Convert the PyTorch Geometric Data object to a NetworkX Graph
G = to_networkx(data, to_undirected=True)

# 3. Pick a starting node and find its 2-hop neighborhood
# We use 'source' here to satisfy the NetworkX requirement
node_id = 0
subset_nodes = list(nx.single_source_shortest_path_length(G, source=node_id, cutoff=2).keys())
subgraph = G.subgraph(subset_nodes)

# 4. Prepare for plotting
plt.figure(figsize=(12, 8))

# Get colors for ONLY the nodes in our subset
# data.y contains the category (0-6) for every paper
node_colors = [data.y[i].item() for i in subgraph.nodes()]

# 5. Draw the graph
# 'spring_layout' spaces the nodes out so they don't overlap too much
pos = nx.spring_layout(subgraph, seed=42)

nx.draw(subgraph,
        pos=pos,
        node_color=node_colors,
        with_labels=True,
        cmap=plt.cm.rainbow,
        node_size=600,
        edge_color='silver',
        linewidths=1,
        font_size=8)

plt.title(f"2-Hop Neighborhood of Paper #{node_id}\n(Colors = Different Research Fields)")
plt.show()
# 1. Define a simple 1-layer GNN
# Input features: 1433 (words in Cora)
# Output features: 16 (a smaller, "compressed" representation)
conv1 = GCNConv(dataset.num_node_features, 16)

# 2. Pass the data through the layer
# GCN needs two things: the Node Features (x) and the Connectivity (edge_index)
x = data.x
edge_index = data.edge_index

# Apply the convolution
h = conv1(x, edge_index)
h = F.relu(h) # Activation function

print("\n=== GNN Tensor Transformation ===")
print(f"Original shape (Input): {x.shape}")
print(f"Processed shape (Hidden): {h.shape}")

# 3. Look at the "New" features of Paper 0
print(f"Paper 0's new 'Message' (first 5 elements): {h[0][:5].detach()}")

class GNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_Model, self).__init__()
        # Layer 1: Takes 1433 features -> 16 hidden features
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Layer 2: Takes 16 hidden features -> 7 classes (categories)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Second layer
        x = self.conv2(x, edge_index)

        # Final output: Log-Softmax (Standard for classification)
        return torch.log_softmax(x, dim=1)


# Initialize model, optimizer, and loss function
model = GNN_Model(dataset.num_node_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss

print("\n=== Starting Training ===")

# Simple Training Loop
model.train()
for epoch in range(101):
    optimizer.zero_grad()
    out = model(data)

    # We only calculate loss on the "train_mask" nodes
    # (The papers we are 'allowed' to see during study)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss.item():.4f}')

# Evaluation
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'\nFinal Accuracy on Test Papers: {acc:.4f}')