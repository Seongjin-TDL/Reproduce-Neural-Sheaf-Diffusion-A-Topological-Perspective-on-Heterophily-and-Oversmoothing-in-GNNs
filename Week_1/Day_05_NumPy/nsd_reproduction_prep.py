import numpy as np

# Example: Simple graph Laplacian for undirected graph
# Adjacency matrix for 4-node line graph : 0-1-2-3
adj = np.array([
    [0,1,0,0],
    [1,0,1,0],
    [0,1,0,1],
    [0,0,1,0]
])
print("Adjacency matrix:\n", adj)

# Degree matrix (diagonal)
degree = np.diag(np.sum(adj, axis=1))   # Q. What does it mean?
print("Degree matrix:\n", degree)

# Laplacian: L = D - A
laplacian = degree - adj
print("Graph Laplacian:\n", laplacian)

# In NSD, the sheaf Laplacian generalizes graph Laplacian using restriction maps.
# For now, just show how I would set up the data structures:
edge_indices = np.array([[0,1,2]
                         ,[1,2,3]])  # edges : 0-1, 1-2, 2-3
edge_maps = [np.eye(1) for _ in
             range(edge_indices.shape[1])]  # Q. What does it mean?
print("Edge indices:\n", edge_indices)
print("Edge maps:", edge_maps)

import torch
laplacian_torch = torch.from_numpy(laplacian).float()
print("Laplacian as PyTorch tensor:\n", laplacian_torch)