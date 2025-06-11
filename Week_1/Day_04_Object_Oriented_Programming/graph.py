# nsd/graph.py
import torch

class Node:
    # Represents a node in a graph with features.
    def __init__(self, node_id: int, features: torch.Tensor):
        self.id = node_id           # Unique identifier (e.g., 0,1,2)
        self.features = features    # Tensor of shape [num_features]

    def update_features(self, new_features: torch.Tensor):
        # Simulates updating node features during message passing.
        self.features = new_features
        print(f"Node {self.id} updated features to {new_features}")
