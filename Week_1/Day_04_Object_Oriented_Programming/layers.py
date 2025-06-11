import torch
import torch.nn as nn

class SheafLayer(nn.Module):
    def __init__(self, edge_index: torch.Tensor):
        super().__init__()
        self.edge_index = edge_index
        self.num_nodes = edge_index.max().item() + 1

    def propagate(self, x: torch.Tensor):
        print(f"Propagating node features via edges: {self.edge_index.tolist()}")
        return x

class SheafDiffusion(nn.Module):
    def __init__(self, edge_index: torch.Tensor, args):
        super().__init__()
        self.edge_index = edge_index
        self.args = args

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Subclasses must implement forward()")

class DiscreteDiagSheafDiffusion(SheafDiffusion):
    def __init__(self, edge_index: torch.Tensor, args):
        super().__init__(edge_index, args)
        self.lin = nn.Linear(args.input_dim, args.hidden_dim)
        self.layers = nn.ModuleList([SheafLayer(edge_index) for _ in range(args.num_layers)])

    def forward(self, x: torch.Tensor):
        x = self.lin(x)
        for layer in self.layers:
            x = layer.propagate(x)
        return x
