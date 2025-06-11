# scripts/main_runner.py
import torch
from config import Args
from layers import DiscreteDiagSheafDiffusion

if __name__ == "__main__":
    edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    model = DiscreteDiagSheafDiffusion(edge_index, Args())
    x = torch.randn(4, 32)  # 4 nodes, 32 features
    output = model(x)
    print("Output shape:", output.shape)