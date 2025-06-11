import torch

def get_graph_connectivity():
    """
    Defines a static, simple graph structure. 
    In a real project, this would be loaded from a dataset. 
    The NSD model takes 'edge_index' as a key input.
    This function simulates providing that input.

    Returns:
        torch.Tensor: A tensor of shape [2, num_edges] representing graph connections.
        int: The number of nodes in the graph.
    """
    # A simple graph: 4 nodes, connected in a line (0-1, 1-2, 2-3)
    edge_index = torch.tensor([
        [0,1,2],    # Source nodes
        [1,2,3]     # Target nodes
    ], dtype=torch.long)
    num_nodes = 4
    print("Module 'graph_operations' : Generated a sample graph edge_index.")
    return edge_index, num_nodes

def compute_node_degrees(edge_index, num_nodes):
    """
    Calculates the degree for each node (number of outgoing edges). 
    Node degrees are fundamental for building graph Laplacians, 
    including the sheaf Laplacian mentioned in the NSD paper, which often involves a degree matrix D.
    
    Args:
        edge_index (torch.Tensor): The graph's edge index.
        num_nodes (int): The total number of nodes in the graph.

    Returns:
        torch.Tensor: A 1D tensor of size [num_nodes] with the degree of each node.
    """
    # Use bincount to efficiently count degrees from the source nodes
    degrees = torch.bincount(edge_index[0], minlength=num_nodes)
    print(f"Module 'graph_operations': Computed node degrees: {degrees}")
    return degrees
