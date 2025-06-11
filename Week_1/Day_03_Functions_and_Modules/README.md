# Day 3: Graph Operations & Modular Code Structure

## Purpose

Learn to structure code into reusable modules and implement foundational graph operations relevant to the Neural Sheaf Diffusion (NSD) paper.  

**Key skills:**  
- Defining graph connectivity data (`edge_index`)  
- Computing node degrees (used in graph Laplacians)  
- Separating concerns into modules for readability and reusability  

---

## Project Structure
* graph_operations.py # Core graph tools
* main_runner.py # Demonstration script


---

## Key Files & Functions

### **1. `graph_operations.py`**
#### `get_graph_connectivity()`
- **Purpose**: Simulates graph connectivity data (`edge_index`) for testing.  
- **Output**:  
  - `edge_index`: Tensor of shape `[2, num_edges]` (source/target nodes).  
  - `num_nodes`: Total nodes in the graph.  


#### `compute_node_degrees(edge_index, num_nodes)`
- **Purpose**: Calculates node degrees (number of outgoing edges per node).  
- **NSD Relevance**: Node degrees are used to build degree matrices (`D`), a component of graph Laplacians like those in the NSD paper.  
- **Implementation**: Uses `torch.bincount` for efficient computation.  

---

### **2. `main_runner.py`**
- **Purpose**: Demonstrates module usage by calling `graph_operations` functions.  
- **Workflow**:  
1. Generate graph data (`edge_index`, `num_nodes`).  
2. Compute node degrees from the graph data.  
3. Print results for verification.  

---


---

## Key Takeaways

- **Modular code** separates logic into reusable components (critical for scaling NSD implementations).  
- **`edge_index`** is a fundamental input for graph neural networks (GNNs) like the NSD model.  
- **Node degrees** are foundational for graph theory operations (e.g., Laplacians, normalization).  


