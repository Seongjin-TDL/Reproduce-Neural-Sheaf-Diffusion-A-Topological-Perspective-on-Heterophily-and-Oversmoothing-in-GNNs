# main_runner.py
# Author: Seongjin-TDL
# Date: 20250610

# PURPOSE:
# This script demonstrates how to import and use a custom module('graph_operations.py'). 
# This is the primary way complex Python projects are structured.
# The NSD codebase uses this pattern extensively, e.g., 'from models import laplacian_builders as lb'.

# We import our custom module and give it a short alias 'gops' for convenience.
# This is a very common practice (e.g., 'import numpy as np').
import graph_operations as gops

def run_demonstration():
    """
    Main function to execute the demonstration of module usage.
    """
    print("--- Starting Day 3 Demonstration: Functions & Modules ---")

    # 1. Call a function from the 'gops' module to get graph data.
    # The function returns two values, which we unpack into two variables.

    graph_edges, total_nodes = gops.get_graph_connectivity()

    print(f"\nReceived from module -> Edge Index:\n{graph_edges}")
    print(f"Received from module -> Total Nodes: {total_nodes}")

    # 2. Pass the data to another function in the same module for calculation.

    node_degs = gops.compute_node_degrees(edge_index=graph_edges, num_nodes=total_nodes)

    print(f"\nFinal result computed via module -> Node Degrees:\n{node_degs}")
    print("\n--- Demonstration Complete ---")
    print("This shows a clear separation of concerns: One file defined the tools, ")
    print("and another uses them. This is key to understanding the NSD code.")

    # This is a standard Python convention. The code inside this block will only run when you execute this file directly(e.g., 'python main_runner.py').
    # It will NOT run if this file is imported by another script.

if __name__ == "__main__":
        run_demonstration()