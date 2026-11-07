import os
import networkx as nx
import matplotlib.pyplot as plt
from biodivine_aeon import *


def visualize_boolean_network(sbml_file_path):
    # Load the Boolean network from SBML file
    bn = BooleanNetwork.from_file(sbml_file_path)

    # Create a directed graph
    G = nx.DiGraph()

    # Initialize node colors
    node_colors = {}
    input_color = "lightgreen"      # Color for input nodes
    output_color = "lightcoral"     # Color for output nodes
    intermediate_color = "white"    # Color for intermediate nodes

    # Add nodes and edges
    for var in bn.variables():
        G.add_node(var)
        predecessors = bn.predecessors(var)
        successors = bn.successors(var)

        # Determine node type
        if len(predecessors) == 0:
            node_colors[var] = input_color      # Input node
        elif len(successors) == 0:
            node_colors[var] = output_color     # Output node
        else:
            node_colors[var] = intermediate_color  # Intermediate node

        # Add edges from predecessors
        for pred in predecessors:
            G.add_edge(pred, var)

    # Draw the graph
    plt.figure(figsize=(12, 8))

    # Use shell layout
    pos = nx.shell_layout(G)

    # Draw nodes with assigned colors
    nx.draw_networkx_nodes(G, pos, node_color=[node_colors[node] for node in G.nodes()], node_size=700,
                           edgecolors='black')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Set title and hide axes
    plt.title("Boolean Network Visualization", fontsize=16)
    plt.axis('off')
    plt.show()


def vis_using_lib():
    import biolqm
    import ginsim

    lqm = biolqm.load("model.sbml")
    lrg = biolqm.to_ginsim(lqm)
    ginsim.show(lrg)


if __name__ == '__main__':
    for file in os.listdir(''):
        if not file.endswith('.sbml'):
            continue
        sbml_file_path = os.path.join('/', file)
        print(f"Visualizing network from file: {file}")
        visualize_boolean_network(sbml_file_path)
