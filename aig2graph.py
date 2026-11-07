import sys
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import networkx as nx
from pyvis.network import Network

@dataclass
class AIGNode:
    id: int
    type: str  # 'input', 'output', 'and', 'const'
    name: str = ''
    inputs: List[Tuple[int, bool]] = None  # (node_id, is_inverted)
    
    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []

def decode_delta(file):
    """Decodes delta-encoded unsigned integers from a file stream."""
    val = 0
    i = 0
    while True:
        byte = file.read(1)
        if not byte:
            raise EOFError("Unexpected end of file while decoding delta.")
        byte_val = ord(byte)
        val |= (byte_val & 0x7f) << (7 * i)
        if (byte_val & 0x80) == 0:
            break
        i += 1
    return val

def parse_aiger(filepath):
    """Parses an AIGER file into a graph structure."""
    nodes: Dict[int, AIGNode] = {}
    
    try:
        with open(filepath, 'rb') as f:
            # Parse header
            header = f.readline().decode('ascii').strip().split()
            if not header[0] == 'aig':
                raise ValueError("Not a valid AIGER file")
            
            M, I, L, O, A = map(int, header[1:])
            
            # Add constant 0 node
            nodes[0] = AIGNode(0, 'const', 'const0')
            
            # Add input nodes
            for i in range(I):
                lit = 2 * (i + 1)
                nodes[lit] = AIGNode(lit, 'input', f'input_{i}')
            
            # Read outputs and create virtual output nodes
            outputs = []
            next_id = 2 * (M + 1)  # Start output node IDs after all other nodes
            for i in range(O):
                line = f.readline().decode('ascii').strip()
                if line:
                    lit = int(line)
                    outputs.append(lit)
                    # Create virtual output node
                    out_id = next_id + 2*i
                    nodes[out_id] = AIGNode(
                        out_id,
                        'output',
                        f'o{lit}',
                        [(lit, bool(lit % 2))]  # Connect to the literal with proper polarity
                    )
            
            # Read AND gates
            current_and = 2 * (I + L + 1)
            for i in range(A):
                try:
                    delta0 = decode_delta(f)
                    delta1 = decode_delta(f)
                    
                    lhs = current_and
                    rhs0 = lhs - delta0
                    rhs1 = rhs0 - delta1
                    
                    # Create AND gate node
                    nodes[lhs] = AIGNode(
                        lhs, 
                        'and',
                        f'and_{i}',
                        [(rhs0, bool(rhs0 % 2)), (rhs1, bool(rhs1 % 2))]
                    )
                    current_and += 2
                except EOFError:
                    break
            
            # Read symbol table
            while True:
                try:
                    line = f.readline().decode('ascii').strip()
                    if not line or line == 'c':
                        break
                    
                    parts = line.split(' ', 2)
                    if len(parts) < 2:
                        continue
                    
                    symbol_type = parts[0][0]
                    idx = int(parts[1])
                    name = parts[2] if len(parts) > 2 else ''
                    
                    if symbol_type == 'i' and 2*(idx+1) in nodes:
                        nodes[2*(idx+1)].name = name
                    elif symbol_type == 'o' and idx < len(outputs):
                        lit = outputs[idx]
                        out_id = 2*(M+1) + 2*idx
                        nodes[out_id] = AIGNode(
                            out_id,
                            'output',
                            name,
                            [(lit, bool(lit % 2))]
                        )
                        
                except (UnicodeDecodeError, ValueError):
                    break
    
    except Exception as e:
        print(f"Error parsing AIG file: {e}")
        return None
    
    return nodes, outputs

def create_visualization(nodes, output_file='aig_visualization.html'):
    """Creates an interactive visualization of the AIG graph in layered circuit style."""
    # Create network with notebook=False for standalone HTML
    net = Network(height='750px', width='100%', directed=True, notebook=False)
    
    # Node colors and shapes
    colors = {
        'input': '#90EE90',   # Light green
        'output': '#ADD8E6',  # Light blue
        'and': '#FFFFFF',     # White for AND gates
        'const': '#D3D3D3'    # Light gray
    }
    
    # All nodes are rectangles
    shapes = {
        'input': 'box',
        'output': 'box',
        'and': 'box',
        'const': 'box'
    }
    
    # Calculate node levels for layered arrangement
    levels = {}
    def calculate_level(node_id):
        if node_id in levels:
            return levels[node_id]
        
        node = nodes.get(node_id)
        if not node or not node.inputs:
            # Inputs and constant nodes are at level 0
            levels[node_id] = 0
            return 0
        
        # Calculate maximum level of inputs and add 1
        max_input_level = 0
        for input_id, _ in node.inputs:
            base_input_id = input_id - (input_id % 2)
            input_level = calculate_level(base_input_id)
            max_input_level = max(max_input_level, input_level)
        
        levels[node_id] = max_input_level + 1
        return levels[node_id]
    
    # Calculate levels for all nodes
    for node_id, node in nodes.items():
        if node_id % 2 == 0:  # Only process non-inverted nodes
            calculate_level(node_id)
    
    # Add nodes with circuit-style shapes and level-based positions
    max_level = max(levels.values()) if levels else 0
    for node_id, node in nodes.items():
        if node_id % 2 == 0:  # Only add non-inverted nodes
            level = levels[node_id]
            # Force input nodes to the leftmost layer (level 0)
            if node.type == 'input':
                level = 0
            # Force output nodes to the rightmost layer
            elif node.type == 'output':
                level = max_level + 1
            
            net.add_node(
                node_id,
                label=str(node_id) if node.type != 'output' else node.name,
                color=colors[node.type],
                shape=shapes[node.type],
                borderWidth=1,
                size=40,
                font={'size': 36, 'color': '#000000'},
                title=f"Type: {node.type}\nName: {node.name}\nID: {node_id}\nLevel: {level}",
                level=level
            )
    
    # Add edges with improved styling
    for node_id, node in nodes.items():
        if node_id % 2 == 0 and node.inputs:  # Only process non-inverted nodes
            for input_id, is_inverted in node.inputs:
                # Get the base (non-inverted) input ID
                base_input_id = input_id - (input_id % 2)
                if base_input_id in nodes:
                    net.add_edge(
                        base_input_id,
                        node_id,
                        color='red' if is_inverted else 'black',
                        width=2,
                        smooth={'type': 'straight'},  # Use straight lines
                        title='Inverted' if is_inverted else 'Direct'
                    )
    
    # Configure physics and layout for layered circuit arrangement
    net.set_options("""
    {
        "physics": {
            "enabled": false
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 300,
                "nodeSpacing": 120,
                "treeSpacing": 200,
                "blockShifting": false,
                "edgeMinimization": true,
                "parentCentralization": true
            }
        },
        "edges": {
            "smooth": {
                "enabled": false,
                "type": "straight"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "nodes": {
            "font": {
                "size": 12,
                "face": "monospace"
            },
            "borderWidth": 1,
            "fixed": {
                "x": false,
                "y": false
            }
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
    }
    """)
    
    try:
        # Save the visualization
        net.write_html(output_file)
        print(f"Layered circuit visualization saved to {output_file}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

def create_aiger_json(nodes, outputs):
    """Creates a JSON representation that can be reconstructed to AIGER format."""
    # Count node types
    input_nodes = [n for n in nodes.values() if n.type == 'input']
    and_nodes = [n for n in nodes.values() if n.type == 'and']
    
    # Process outputs without creating virtual nodes
    output_literals = []
    output_names = []
    for node in nodes.values():
        if node.type == 'output':
            # Store the literal and name
            lit = node.inputs[0][0]
            # if node.inputs[0][1]:  # if inverted
            #     lit ^= 1
            output_literals.append(lit)
            output_names.append(node.name)
    
    aiger_data = {
        'header': {
            'M': max(n.id for n in nodes.values() if n.type != 'output') // 2,
            'I': len(input_nodes),
            'L': 0,  # Latches not supported yet
            'O': len(output_literals),
            'A': len(and_nodes)
        },
        'inputs': [
            {
                'id': node.id,
                'name': node.name
            }
            for node in sorted(input_nodes, key=lambda x: x.id)
        ],
        'outputs': [
            {
                'name': name,
                'literal': lit
            }
            for name, lit in zip(output_names, output_literals)
        ],
        'ands': [
            {
                'id': node.id,
                'name': node.name,
                'inputs': [
                    {'id': input_id, 'inverted': is_inverted}
                    for input_id, is_inverted in node.inputs
                ]
            }
            for node in sorted(and_nodes, key=lambda x: x.id)
        ]
    }
    
    return aiger_data

def save_aiger_json(nodes, outputs, output_file):
    """Saves the AIG graph in a JSON format that preserves AIGER structure."""
    aiger_data = create_aiger_json(nodes, outputs)
    try:
        with open(output_file, 'w') as f:
            json.dump(aiger_data, f, indent=2)
        print(f"AIGER structure saved to {output_file}")
    except Exception as e:
        print(f"Error saving AIGER JSON file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python aig2graph.py <path_to_aig_file>")
        return
    
    aig_file = sys.argv[1]
    if not os.path.exists(aig_file):
        print(f"Error: File {aig_file} not found")
        return
    
    result = parse_aiger(aig_file)
    if result:
        nodes, outputs = result
        # Create visualization
        create_visualization(nodes)
        
        # Save AIGER structure as JSON
        json_output_file = os.path.splitext(aig_file)[0] + '_aiger.json'
        save_aiger_json(nodes, outputs, json_output_file)

if __name__ == "__main__":
    main()

