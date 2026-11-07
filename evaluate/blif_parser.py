import numpy as np
from typing import Dict


def parse_blif(blif_content: str) -> Dict[str, Dict]:
    if not blif_content:
        return {}
    
    functions = {}
    current_function = None
    
    lines = blif_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.startswith('.names'):
            parts = line.split()
            if len(parts) >= 2:
                inputs = parts[1:-1]  # All except .names and output name are inputs
                output = parts[-1]
                
                current_function = {
                    'inputs': inputs,
                    'truth_table': [0] * (2 ** len(inputs))
                }
                functions[output] = current_function
                
        elif line.startswith('.end'):
            current_function = None
            
        elif current_function is not None and line and not line.startswith('.'):
            # Parse truth table row
            parts = line.split()
            if len(parts) >= 1:
                if len(parts) == 1:
                    # Constant output (single value, no input pattern)
                    output_value = int(parts[0])
                    current_function['truth_table'] = [output_value] * len(current_function['truth_table'])
                elif len(parts) >= 2:
                    input_pattern = parts[0]
                    output_value = int(parts[1])
                    
                    # Convert input pattern to truth table index
                    if len(input_pattern) == len(current_function['inputs']):
                        index = sum(int(bit) << (len(input_pattern) - 1 - i) 
                                  for i, bit in enumerate(input_pattern))
                        if 0 <= index < len(current_function['truth_table']):
                            current_function['truth_table'][index] = output_value
                        
                # Count truth table entries
                if 'num_entries' not in current_function:
                    current_function['num_entries'] = 0
                current_function['num_entries'] += 1
                        

    for func_name, func in functions.items():
        if func.get('num_entries', 0) == 1:
            # Find the specified index by re-parsing
            current_func_name = None
            for line in lines:
                line = line.strip()
                if line.startswith('.names'):
                    current_func_name = line.split()[-1] if len(line.split()) >= 2 else None
                elif current_func_name == func_name and line and not line.startswith('.'):
                    parts = line.split()
                    if len(parts) >= 2:
                        input_pattern, output_value = parts[0], int(parts[1])
                        
                        if len(input_pattern) == len(func['inputs']):
                            specified_index = sum(int(bit) << (len(input_pattern) - 1 - i) 
                                                for i, bit in enumerate(input_pattern))
                            
                            if output_value == 0:
                                func['truth_table'] = [1 if i != specified_index else 0 
                                                     for i in range(len(func['truth_table']))]
                            break
    
    return functions


def evaluate_blif_functions(functions: Dict[str, Dict], X: np.ndarray) -> np.ndarray:
    if not functions:
        return np.zeros((len(X), 1), dtype=int)
    
    # Find output node 
    output_node = None
    for node_name in functions:
        is_output = True
        for other_name, other_func in functions.items():
            if node_name in other_func['inputs']:
                is_output = False
                break
        if is_output:
            output_node = node_name
            break
    
    if not output_node:
        output_node = list(functions.keys())[-1]
    
    results = np.zeros((len(X), 1), dtype=int)
    
    for sample_idx in range(len(X)):
        x = X[sample_idx]
        
        # Create variable mapping
        var_values = {f'x{i}': int(x[i]) for i in range(len(x))}
        
        # Calculate all nodes in topological order
        node_values = var_values.copy()
        
        # Calculate intermediate nodes
        for node_name, func in functions.items():
            if node_name == output_node:
                continue
            node_values[node_name] = evaluate_single_function(func, node_values, x)
        
        # Calculate output node
        if output_node in functions:
            results[sample_idx, 0] = evaluate_single_function(functions[output_node], node_values, x)
    
    return results


def evaluate_single_function(func: Dict, node_values: Dict[str, int], x: np.ndarray) -> int:
    inputs = func['inputs']
    truth_table = func['truth_table']
    
    # Build input pattern
    input_pattern = ''
    for input_name in inputs:
        if input_name in node_values:
            input_pattern += str(node_values[input_name])
        elif input_name.startswith('x'):
            try:
                idx = int(input_name[1:])
                input_pattern += str(int(x[idx])) if 0 <= idx < len(x) else '0'
            except:
                input_pattern += '0'
        else:
            input_pattern += '0'
    
    # Convert input pattern to truth table index
    if len(input_pattern) == len(inputs):
        index = sum(int(bit) << (len(input_pattern) - 1 - i) 
                   for i, bit in enumerate(input_pattern))
        return truth_table[index] if 0 <= index < len(truth_table) else 0
    
    return 0


def evaluate_blif(blif_content: str, X: np.ndarray) -> np.ndarray:
    functions = parse_blif(blif_content)
    return evaluate_blif_functions(functions, X)
