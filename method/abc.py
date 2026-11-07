import subprocess
import tempfile
import os
import numpy as np
from typing import List, Tuple
from evaluate.data_loader import split_data  
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics  
from evaluate.operator_config import get_method_config  
from evaluate.blif_parser import evaluate_blif


def set_operators(operators):
    config = get_method_config("abc")
    config.set_operators(operators, "ABC")


def truth_table_to_pla(X: np.ndarray, Y: np.ndarray) -> str:

    if X.size == 0 or Y.size == 0:
        return ""

    num_inputs = X.shape[1]
    num_outputs = Y.shape[1]

    lines = [f".i {num_inputs}", f".o {num_outputs}"]

    for in_row, out_row in zip(X, Y):
        in_bits = ''.join(map(str, in_row.astype(int)))
        out_bits = ''.join(map(str, out_row.astype(int)))
        lines.append(f"{in_bits} {out_bits}")

    lines.append(".e")
    return '\n'.join(lines)


def run_abc_synthesis_multi_output(X: np.ndarray, Y: np.ndarray) -> Tuple[List[str], int]:

    abc_path = os.path.join(os.path.dirname(__file__), '..', 'external', 'abc', 'abc')
    
    num_outputs = Y.shape[1]
    truth_contents = []
    abc_stdouts = []

    for output_idx in range(num_outputs):
 
        # Create single-output PLA
        Y_single = Y[:, output_idx:output_idx+1]
        pla_content = truth_table_to_pla(X, Y_single)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pla', delete=False) as pla_file:
            pla_file.write(pla_content)
            pla_file_path = pla_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.truth', delete=False) as truth_file:
            truth_file_path = truth_file.name

        # ABC synthesis commands
        # Core commands: read_pla, strash, logic, print_stats, write_blif
        # Optional optimization commands can be added
        abc_commands = [
            f"read_pla {pla_file_path}",
            "strash",  # Convert to AIG
            "logic",   # Ensure network is in AIG format
            "print_stats",  # Get gate count
            f"write_blif {truth_file_path}"  # Export BLIF format
        ]

        cmd = [abc_path, "-c", "; ".join(abc_commands)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise Exception(f"ABC execution failed for output {output_idx}: {result.stderr}")
        
        # Store ABC stdout for gate counting
        abc_stdouts.append(result.stdout)

        # Read BLIF content
        with open(truth_file_path, 'r') as f:
            blif_content = f.read()

        truth_contents.append(blif_content)

        os.unlink(pla_file_path)
        os.unlink(truth_file_path)

    return truth_contents, abc_stdouts


def find_expressions(X, Y, split=0.75):

    print("=" * 60)
    print(" ABC (Logic Synthesis)")
    print("=" * 60)

    expressions = []
    accuracies = []
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)
    
    truth_contents, abc_stdouts = run_abc_synthesis_multi_output(X_train, Y_train)
    
    y_train_pred = np.zeros((len(X_train), Y_train.shape[1]), dtype=int)
    y_test_pred = np.zeros((len(X_test), Y_test.shape[1]), dtype=int)
    
    for output_idx, blif_content in enumerate(truth_contents):
        if blif_content:
            train_pred = evaluate_blif(blif_content, X_train)
            test_pred = evaluate_blif(blif_content, X_test)
            if train_pred.shape[1] > 0:
                y_train_pred[:, output_idx] = train_pred[:, 0]
            if test_pred.shape[1] > 0:
                y_test_pred[:, output_idx] = test_pred[:, 0]
    
    # Generate expressions from BLIF content
    for output_idx in range(Y_train.shape[1]):
        if output_idx < len(truth_contents) and truth_contents[output_idx]:
            blif_lines = truth_contents[output_idx].strip().split('\n')
            clean_lines = []
            for line in blif_lines:
                line = line.strip()
                # Keep only essential BLIF lines, remove comments and model info
                if line and not line.startswith('#') and not line.startswith('.model'):
                    clean_lines.append(line)
            expr = '\n'.join(clean_lines)
        else:
            expr = "0"
        
        expressions.append(expr)
    aggregated_metrics = aggregate_multi_output_metrics(Y_train, Y_test,
                                                        y_train_pred,
                                                        y_test_pred)
    accuracy_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if aggregated_metrics:
        accuracy_tuple = (
            aggregated_metrics['train_bit_acc'],
            aggregated_metrics['test_bit_acc'],
            aggregated_metrics['train_sample_acc'],
            aggregated_metrics['test_sample_acc'],
            aggregated_metrics['train_output_acc'],
            aggregated_metrics['test_output_acc'])
    accuracies = [accuracy_tuple]

    # Store abc_stdouts globally for experiment_runner to access
    global _abc_stdouts
    _abc_stdouts = abc_stdouts
    
    extra_info = {
        'all_vars_used': True,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, accuracies, extra_info


def get_abc_stdouts():
    """Get ABC stdout information for gate counting"""
    global _abc_stdouts
    return _abc_stdouts if '_abc_stdouts' in globals() else None