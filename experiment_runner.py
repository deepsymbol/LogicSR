import os
import time
import json
import re
import numpy as np
import importlib
from typing import List, Dict, Optional
from evaluate.data_loader import load_truth_file, split_data
from evaluate.gate_counter import count_gates
from evaluate.metrics import aggregate_multi_output_metrics, evaluate_expression
from evaluate.blif_parser import evaluate_blif
from evaluate.seed_config import (DEFAULT_SEED, result_suffix, seed_global_rngs,
                                  set_run_seed)
import ast


def load_method(method_name: str):
    module = importlib.import_module(
        f'method.{method_name.replace(" ", "_")}')
    return module if hasattr(module, 'find_expressions') else None


def result_path(output_dir: str, method_name: str, data_file: str,
                seed: Optional[int] = None) -> str:
    """Where one (method, instance, seed) result is stored.

    The default seed keeps the historical '<method>.json' name so results
    produced before seeds were configurable remain readable.
    """
    rel_path = os.path.relpath(data_file, start='data')
    output_subdir = os.path.join(output_dir, os.path.splitext(rel_path)[0])
    return os.path.join(output_subdir,
                        f"{method_name}{result_suffix(seed)}.json")


def save_single_result(
    results: Dict,
    output_dir: str,
    method_name: str,
    data_file: str,
    file_index: int,
    seed: Optional[int] = None
):
    """Save single experiment result to JSON file"""
    output_path = result_path(output_dir, method_name, data_file, seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = {
        'data_file': data_file,
        'expression': results['expressions'][file_index],
        'train_bit_acc': float(results['train_accuracy'][file_index]),
        'test_bit_acc': float(results['test_accuracy'][file_index]),
        'train_sample_acc': float(results['train_sample_acc'][file_index]),
        'test_sample_acc': float(results['test_sample_acc'][file_index]),
        'train_output_acc': float(results['train_output_acc'][file_index]),
        'test_output_acc': float(results['test_output_acc'][file_index]),
        'num_gates': float(results['num_gates'][file_index]) if not np.isnan(results['num_gates'][file_index]) else None,
        'execution_time': float(results['execution_time'][file_index]),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'progress': f"{file_index + 1}/{len(results['expressions'])}"
    }
    
    # Add flag if using ground truth evaluation
    if 'use_ground_truth' in results and results['use_ground_truth']:
        result['evaluated_with_ground_truth'] = True
    
    # Add LLM metrics if available (for GPT methods)
    # Token usage can be used to calculate costs based on current API pricing
    if 'llm_metrics' in results and file_index < len(results['llm_metrics']):
        llm_metrics = results['llm_metrics'][file_index]
        if llm_metrics:
            result['llm_metrics'] = {
                'prompt_tokens': int(llm_metrics.get('prompt_tokens', 0)),
                'completion_tokens': int(llm_metrics.get('completion_tokens', 0)),
                'total_tokens': int(llm_metrics.get('total_tokens', 0)),
                'api_response_time_s': float(llm_metrics.get('api_response_time_s', 0.0))
            }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f" Results saved to: {output_path}")


def check_result_exists(output_dir: str, method_name: str, data_file: str,
                        seed: Optional[int] = None) -> bool:
    """Check if result file already exists"""
    return os.path.exists(result_path(output_dir, method_name, data_file, seed))


def load_existing_result(output_dir: str, method_name: str, data_file: str,
                         seed: Optional[int] = None) -> Optional[Dict]:
    """Load existing result file"""
    output_path = result_path(output_dir, method_name, data_file, seed)

    try:
        with open(output_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_ground_truth_file(noisy_file_path: str) -> Optional[str]:
    """Find corresponding ground truth file path (replace noise_X with the noiseless level)

    Args:
        noisy_file_path: Noisy file path, e.g., data/ANO/in5_out5/and10/noise_0.05/1.truth
    Returns:
        Ground truth file path, e.g., data/ANO/in5_out5/and10/noise_0/1.truth
        Returns None if not found or path doesn't contain noise
    """
    # Check if path contains noise
    if 'noise_' not in noisy_file_path:
        return None

    # Replace noise_X with the noiseless level. ANO/AN spell it noise_0 while
    # BBM spells it noise_00, so both are tried before giving up.
    for noiseless in ('noise_0', 'noise_00'):
        ground_truth_path = re.sub(r'noise_[0-9.]+', noiseless, noisy_file_path)
        if os.path.exists(ground_truth_path):
            return ground_truth_path

    return None


def reevaluate_with_ground_truth(expressions, X, Y_noisy, Y_ground_truth, split=0.75, method_name=None):
    """Re-evaluate expressions using ground truth
    
    Args:
        expressions: List of learned expressions
        X: Input data (loaded from noisy file)
        Y_noisy: Noisy output data (used for data splitting)
        Y_ground_truth: Ground truth output data
        split: Train/test split ratio
        method_name: Method name, used to determine how to evaluate expressions
    Returns:
        Dictionary of accuracy metrics calculated using ground truth
    """
    # Split data using noisy data (maintaining the same split)
    # Use the same random_state to ensure consistent splitting
    X_train, X_test, _, _ = split_data(X, Y_noisy, test_size=1-split, random_state=42)
    
    # Use ground truth Y values with the same split
    # Since split_data uses the same random_state, the split should be consistent
    _, _, Y_train_gt, Y_test_gt = split_data(X, Y_ground_truth, test_size=1-split, random_state=42)
    
    # Evaluate each expression
    train_pred_columns = []
    test_pred_columns = []
    
    for expr in expressions:
        expr_str = str(expr) if not isinstance(expr, str) else expr
        
        # Select evaluation function based on method type
        if method_name == 'abc':
            # ABC returns BLIF format
            train_pred = evaluate_blif(expr_str, X_train)
            test_pred = evaluate_blif(expr_str, X_test)
            if train_pred.shape[1] > 0:
                train_pred = train_pred[:, 0]
            else:
                train_pred = np.zeros(len(X_train), dtype=int)
            if test_pred.shape[1] > 0:
                test_pred = test_pred[:, 0]
            else:
                test_pred = np.zeros(len(X_test), dtype=int)
        elif method_name == 'qm':
            # QM returns set string, e.g., "{'1--', '0--'}"
            from method.qm import evaluate_qm_raw_output
            try:
                # Try to parse set string
                qm_output = ast.literal_eval(expr_str)
                if isinstance(qm_output, set):
                    train_pred = evaluate_qm_raw_output(qm_output, X_train)
                    test_pred = evaluate_qm_raw_output(qm_output, X_test)
                else:
                    train_pred = np.zeros(len(X_train), dtype=int)
                    test_pred = np.zeros(len(X_test), dtype=int)
            except:
                train_pred = np.zeros(len(X_train), dtype=int)
                test_pred = np.zeros(len(X_test), dtype=int)
        elif method_name == 'bdd':
            # BDD returns cube list string, e.g., "[{'x1': True}, {'x2': False}]"
            from method.bdd import evaluate_bdd_raw_output
            try:
                bdd_output = ast.literal_eval(expr_str)
                if isinstance(bdd_output, list):
                    train_pred = evaluate_bdd_raw_output(bdd_output, X_train)
                    test_pred = evaluate_bdd_raw_output(bdd_output, X_test)
                else:
                    train_pred = np.zeros(len(X_train), dtype=int)
                    test_pred = np.zeros(len(X_test), dtype=int)
            except:
                train_pred = np.zeros(len(X_train), dtype=int)
                test_pred = np.zeros(len(X_test), dtype=int)
        else:
            # Other methods use standard expression evaluation
            train_pred = evaluate_expression(expr_str, X_train)
            test_pred = evaluate_expression(expr_str, X_test)
        
        # Ensure it's a 1D array
        if train_pred.ndim > 1:
            train_pred = train_pred.flatten()
        if test_pred.ndim > 1:
            test_pred = test_pred.flatten()
        
        train_pred_columns.append(train_pred)
        test_pred_columns.append(test_pred)
    
    # Merge multi-output predictions
    if len(train_pred_columns) > 0:
        y_train_pred = np.column_stack(train_pred_columns) if len(train_pred_columns) > 1 else train_pred_columns[0].reshape(-1, 1)
        y_test_pred = np.column_stack(test_pred_columns) if len(test_pred_columns) > 1 else test_pred_columns[0].reshape(-1, 1)
    else:
        y_train_pred = np.zeros((len(X_train), Y_train_gt.shape[1]))
        y_test_pred = np.zeros((len(X_test), Y_test_gt.shape[1]))
    
    # Ensure dimension matching
    if y_train_pred.ndim == 1:
        y_train_pred = y_train_pred.reshape(-1, 1)
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape(-1, 1)
    
    # Calculate metrics
    aggregated_metrics = aggregate_multi_output_metrics(
        Y_train_gt, Y_test_gt, y_train_pred, y_test_pred
    )
    
    return aggregated_metrics


def run_experiment(
    method_module,
    data_files: List[str],
    split: float,
    output_dir: str,
    method_name: str,
    input_size: Optional[int] = None,
    output_size: Optional[int] = None,
    operators: Optional[List[str]] = None,
    skip_existing: bool = False,
    use_ground_truth: bool = False,
    seed: int = DEFAULT_SEED
) -> Dict:
    """Run experiment with immediate result saving and optional skip existing results"""

    # Every stochastic method draws from the global RNGs unless it exposes its
    # own seed argument, so fixing them here is what makes a seeded run
    # reproducible without touching each method.
    set_run_seed(seed, method_name)

    # Initialize results structure
    results = {
        'expressions': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'num_gates': [],
        'execution_time': [],
        'train_sample_acc': [],
        'test_sample_acc': [],
        'train_output_acc': [],
        'test_output_acc': [],
        'llm_metrics': [],  # For LLM methods (GPT-4o, GPT-o3)
        'use_ground_truth': use_ground_truth  # Flag for ground truth evaluation
    }
    
    # Initialize result lists for each data file
    for _ in data_files:
        results['expressions'].append([])
        results['train_accuracy'].append(0.0)
        results['test_accuracy'].append(0.0)
        results['num_gates'].append(np.nan)
        results['execution_time'].append(0.0)
        results['train_sample_acc'].append(False)
        results['test_sample_acc'].append(False)
        results['train_output_acc'].append(False)
        results['test_output_acc'].append(False)
        results['llm_metrics'].append(None)

    def _handle_error_case(i, test_file, start_time, error_msg="Method error"):
        """Handle error cases with consistent result setting"""
        print(f"  {error_msg} for {test_file}")
        results['expressions'][i] = ["FAIL"]
        results['train_accuracy'][i] = 0.0
        results['test_accuracy'][i] = 0.0
        results['num_gates'][i] = np.nan
        results['execution_time'][i] = time.time() - start_time
        results['train_sample_acc'][i] = 0.0
        results['test_sample_acc'][i] = 0.0
        results['train_output_acc'][i] = 0.0
        results['test_output_acc'][i] = 0.0
        save_single_result(results, output_dir, method_name, test_file, i, seed)

    # Count skipped and files to run
    skipped_count = 0
    total_count = len(data_files)
    
    for i, test_file in enumerate(data_files):
        start_time = time.time()
        
        # Check if skipping existing results
        if skip_existing and check_result_exists(output_dir, method_name, test_file, seed):
            existing_result = load_existing_result(output_dir, method_name, test_file, seed)
            if existing_result:
                print(f"  ⏭  Skipping existing result: {test_file}")
                print(f"     Existing result: {existing_result.get('expression', 'N/A')}")
                print(f"     Test accuracy: {existing_result.get('test_bit_acc', 'N/A')}")
                
                # Restore data from existing results
                results['expressions'][i] = existing_result.get('expression', ["SKIPPED"])
                results['train_accuracy'][i] = existing_result.get('train_bit_acc', 0.0)
                results['test_accuracy'][i] = existing_result.get('test_bit_acc', 0.0)
                results['num_gates'][i] = existing_result.get('num_gates', np.nan)
                results['execution_time'][i] = existing_result.get('execution_time', 0.0)
                results['train_sample_acc'][i] = existing_result.get('train_sample_acc', 0.0)
                results['test_sample_acc'][i] = existing_result.get('test_sample_acc', 0.0)
                results['train_output_acc'][i] = existing_result.get('train_output_acc', 0.0)
                results['test_output_acc'][i] = existing_result.get('test_output_acc', 0.0)
                
                skipped_count += 1
                continue
        
        X, Y = load_truth_file(test_file)
        
        # If ground truth evaluation is enabled, load ground truth data
        Y_ground_truth = None
        if use_ground_truth:
            ground_truth_file = find_ground_truth_file(test_file)
            if ground_truth_file:
                print(f"  Using ground truth file for evaluation: {ground_truth_file}")
                _, Y_ground_truth = load_truth_file(ground_truth_file)
            elif 'noise_' in test_file:
                print(f"  Warning: Ground truth file not found, using original data for evaluation")

        # Call method with appropriate parameters
        neural_network_methods = ['nn_neuralut', 'satnet', 'nn_difflogic', 'nn_ai4eda']
        symbolic_regression_methods = ['sr_gplearn', 'sr_deap', 'sr_tengp', 'sr_operon', 'sr_pysr']
        llm_methods = ['gpt_4o', 'gpt_o3', 'gpt_o3_pot', 'gpt_4o_pot']  # LLM methods that may have expression evaluation issues
        
        if getattr(method_module, '__name__', '').endswith('nn_boolformer'):
            # Extract noise from path for nn_boolformer
            m = re.search(r'noise_([0-9.]+)', test_file)
            noise = float(m.group(1)) if m else 0.0
            result = method_module.find_expressions(X, Y, split=split, noise=noise, allowed_operators=operators)
        elif (method_name in neural_network_methods or method_name in symbolic_regression_methods or method_name in llm_methods) and use_ground_truth and Y_ground_truth is not None:
            # For neural network, symbolic regression, and LLM methods, pass Y_ground_truth directly
            # They will calculate metrics internally using model/expression predictions vs ground truth
            result = method_module.find_expressions(X, Y, split=split, Y_ground_truth=Y_ground_truth)
        else:
            result = method_module.find_expressions(X, Y, split=split)
            
        if result is None:
            _handle_error_case(i, test_file, start_time)
            continue

        # Handle return values: (expressions, accuracies, extra_info)
        if len(result) >= 3:
            expressions, accuracies, extra_info = result[0], result[1], result[2]
        else:
            expressions, accuracies = result[0], result[1]
            extra_info = None

        aggregated_metrics = None
        if extra_info and isinstance(extra_info, dict):
            aggregated_metrics = extra_info.get('aggregated_metrics')
        
        # If ground truth evaluation is enabled, recalculate accuracy using ground truth
        if use_ground_truth and Y_ground_truth is not None:
            if method_name in neural_network_methods or method_name in symbolic_regression_methods or method_name in llm_methods:
                # For neural network, symbolic regression, and LLM methods, 
                # aggregated_metrics should already be calculated using ground truth
                if aggregated_metrics:
                    print(f"  Using ground truth metrics from model/expression predictions")
                    accuracy_tuple = (
                        aggregated_metrics['train_bit_acc'],
                        aggregated_metrics['test_bit_acc'],
                        aggregated_metrics['train_sample_acc'],
                        aggregated_metrics['test_sample_acc'],
                        aggregated_metrics['train_output_acc'],
                        aggregated_metrics['test_output_acc']
                    )
                    accuracies = [accuracy_tuple]
            else:
                # For other methods, use expression evaluation
                print(f"  Re-evaluating accuracy using ground truth...")
                gt_metrics = reevaluate_with_ground_truth(expressions, X, Y, Y_ground_truth, split=split, method_name=method_name)
                if gt_metrics:
                    aggregated_metrics = gt_metrics
                    # Update accuracy_tuple
                    accuracy_tuple = (
                        gt_metrics['train_bit_acc'],
                        gt_metrics['test_bit_acc'],
                        gt_metrics['train_sample_acc'],
                        gt_metrics['test_sample_acc'],
                        gt_metrics['train_output_acc'],
                        gt_metrics['test_output_acc']
                    )
                    accuracies = [accuracy_tuple]

        # Detect all "0" result case (applies to all methods)
        if expressions and len(expressions) > 0 and all(str(expr) == "0" for expr in expressions):
            print("  Detected all '0' results, terminate subsequent tests at current scale and larger scales")
            print(f"   Debug: expressions = {expressions}")
            print(f"   File: {test_file}")
            _handle_error_case(i, test_file, start_time, "All '0' results detected")

        # Filter valid expressions
        valid_expr = [e for e in expressions if e not in [None, "FAIL", 0]]

        # Count gates with special handling for ABC method and SATNet
        if method_name == 'abc' and hasattr(method_module, 'get_abc_stdouts'):
            abc_stdouts = method_module.get_abc_stdouts()
            num_gates = count_gates(valid_expr, X.shape[1], abc_stdouts, extra_info)
        else:
            num_gates = count_gates(valid_expr, X.shape[1], None, extra_info)
            
        # Store results
        results['expressions'][i] = expressions
        results['train_accuracy'][i] = np.mean([acc[0] for acc in accuracies])
        results['test_accuracy'][i] = np.mean([acc[1] for acc in accuracies])
        results['num_gates'][i] = num_gates
        results['execution_time'][i] = time.time() - start_time
        results['train_sample_acc'][i] = np.mean([acc[2] if len(acc) > 2 else 0.0 for acc in accuracies])
        results['test_sample_acc'][i] = np.mean([acc[3] if len(acc) > 3 else 0.0 for acc in accuracies])
        results['train_output_acc'][i] = np.mean([acc[4] if len(acc) > 4 else 0.0 for acc in accuracies])
        results['test_output_acc'][i] = np.mean([acc[5] if len(acc) > 5 else 0.0 for acc in accuracies])

        if aggregated_metrics:
            results['train_accuracy'][i] = aggregated_metrics.get('train_bit_acc', results['train_accuracy'][i])
            results['test_accuracy'][i] = aggregated_metrics.get('test_bit_acc', results['test_accuracy'][i])
            results['train_sample_acc'][i] = aggregated_metrics.get('train_sample_acc', results['train_sample_acc'][i])
            results['test_sample_acc'][i] = aggregated_metrics.get('test_sample_acc', results['test_sample_acc'][i])
            results['train_output_acc'][i] = aggregated_metrics.get('train_output_acc', results['train_output_acc'][i])
            results['test_output_acc'][i] = aggregated_metrics.get('test_output_acc', results['test_output_acc'][i])
        
        # Store LLM metrics if available (for GPT methods)
        if extra_info and isinstance(extra_info, dict) and 'llm_metrics' in extra_info:
            results['llm_metrics'][i] = extra_info['llm_metrics']
        # Print results
        print(f"  File: {test_file}")
        
        # Display expression
        expr_display = expressions[0] if len(expressions) == 1 else expressions
        print(f"  Expression: {expr_display}")
            
        print(f"  Train Bit Acc: {results['train_accuracy'][i]:.3f}, Test Bit Acc: {results['test_accuracy'][i]:.3f}")
        print(f"  Train Sample Acc: {results['train_sample_acc'][i]:.3f}, Test Sample Acc: {results['test_sample_acc'][i]:.3f}")
        print(f"  Train Output Acc: {results['train_output_acc'][i]:.3f}, Test Output Acc: {results['test_output_acc'][i]:.3f}\n")
        
        # Save results
        save_single_result(results, output_dir, method_name, test_file, i, seed)

    # Output statistics
    if skip_existing:
        print(f"\n Experiment Statistics:")
        print(f"   Total files: {total_count}")
        print(f"   Skipped files: {skipped_count}")
        print(f"   Files actually run: {total_count - skipped_count}")
        if skipped_count > 0:
            print(f"   Skip rate: {skipped_count/total_count*100:.1f}%")

    return results
