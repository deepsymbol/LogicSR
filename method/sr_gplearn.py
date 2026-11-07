import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from evaluate.data_loader import split_data 
from evaluate.metrics import  calculate_metrics, aggregate_multi_output_metrics  
from evaluate.operator_config import get_method_config
from evaluate.seed_config import get_run_seed

def set_operators(operators):
    config = get_method_config("sr_gplearn")
    config.set_operators(operators, "SR GPlearn")

def setup_gplearn_functions():
    """Setup function set for GPlearn"""
    function_set = []
    config = get_method_config("sr_gplearn")
    operators = config.get_operators()

    if 'and' in operators:
        and_func = make_function(
            function=lambda x, y: np.logical_and(x, y).astype(float), 
            name='and', arity=2)
        function_set.append(and_func)
    
    if 'or' in operators:
        or_func = make_function(
            function=lambda x, y: np.logical_or(x, y).astype(float), 
            name='or', arity=2)
        function_set.append(or_func)
    
    if 'not' in operators:
        not_func = make_function(
            function=lambda x: np.logical_not(x).astype(float), 
            name='not', arity=1)
        function_set.append(not_func)
    
    return function_set


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using GPlearn
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print("GPlearn (Genetic Programming)")
    print("=" * 60)

    config = get_method_config("sr_gplearn")
    expressions = []
    metrics_list = []
    train_pred_columns = []
    test_pred_columns = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)

    for output_idx in range(Y_train.shape[1]):
        y_train = Y_train[:, output_idx]
        y_test = Y_test[:, output_idx]

        print(f" Processing output {output_idx+1}...")

        # Prepare data for gplearn (convert to float and clip to [0,1] range)
        X_train_float = np.clip(X_train.astype(float), 0.0, 1.0)
        y_train_float = np.clip(y_train.astype(float).flatten(), 0.0, 1.0)
        
        # Setup function set and train the model
        function_set = setup_gplearn_functions()
        est = SymbolicRegressor(
            population_size=100,
            generations=50,
            function_set=function_set,
            const_range=None,  # Disable constants
            init_depth=(2, 6),
            init_method='half and half',
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=1.0,
            metric='mse',
            random_state=get_run_seed(),
            verbose=0)
        
        est.fit(X_train_float, y_train_float)
        
        # Extract the best expression from the trained model
        expr = str(est._program)
        import re
        # Convert gplearn variable names (X0,X1,X2...) to LogicSR standard format (x1,x2,x3...)
        expr = re.sub(r'X(\d+)', lambda m: f'x{int(m.group(1)) + 1}', expr)

        # Track which input variables are used in the expression
        for v in range(1, X.shape[1] + 1):
            if f"x{v}" in expr:
                used_vars.add(f"x{v}")

        y_train_pred_raw = est.predict(X_train_float)
        y_test_pred_raw = est.predict(X_test.astype(float))
        
        y_train_pred = (y_train_pred_raw > 0.5).astype(int)
        y_test_pred = (y_test_pred_raw > 0.5).astype(int)

        expressions.append(expr)
        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)

    # If Y_ground_truth is provided, use it for evaluation instead of noisy Y
    if Y_ground_truth is not None:
        _, _, Y_train_gt, Y_test_gt = split_data(X, Y_ground_truth, test_size=1-split, random_state=42)
        aggregated_metrics = aggregate_multi_output_metrics(Y_train_gt, Y_test_gt,
                                                            train_pred_columns,
                                                            test_pred_columns)
        print(f"  Using ground truth for evaluation")
    else:
        aggregated_metrics = aggregate_multi_output_metrics(Y_train, Y_test,
                                                            train_pred_columns,
                                                            test_pred_columns)
    accuracy_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if aggregated_metrics:
        accuracy_tuple = (
            aggregated_metrics['train_bit_acc'],
            aggregated_metrics['test_bit_acc'],
            aggregated_metrics['train_sample_acc'],
            aggregated_metrics['test_sample_acc'],
            aggregated_metrics['train_output_acc'],
            aggregated_metrics['test_output_acc'])
    metrics_list = [accuracy_tuple]
    all_vars_used = all(f'x{i}' in used_vars for i in range(1, X.shape[1] + 1))
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, metrics_list, extra_info