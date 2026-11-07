import sys
import os
import torch
from evaluate.data_loader import split_data
from evaluate.metrics import (evaluate_expression, calculate_metrics,
                              aggregate_multi_output_metrics)
from evaluate.operator_config import get_method_config

# Import boolformer from external directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'boolformer'))
from boolformer import load_boolformer


def set_operators(operators):
    config = get_method_config("nn_boolformer")
    config.set_operators(operators, "BoolFormer")


def get_boolformer_expression(tree, allowed_ops):
    """Convert BoolFormer tree to expression"""
    expr = str(tree)
    # Convert variable names x_0 -> x1, x_1 -> x2, etc.
    import re
    expr = re.sub(r'x_(\d+)', lambda m: f'x{int(m.group(1)) + 1}', expr)
    return expr


def train_boolformer_model(X, Y, noise=0):
    if noise == 0:
        model = load_boolformer('noiseless')
    else:
        model = load_boolformer('noisy')
    
    # Force-enable GPU (if available)
    if torch.cuda.is_available() and model.env and hasattr(model.env, 'params'):
        model.env.params.cpu = False
        # Move model modules to GPU
        if hasattr(model, 'embedder') and model.embedder is not None:
            model.embedder = model.embedder.cuda()
        if hasattr(model, 'encoder') and model.encoder is not None:
            model.encoder = model.encoder.cuda()
        if hasattr(model, 'decoder') and model.decoder is not None:
            model.decoder = model.decoder.cuda()
    
    return model


def generate_boolformer_expressions(model, X, Y, allowed_ops):
    num_outputs = Y.shape[1] if len(Y.shape) > 1 else 1
    inputs = [X] * num_outputs
    outputs = [Y[:, i]
               for i in range(num_outputs)] if num_outputs > 1 else [Y]
    
    pred_trees, errors, complexities = model.fit(inputs,
                                                 outputs,
                                                 verbose=False,
                                                 beam_size=1,
                                                 beam_type="search")
    
    expressions = []
    for i, pred_tree in enumerate(pred_trees):
        expr = get_boolformer_expression(pred_tree, allowed_ops)
        expressions.append(expr)
    return expressions


def find_expressions(X, Y, split=0.75, noise=0, allowed_operators=None):
    
    print("=" * 60)
    print("BoolFormer (Neural Network)")
    print("=" * 60)
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)
    # Fit on the training rows only, like every other method
    model = train_boolformer_model(X_train, Y_train, noise)
    expressions = generate_boolformer_expressions(model, X_train, Y_train,
                                                  allowed_operators)
    metrics_list = []
    train_pred_columns = []
    test_pred_columns = []

    for i, expr in enumerate(expressions):

        y_train_pred = evaluate_expression(expr, X_train)
        y_test_pred = evaluate_expression(expr, X_test)
        y_train_true = Y_train[:,
                               i] if Y_train.shape[1] > 1 else Y_train
        y_test_true = Y_test[:, i] if Y_test.shape[1] > 1 else Y_test
        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)
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
    extra_info = {
        'all_vars_used': True,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, metrics_list, extra_info