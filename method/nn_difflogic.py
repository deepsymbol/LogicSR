import numpy as np
import os
import sys
import datetime
import torch
import torch.nn as nn
from evaluate.data_loader import split_data  
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics  

# Add difflogic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'difflogic'))

# Import difflogic
from difflogic import LogicLayer, GroupSum
from difflogic.operator_config import set_operator_set
print(" DiffLogic library loaded successfully")


def set_operators(operators):
    """
    Set operators for DiffLogic based on command line -p parameter.
    Maps operator names (and, or, not) to DiffLogic operator sets.
    
    Args:
        operators: List of operator names from -p parameter, e.g., ['and', 'or', 'not']
    """
    if not operators:
        # Default to extended set if no operators specified
        set_operator_set("extended")
        return
    
    operators_set = set(operators)
    
    # Map command line operators to DiffLogic operator sets
    # minimal: only and and not
    # basic: and, or, and not
    # extended: all 16 operators
    
    has_and = 'and' in operators_set
    has_or = 'or' in operators_set
    has_not = 'not' in operators_set
    
    if has_and and has_or and has_not:
        # Has all three basic operators -> use basic set
        set_operator_set("basic")
    elif has_and and has_not and not has_or:
        # Only and and not -> use minimal set
        set_operator_set("minimal")
    else:
        # Default to extended set for other combinations
        set_operator_set("extended")


def get_config(input_size: int, output_size: int):
    """Dynamic DiffLogic configuration for logic synthesis"""
    if input_size <= 5:
        k, l = max(256, input_size * 40), 3
    elif input_size <= 10:
        k, l = max(512, input_size * 50), 3
    elif input_size <= 20:
        k, l = max(1024, input_size * 60), 4
    elif input_size <= 40:
        k, l = max(2048, input_size * 80), 4
    else:  # input_size > 40
        k, l = max(4096, input_size * 100), 5
    
    # Ensure the last layer output is divisible by output_size
    # GroupSum requires: output_dim % output_size == 0
    k = min(k, 50000)
    if output_size > 1:
        k = ((k // output_size) + 1) * output_size
    
    return k, l


def create_difflogic_model(input_size: int, output_size: int, device: str = 'cuda'):
    """Create standard difflogic model using original architecture"""
    k, l = get_config(input_size, output_size)
    
    logic_layers = []
    logic_layers.append(torch.nn.Flatten())

    # First layer
    logic_layers.append(LogicLayer(in_dim=input_size, out_dim=k, device=device))
    
    # Hidden layers
    for _ in range(l - 1):
        logic_layers.append(LogicLayer(in_dim=k, out_dim=k, device=device))
    
    # Output layer
    model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(k=output_size, tau=10.0, device=device)
    )
    
    return model


def train_difflogic_model(X_train: np.ndarray, Y_train: np.ndarray, input_size: int, output_size: int, device: str = 'cuda'):
    """Train difflogic model using standard approach"""
    
    # Create model
    model = create_difflogic_model(input_size, output_size, device)
    model = model.to(device)
    
    # Create log directory and file
    log_dir = "./logs_base"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"difflogic_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    
    # Convert data to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32, device=device)
    
    # difflogic training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            # Calculate accuracies
            with torch.no_grad():
                preds = (torch.sigmoid(output) > 0.5).float()
                bit_acc = (preds == Y_tensor).float().mean().item()
                sample_acc = ((preds == Y_tensor).all(dim=1).float().mean().item())
            
            log_line = (f"Epoch {epoch:04d} | Loss={loss.item():.4f} "
                        f"| BitAcc={bit_acc:.3f} | SampleAcc={sample_acc:.3f}")
            print(f"   {log_line}")
            log_file.write(log_line + "\n")
            log_file.flush()
    
    log_file.close()
    return model


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using DiffLogic
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print(" DiffLogic (Neural Network)")
    print("=" * 60)

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)
    num_inputs = X.shape[1]
    num_outputs = Y.shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_difflogic_model(X_train, Y_train, num_inputs, num_outputs, device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        Y_pred_train = (model(X_train_tensor) > 0).cpu().numpy().astype(int)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        Y_pred_test = (model(X_test_tensor) > 0).cpu().numpy().astype(int)

    # If Y_ground_truth is provided, use it for evaluation instead of noisy Y
    if Y_ground_truth is not None:
        _, _, Y_train_gt, Y_test_gt = split_data(X, Y_ground_truth, test_size=1-split, random_state=42)
        aggregated_metrics = aggregate_multi_output_metrics(Y_train_gt, Y_test_gt,
                                                            Y_pred_train,
                                                            Y_pred_test)
        print(f"  Using ground truth for evaluation")
    else:
        aggregated_metrics = aggregate_multi_output_metrics(Y_train, Y_test,
                                                            Y_pred_train,
                                                            Y_pred_test)
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

    # Calculate complexity based on model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    expressions = ["NEURAL_NETWORK_DiffLogic"] * num_outputs

    all_vars_used = False
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, accuracies, extra_info
