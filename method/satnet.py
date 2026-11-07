import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from evaluate.data_loader import split_data
from evaluate.metrics import aggregate_multi_output_metrics

# Import SATNet from external directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'SATNet'))
import satnet


def train_satnet_model(X_train: np.ndarray, Y_train: np.ndarray, 
                       num_inputs: int, num_outputs: int, 
                       device: str = 'cuda', 
                       m: int = None, aux: int = None,
                       n_epochs: int = 100,
                       lr: float = 0.1,
                       batch_size: int = 100):
    """
    Train a single SATNet model to handle all outputs simultaneously
    (Following the Sudoku example pattern)
    
    Args:
        X_train: Training input data (n_samples, n_inputs)
        Y_train: Training output data (n_samples, n_outputs)
        num_inputs: Number of input variables
        num_outputs: Number of output variables
        device: 'cuda' or 'cpu'
        m: Rank of clause matrix (default: 4 * (num_inputs + num_outputs))
        aux: Number of auxiliary variables (default: num_inputs + num_outputs)
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        
    Returns:
        Trained SATNet model
    """
    # Total number of variables: inputs + outputs
    n = num_inputs + num_outputs
    
    if m is None:
        m = max(4, 4 * n)
        # Ensure m is multiple of 4 for CPU compatibility
        m = ((m + 3) // 4) * 4
    
    if aux is None:
        aux = max(0, n)
    
    # Create a single SATNet model for all outputs
    model = satnet.SATNet(n, m, aux=aux, max_iter=40, eps=1e-4, prox_lam=1e-2)
    model = model.to(device)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32, device=device)
    
    # Prepare input: concatenate X and Y (Y initialized as 0 for unknown outputs)
    # Following Sudoku pattern: z contains all variables (inputs + outputs)
    Y_initial = torch.zeros_like(Y_tensor)  # Unknown outputs initialized as 0
    z = torch.cat([X_tensor, Y_initial], dim=1)  # (n_samples, num_inputs + num_outputs)
    
    # is_input: 1 for X (known inputs), 0 for Y (unknown outputs)
    is_input = torch.cat([
        torch.ones(X_tensor.shape[0], num_inputs, dtype=torch.int32, device=device),
        torch.zeros(Y_tensor.shape[0], num_outputs, dtype=torch.int32, device=device)
    ], dim=1)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    model.train()
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Mini-batch training
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_z = z[start_idx:end_idx]
            batch_is_input = is_input[start_idx:end_idx]
            batch_Y = Y_tensor[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass: model predicts all variables, but only outputs are updated
            z_pred = model(batch_z, batch_is_input)
            Y_pred = z_pred[:, num_inputs:]  # Extract output part (all outputs)
            
            # Loss: binary cross entropy on all output variables
            # Note: SATNet outputs are already in [0,1] range, but clamp ensures numerical stability
            loss = criterion(torch.clamp(Y_pred, 0, 1), batch_Y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using SATNet (single model for all outputs)
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print(" SATNet (Differentiable MAXSAT Solver)")
    print("=" * 60)
    
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)
    num_inputs = X.shape[1]
    num_outputs = Y.shape[1]
    
    # Determine device: if CUDA_VISIBLE_DEVICES is set, PyTorch will see it as cuda:0
    # Otherwise, use the default cuda:0 (which is the first available GPU)
    if torch.cuda.is_available():
        device = 'cuda:0'  # PyTorch will use the GPU specified by CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible:
            print(f" Using device: {device} (CUDA_VISIBLE_DEVICES={cuda_visible})")
        else:
            print(f" Using device: {device} (default GPU 0)")
    else:
        device = 'cpu'
        print(f" Using device: {device}")
    print(f" Training 1 SATNet model for {num_outputs} output(s)...")
    
    # Train a single model for all outputs
    model = train_satnet_model(
        X_train, Y_train, 
        num_inputs, num_outputs, 
        device=device,
        n_epochs=300,
        lr=0.1,
        batch_size=min(100, X_train.shape[0])
    )
    
    # Prepare test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32, device=device)
    
    # For training predictions
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32, device=device)
    
    # Prepare inputs for prediction (following Sudoku pattern)
    # Unknown outputs initialized as 0
    Y_train_initial = torch.zeros_like(Y_train_tensor)
    Y_test_initial = torch.zeros_like(Y_test_tensor)
    
    z_train = torch.cat([X_train_tensor, Y_train_initial], dim=1)
    z_test = torch.cat([X_test_tensor, Y_test_initial], dim=1)
    
    is_input_train = torch.cat([
        torch.ones(X_train_tensor.shape[0], num_inputs, dtype=torch.int32, device=device),
        torch.zeros(Y_train_tensor.shape[0], num_outputs, dtype=torch.int32, device=device)
    ], dim=1)
    
    is_input_test = torch.cat([
        torch.ones(X_test_tensor.shape[0], num_inputs, dtype=torch.int32, device=device),
        torch.zeros(Y_test_tensor.shape[0], num_outputs, dtype=torch.int32, device=device)
    ], dim=1)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Training predictions
        z_pred_train = model(z_train, is_input_train)
        Y_pred_train = z_pred_train[:, num_inputs:]  # Extract all outputs
        Y_pred_train_binary = (torch.clamp(Y_pred_train, 0, 1) > 0.5).cpu().numpy().astype(int)
        
        # Test predictions
        z_pred_test = model(z_test, is_input_test)
        Y_pred_test = z_pred_test[:, num_inputs:]  # Extract all outputs
        Y_pred_test_binary = (torch.clamp(Y_pred_test, 0, 1) > 0.5).cpu().numpy().astype(int)
    
    # If Y_ground_truth is provided, use it for evaluation instead of noisy Y
    if Y_ground_truth is not None:
        _, _, Y_train_gt, Y_test_gt = split_data(X, Y_ground_truth, test_size=1-split, random_state=42)
        aggregated_metrics = aggregate_multi_output_metrics(
            Y_train_gt, Y_test_gt,
            Y_pred_train_binary, Y_pred_test_binary
        )
        print(f"  Using ground truth for evaluation")
    else:
        aggregated_metrics = aggregate_multi_output_metrics(
            Y_train, Y_test,
            Y_pred_train_binary, Y_pred_test_binary
        )
    
    accuracy_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if aggregated_metrics:
        accuracy_tuple = (
            aggregated_metrics['train_bit_acc'],
            aggregated_metrics['test_bit_acc'],
            aggregated_metrics['train_sample_acc'],
            aggregated_metrics['test_sample_acc'],
            aggregated_metrics['train_output_acc'],
            aggregated_metrics['test_output_acc']
        )
    accuracies = [accuracy_tuple]
    
    # Expressions (similar to neural network methods)
    expressions = ["SATNet_CONSTRAINTS"] * num_outputs
    
    # Complexity calculation removed - user will calculate separately
    # Store basic model info for reference (without complexity calculation)
    model_info = {
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'n': num_inputs + num_outputs,
        'm': model.S.shape[1],  # clause matrix rank
        'aux': model.aux,
        'S_shape': model.S.shape
    }
    
    all_vars_used = False
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics,
        'model_info': model_info
    }
    
    return expressions, accuracies, extra_info
