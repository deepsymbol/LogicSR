import os
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
from evaluate.operator_config import get_method_config
from evaluate.data_loader import split_data
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics

# Add AI4EDA_TNet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'AI4EDA_TNet', 'experiments'))

# Import AI4EDA_TNet original components
from model import TNet
from truth_table_datasets import TruthTableDataset


def set_operators(operators):
    config = get_method_config("nn_ai4eda")
    config.set_operators(operators, "AI4EDA TNet")


def create_truth_table_file(X: np.ndarray, Y: np.ndarray) -> str:
    """Create truth table file for AI4EDA_TNet from input-output pairs"""
    import tempfile
    
    num_inputs = X.shape[1]
    num_outputs = Y.shape[1]
    full_table_size = 2**num_inputs
    
    # Create mapping from input patterns to output values
    pattern_to_outputs = {}
    for i, (input_row, output_row) in enumerate(zip(X, Y)):
        pattern_int = 0
        for j, bit in enumerate(input_row):
            pattern_int += int(bit) * (2**(num_inputs - 1 - j))
        pattern_to_outputs[pattern_int] = output_row
    
    # Create complete truth table for each output
    truth_table_lines = []
    for output_idx in range(num_outputs):
        output_values = ['0'] * full_table_size
        
        # Fill in known values
        for pattern_int, output_row in pattern_to_outputs.items():
            output_values[pattern_int] = str(int(output_row[output_idx]))
        
        # Join into a single line
        truth_table_lines.append(''.join(output_values))
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.truth', delete=False)
    temp_file.write('\n'.join(truth_table_lines))
    temp_file.close()
    
    return temp_file.name


def train_ai4eda_model(truth_table_file: str, num_inputs: int, num_outputs: int, device: str = 'cuda'):
    
    # Create model with default TNet configuration
    model = TNet(
        in_dim=num_inputs,
        out_dim=num_outputs,
        up_k=10,
        up_l=30,
        down_k=60,
        down_l=10,
        device=device,
        tau=1.0,
        descent_layer=False,
        descent_layer_in=False
    )
    model = model.to(device)
    
    # Create log directory and file
    log_dir = "./logs_base"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"ai4eda_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    
    dataset = TruthTableDataset(
        input_nums=num_inputs, 
        truth_table_file=truth_table_file, 
        truth_flip=False
    )
    
    batch_size = min(2**num_inputs, 1024)  # Dynamic batch size based on input bits, max 1024 (AI4EDA default)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(5000):
        model.train()
        total_loss = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            for param in model.parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-1, 1)
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            # Calculate accuracies on the last batch
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                bit_acc = (preds == y_batch).float().mean().item()
                sample_acc = ((preds == y_batch).all(dim=1).float().mean().item())
            
            log_line = (f"Epoch {epoch:04d} | Loss={total_loss/len(train_loader):.4f} "
                        f"| BitAcc={bit_acc:.3f} | SampleAcc={sample_acc:.3f}")
            print(f"   {log_line}")
            log_file.write(log_line + "\n")
            log_file.flush()
    
    log_file.close()
    return model


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using AI4EDA TNet
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print(" AI4EDA TNet (Neural Network)")
    print("=" * 60)

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)
    num_inputs = X.shape[1]
    num_outputs = Y.shape[1]

    truth_table_file = create_truth_table_file(X_train, Y_train)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_ai4eda_model(truth_table_file, num_inputs, num_outputs, device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        Y_pred_train = (model(X_train_tensor) > 0).cpu().numpy().astype(int)
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

    os.unlink(truth_table_file)

    expressions = ["NEURAL_NETWORK_AI4EDA"] * num_outputs
    all_vars_used = False
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, accuracies, extra_info
