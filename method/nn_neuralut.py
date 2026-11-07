import torch
import torch.nn as nn
import numpy as np
import sys
import os
import datetime
from typing import Optional
from evaluate.metrics import (calculate_metrics, aggregate_multi_output_metrics)  
from evaluate.operator_config import get_method_config  
from evaluate.data_loader import split_data 

# Add NeuraLUT to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'NeuraLUT', 'src'))

# NeuraLUT training parameters (configurable)
DEFAULT_EPOCHS = 500
DEFAULT_LR = 0.003
DEFAULT_BATCH_SIZE = 128


def set_operators(operators):
    cfg = get_method_config("nn_neuralut")
    cfg.set_operators(operators, "NeuraLUT")


class TruthTableDataset(torch.utils.data.Dataset):
    """Dataset wrapper for truth table"""
    
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_neuralut_model(input_size: int, output_size: int, device: str = 'cuda'):
    """Create NeuraLUT model using original API"""
    
    # Import the original NeuraLUT model
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'NeuraLUT'))
    from models import MnistNeqModel
    
    model_config = {
        "input_length": input_size,
        "output_length": output_size,
        "hidden_layers": [256, 100, 100, 100],  
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 1,  
        "input_fanin": min(6, input_size),  # Dynamic fanin based on input size
        "hidden_fanin": 6,
        "output_fanin": min(6, output_size),  # Dynamic fanin based on output size
        "width_n": 16,
        "cuda": device == 'cuda'
    }
    
    model = MnistNeqModel(model_config)
    
    return model


def train_neuralut_model(X_train: np.ndarray,
                         Y_train: np.ndarray,
                         num_inputs: int,
                         num_outputs: int,
                         device: str = 'cuda') -> Optional[nn.Module]:
    model = create_neuralut_model(num_inputs, num_outputs, device).to(device)
    
    # Create log directory and file
    log_dir = "./logs_base"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"neuralut_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    
    # Create data loaders
    train_dataset = TruthTableDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=True
    )

    # Configure optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR)

    # Configure criterion
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(DEFAULT_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if epoch % 50 == 0:
            # Calculate accuracies on the last batch
            with torch.no_grad():
                preds = (torch.sigmoid(output) > 0.5).float()
                bit_acc = (preds == target).float().mean().item()
                sample_acc = ((preds == target).all(dim=1).float().mean().item())
            
            log_line = (f"Epoch {epoch:04d} | Loss={total_loss/len(train_loader):.4f} "
                        f"| BitAcc={bit_acc:.3f} | SampleAcc={sample_acc:.3f}")
            print(f"    {log_line}")
            log_file.write(log_line + "\n")
            log_file.flush()

    log_file.close()
    return model


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using NeuraLUT
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    print("=" * 60)
    print(" NeuraLUT (Neural Network)")
    print("=" * 60)

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)
    num_inputs = X.shape[1]
    num_outputs = Y.shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_neuralut_model(X_train, Y_train, num_inputs, num_outputs, device)
    
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

    expressions = ["NEURAL_NETWORK_NEURALUT"] * num_outputs

    extra_info = {
        'all_vars_used': False,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, accuracies, extra_info