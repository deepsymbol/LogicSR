import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple, Optional


def load_truth_files(data_path: Optional[str] = None,
                     io_spec: Optional[Tuple[int, int]] = None) -> List[str]:
    """Data Discovery: Find all .truth files based on path or io_spec
    
    Args:
        data_path: Directory or file path to search
        io_spec: Tuple of (input_size, output_size) for pattern matching
    Returns:
        List of file paths to .truth files
    """
    if data_path:
        if os.path.isfile(data_path):
            return [data_path]
        if os.path.isdir(data_path):
            files = sorted(str(f) for f in Path(data_path).rglob('*.truth'))
            return files

        # Path relative to data directory
        data_subdir = Path('data') / data_path
        if data_subdir.exists():
            if data_subdir.is_file():
                return [str(data_subdir)]
            elif data_subdir.is_dir():
                files = sorted(str(f) for f in data_subdir.rglob('*.truth'))
                return files

        print(f"Data path {data_path} not found")
        return []

    # Use io_spec mode
    data_dir = Path('data')
    pattern = f"*/in{io_spec[0]}_out{io_spec[1]}/*.truth" if io_spec else "**/*.truth"
    files = sorted(str(f) for f in data_dir.glob(pattern))
    return files


def load_truth_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Data Loading: Load content from a single .truth file
    
    Args:
        file_path: Path to the .truth file
    Returns:
        X: Input data
        y: Output data
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Convert binary strings to array
    data = np.array([[int(bit) for bit in line] for line in lines])

    # Extract input_size and output_size from file path
    # Support multiple path formats:
    # 1. in{N}_out{M}
    # 2. ANO/in{N}_out{M}/and{N}/noise_{X}
    path_parts = Path(file_path).parts

    # Find path parts containing 'in' and 'out'
    input_size = None
    output_size = None

    for part in path_parts:
        if 'in' in part and 'out' in part:
            # Handle in{N}_out{M} format
            sub_parts = part.split('_')
            for sub_part in sub_parts:
                if sub_part.startswith('in'):
                    input_size = int(sub_part.replace('in', ''))
                elif sub_part.startswith('out'):
                    output_size = int(sub_part.replace('out', ''))
            break

    # If not found, try to infer from filename or parent directory
    if input_size is None or output_size is None:
        # Infer from data row count: first N rows are input, last M rows are output
        total_rows = len(data)
        # Assume input and output counts are equal, or adjust based on actual situation
        input_size = total_rows // 2
        output_size = total_rows - input_size

    # First input_size rows are input variables, last output_size rows are output variables
    # Each column represents one case, need to transpose before returning
    X = data[:
             input_size].T  # After transpose, each row is a sample, each column is an input variable
    y = data[
        input_size:].T  # After transpose, each row is a sample, each column is an output variable
    return X, y


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets
    
    Args:
        X: Input data
        Y: Output data
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, Y_train, Y_test
    """
    # For small datasets (usually truth tables), use complete data for training to avoid information loss
    # Truth tables usually have only 2^n rows, where n is the number of input variables
    # When number of input variables ≤ 7 (2^7=128), use complete data; when over 7 input variables, perform split
    if len(X) <= 128:
        # Small dataset: use complete data for training, test also uses same data
        return X, X, Y, Y
    elif test_size is not None and test_size <= 0:
        return X, X, Y, Y
    else:
        # Large dataset: normal split
        return train_test_split(X,
                                Y,
                                test_size=test_size,
                                random_state=random_state)