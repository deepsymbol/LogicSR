import numpy as np
import ast
from typing import Union, List


def count_gates_from_qm_output(qm_output):
    """Count gates from QM output format"""
    if not qm_output:
        return 0
    if isinstance(qm_output, str):
        qm_output = ast.literal_eval(qm_output)
    total_not = 0
    total_and2 = 0
    total_or2 = 0
    for pattern in qm_output:
        k = sum(1 for bit in pattern if bit != '-')
        z = pattern.count('0')
        total_not += z
        total_and2 += max(k - 1, 0)
    m = len(qm_output)
    total_or2 += max(m - 1, 0)
    return total_not + total_and2 + total_or2


def count_gates_from_bdd_output(bdd_output):
    """Count gates from BDD output format"""
    if not bdd_output:
        return 0
    if isinstance(bdd_output, str):
        bdd_output = ast.literal_eval(bdd_output)
    total_not = 0
    total_and2 = 0
    total_or2 = 0
    for path in bdd_output:
        k = len(path)
        z = sum(1 for v in path.values() if v is False)
        total_not += z
        total_and2 += max(k - 1, 0)
    m = len(bdd_output)
    total_or2 += max(m - 1, 0)
    return total_not + total_and2 + total_or2


def count_gates_from_abc_output(abc_output, abc_stdout=None):
    """Count AND gates from ABC output using print_stats"""
    if not abc_stdout:
        return 0
    
    # Extract from ABC print_stats output
    for line in abc_stdout.split('\n'):
        # Handle both 'nd =' and 'and =' formats
        if 'nd =' in line or 'and =' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if (part == 'nd' or part == 'and') and i + 2 < len(parts) and parts[i + 1] == '=':
                    return int(parts[i + 2])
    
    return 0


def count_gates(expr: Union[str, List[str]], input_size: int = None, abc_stdouts: List[str] = None, extra_info: dict = None) -> float:
    """
    Count gates in expression(s) with support for multiple formats
    
    Args:
        expr: Expression string or list of expressions
        input_size: Number of input variables (for neural networks)
        
    Returns:
        Average gate count (float)
    """
    if isinstance(expr, list):
        # Handle list of expressions
        if not expr:
            return np.nan
        valid_expr = [e for e in expr if e not in [None, "FAIL", 0, "-1"]]
        if not valid_expr:
            return np.nan
        if abc_stdouts and len(abc_stdouts) == len(valid_expr):
            # Use ABC stdout for accurate gate counting
            gate_counts = []
            for i, e in enumerate(valid_expr):
                if '.names' in str(e) and '.end' in str(e):
                    gate_counts.append(count_gates_from_abc_output(str(e), abc_stdouts[i]))
                else:
                    gate_counts.append(count_single_expression_gates(str(e), input_size, extra_info))
            return np.mean(gate_counts)
        else:
            return np.mean([count_single_expression_gates(str(e), input_size, extra_info) for e in valid_expr])
    else:
        # Handle single expression
        return count_single_expression_gates(str(expr), input_size, extra_info)

def count_single_expression_gates(expr_str: str, input_size: int = None, extra_info: dict = None) -> int:
    """
    Count gates in a single expression string
    
    Args:
        expr_str: Expression string
        input_size: Number of input variables (for neural networks)
        
    Returns:
        Gate count (int)
    """
    expr_str = str(expr_str).strip()
    
    if not expr_str or expr_str in ["FAIL", "-1", "0"]:
        return 0
    
    # Neural network format: NEURAL_NETWORK_*
    if expr_str.startswith('NEURAL_NETWORK_'):
        from evaluate.metrics import parse_neural_network_complexity
        gates, _ = parse_neural_network_complexity(expr_str, input_size or 0)
        return gates
    
    # SATNet format: SATNet_*
    if expr_str.startswith('SATNet_'):
        # Complexity calculation removed - user will calculate separately
        # Return 0 as placeholder (user should calculate complexity externally)
        return 0
    
    # QM format: {...}
    elif expr_str.startswith('{') and expr_str.endswith('}'):
        try:
            return count_gates_from_qm_output(expr_str)
        except (ValueError, SyntaxError, TypeError):
            # Fallback to string counting
            return count_string_gates(expr_str)
    
    # BDD format: [{...}]
    elif expr_str.startswith("[{") and expr_str.endswith("}]"):
        return count_gates_from_bdd_output(expr_str)
    
    # Espresso format: check if it's a list of expressions
    elif expr_str.startswith('[') and expr_str.endswith(']'):
        try:
            parsed = ast.literal_eval(expr_str)
            if isinstance(parsed, list):
                # For espresso format, count gates for each expression in the list
                total_gates = 0
                for expr in parsed:
                    if expr and expr not in ['0', 'False']:
                        total_gates += count_string_gates(str(expr))
                return total_gates
        except (ValueError, SyntaxError, TypeError):
            pass
    
    # Standard symbolic expressions
    return count_string_gates(expr_str)


def count_string_gates(expr_str: str) -> int:
    """
    Count gates by string pattern matching for standard symbolic expressions
    
    Args:
        expr_str: Expression string
        
    Returns:
        Gate count (int)
    """
    # Count different gate patterns
    gate_count = 0
    
    # Space-surrounded operators: ' and ', ' or ', 'not '
    gate_count += expr_str.count(' and ')
    gate_count += expr_str.count(' or ')
    gate_count += expr_str.count('not ')
    
    # Function call format: 'and(', 'or(', 'not('
    gate_count += expr_str.count('and(')
    gate_count += expr_str.count('or(')
    gate_count += expr_str.count('not(')
    
    # Capital function format: 'And(', 'Or('
    gate_count += expr_str.count('And(')
    gate_count += expr_str.count('Or(')
    
    # Symbol format: '~'
    gate_count += expr_str.count('~')
    
    return gate_count