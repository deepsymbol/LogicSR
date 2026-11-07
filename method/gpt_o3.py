import requests
import numpy as np
import os
import re
from evaluate.data_loader import split_data  
from evaluate.metrics import (evaluate_expression, calculate_metrics,
                              aggregate_multi_output_metrics) 
from evaluate.operator_config import get_method_config  


def set_operators(operators):
    config = get_method_config("gpt_o3")
    config.set_operators(operators, "GPT-O3")


MODEL_CONFIG = {
    'endpoint':
    f"{os.getenv('GPT_O3_API_BASE','')}/openai/deployments/o3/chat/completions?api-version={os.getenv('GPT_O3_API_VERSION','2025-01-01-preview')}",
    'model': 'o3'
}


class LLMLogicEvaluator:

    def __init__(self):
        self.headers = {
            'api-key': os.getenv('GPT_O3_API_KEY', ''),
            'Content-Type': 'application/json'
        }

    def generate_prompt_multi(self, X: np.ndarray, Y: np.ndarray) -> str:
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]

        prompt = (
            "You are performing a logic symbolic regression task. Based on the truth table's input-output relationships, find the simplest boolean expressions.\n\n"
        )

        prompt += "Task Description:\n"
        prompt += "1. You are given a complete truth table with inputs and outputs.\n"
        prompt += "2. For each output yK, produce a correct and as-simplified-as-possible boolean expression.\n"
        prompt += "3. Use variables x1..x{n_inputs} for inputs. Do not use variables outside this range.\n".replace("{n_inputs}", str(n_inputs))
        prompt += "4. Return exactly one line per output in the form 'yK = <expression>'.\n"
        prompt += "5. Use explicit parentheses to indicate grouping.\n"
        prompt += "6. If an output yK is a constant function, you may use the constant 0 or 1 for its expression.\n"
        prompt += "7. No extra commentary besides the lines 'yK = ...'.\n\n"

        prompt += "Optimization Goal (in order of priority):\n"
        prompt += "- (1) Exact correctness on the provided truth table.\n"
        prompt += "- (2) Minimal expression size (fewest gates / shortest simplified form).\n\n"

        # Operator list - dynamically based on configuration
        config = get_method_config("gpt_o3")

        op_desc = []
        if config.has_and():
            op_desc.append("'and'")
        if config.has_or():
            op_desc.append("'or'")
        if config.has_not():
            op_desc.append("'not'")
        prompt += "Operator Configuration:\n"
        prompt += f"1. Available Operators: {', '.join(op_desc)} (dynamically configured)\n"
        prompt += "2. Usage Rules:\n"
        prompt += "   (a) Use lowercase operator names only\n"
        prompt += "   (b) Strictly use only the available operators listed above\n"
        prompt += "3. Configuration Note: The operator set is dynamically controlled by the system configuration. Only the operators marked as available in the current configuration may be used in your expressions.\n\n"

        # Few-shot teaching examples to guide the model (symbolic regression on truth tables)
        prompt += "Few-shot examples (teaching sequences):\n"
        # Example 1: AND
        prompt += "Example 1 (2-input AND):\n"
        prompt += "| x1 | x2 | y1 |\n"
        prompt += "|----|----|----|\n"
        prompt += "| 0  | 0  | 0  |\n"
        prompt += "| 0  | 1  | 0  |\n"
        prompt += "| 1  | 0  | 0  |\n"
        prompt += "| 1  | 1  | 1  |\n"
        prompt += "Answer:\n"
        prompt += "y1 = x1 and x2\n\n"

        # Example 2: XOR
        prompt += "Example 2 (2-input XOR):\n"
        prompt += "| x1 | x2 | y1 |\n"
        prompt += "|----|----|----|\n"
        prompt += "| 0  | 0  | 0  |\n"
        prompt += "| 0  | 1  | 1  |\n"
        prompt += "| 1  | 0  | 1  |\n"
        prompt += "| 1  | 1  | 0  |\n"
        prompt += "Answer:\n"
        prompt += "y1 = (x1 and not x2) or (not x1 and x2)\n\n"

        # Data Preparation: Complete Truth Table
        prompt += "Data Preparation: Complete Truth Table\n"
        prompt += "1. Data Loading: The complete truth table data has been prepared and loaded for analysis.\n"
        prompt += f"2. Table Structure: {n_inputs} input variable(s) (x1..x{n_inputs}) and {n_outputs} output variable(s) (y1..y{n_outputs}).\n\n"
        
        # Build complete truth table
        header = [f"x{i+1}" for i in range(n_inputs)
                  ] + [f"y{j+1}" for j in range(n_outputs)]
        prompt += "Complete truth table:\n"
        prompt += "| " + " | ".join(header) + " |\n"
        prompt += "|" + "|".join(["-" * 4 for _ in header]) + "|\n"

        # Show all rows (complete truth table)
        for idx in range(len(X)):
            row_vals = [str(int(b))
                        for b in X[idx]] + [str(int(b)) for b in Y[idx]]
            prompt += "| " + " | ".join(row_vals) + " |\n"

        prompt += "\nReturn results now:"
        return prompt

    def query_model(self, prompt: str) -> str:
        """Send prompt and return assistant content string."""
        print(f" Sending request to: {MODEL_CONFIG['endpoint']}")
        print(f" Using model: {MODEL_CONFIG['model']}")
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional logic circuit expert."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
        }
        # Optional decoding params via environment variables (only attached if set)
        def _env_float(name):
            val = os.getenv(name)
            try:
                return float(val) if val not in (None, "") else None
            except Exception:
                return None
        def _env_int(name):
            val = os.getenv(name)
            try:
                return int(val) if val not in (None, "") else None
            except Exception:
                return None

        temperature = _env_float("LLM_TEMPERATURE")
        top_p = _env_float("LLM_TOP_P")
        max_tokens = _env_int("LLM_MAX_TOKENS")
        seed = _env_int("LLM_SEED")
        effort = os.getenv("LLM_REASONING_EFFORT")  # e.g., medium/high (if supported)

        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if seed is not None:
            data["seed"] = seed
        if effort:
            data["reasoning"] = {"effort": effort}

        print(" Using decoding params (unset => server defaults):",
              {
                  "temperature": temperature,
                  "top_p": top_p,
                  "max_tokens": max_tokens,
                  "seed": seed,
                  "reasoning.effort": effort if effort else None,
              })
        print(" Sending prompt to GPT...")
        response = requests.post(MODEL_CONFIG["endpoint"],
                                 json=data,
                                 headers=self.headers)
        content = response.json()['choices'][0]['message']['content'].strip()
        return content


    def parse_multi_response(self, raw_resp: str, n_outputs: int) -> list:
        exprs = ["0"] * n_outputs
        lines = [ln.strip() for ln in raw_resp.split("\n") if ln.strip()]

        for ln in lines:
            m = re.match(r"y(\d+)\s*[=:]\s*(.+)", ln, flags=re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                expr_part = m.group(2).strip()
                if 0 <= idx < n_outputs:
                    exprs[idx] = self.validate_expression(expr_part)

        return exprs


    def validate_expression(self, expr):
        if not expr or expr in ['0', '1']:
            return expr if expr else "0"

        # Clean up common GPT formatting issues first
        expr = expr.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')
        expr = expr.replace('∧', 'and').replace('∨', 'or').replace('¬', 'not')
        expr = expr.replace('&', 'and').replace('|', 'or').replace('~', 'not')

        # Check if it's a valid expression (contains operators or is a single variable)
        if any(op in expr for op in ['and', 'or', 'not']) or any(f'x{i}' in expr for i in range(1, 10)):
            # Remove extra parentheses and clean up
            expr = ' '.join(expr.split())
            return expr

        return "0"


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using GPT model
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using expression predictions vs ground truth.
    """
    print("=" * 60)
    print(" GPT-o3 (Large Language Model)")
    print("=" * 60)

    expressions = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)
    evaluator = LLMLogicEvaluator()

    prompt_multi = evaluator.generate_prompt_multi(X_train, Y_train)
    raw_resp = evaluator.query_model(prompt_multi)
    expr_list = evaluator.parse_multi_response(raw_resp, Y_train.shape[1])

    train_pred_columns = []
    test_pred_columns = []

    for idx, expr in enumerate(expr_list):
        y_train = Y_train[:, idx]
        y_test = Y_test[:, idx]
        
        for v in range(1, X.shape[1] + 1):
            if f"x{v}" in expr:
                used_vars.add(f"x{v}")
        
        y_train_pred = evaluate_expression(expr, X_train)
        y_test_pred = evaluate_expression(expr, X_test)
        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)
        expressions.append(expr)
    
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
    extra_info = {
        "aggregated_metrics": aggregated_metrics
    }
    return expressions, metrics_list, extra_info