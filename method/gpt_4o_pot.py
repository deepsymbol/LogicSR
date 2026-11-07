import requests
import numpy as np
import os
import re
import time
import io
import contextlib
import textwrap
from typing import Optional
from evaluate.data_loader import split_data
from evaluate.metrics import (evaluate_expression, calculate_metrics,
                              aggregate_multi_output_metrics)
from evaluate.operator_config import get_method_config


def set_operators(operators):
    config = get_method_config("gpt_4o")
    config.set_operators(operators, "GPT-4O")


MODEL_CONFIG = {
    'endpoint':
    f"{os.getenv('GPT4O_API_BASE','')}/openai/deployments/gpt-4o/chat/completions?api-version={os.getenv('GPT4O_API_VERSION','2024-08-01-preview')}",
    'model': os.getenv('GPT4O_MODEL', 'gpt-4o')
}

# Cost calculation can be done externally based on token usage
# Token usage is recorded in llm_metrics for user to calculate costs


class SolverExecutionError(Exception):

    def __init__(self, message, stdout_text="", stderr_text=""):
        super().__init__(message)
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text


class LLMLogicEvaluator:

    def __init__(self):
        self.headers = {
            'api-key': os.getenv('GPT4O_API_KEY', ''),
            'Content-Type': 'application/json'
        }

    def _allowed_operator_text(self, config) -> str:
        op_desc = []
        if config.has_and():
            op_desc.append("'and'")
        if config.has_or():
            op_desc.append("'or'")
        if config.has_not():
            op_desc.append("'not'")
        allowed_ops = ", ".join(op_desc) if op_desc else "'and', 'or', 'not'"
        return (f"Allowed operators: {allowed_ops}.\n"
                "Use lowercase operator names. Only use the allowed operators above.\n\n")

    def _format_truth_table_markdown(self, X: np.ndarray,
                                     Y: np.ndarray) -> str:
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]
        header = [f"x{i+1}" for i in range(n_inputs)
                  ] + [f"y{j+1}" for j in range(n_outputs)]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["-" * 4 for _ in header]) + "|")
        for idx in range(len(X)):
            row_vals = [str(int(b))
                        for b in X[idx]] + [str(int(b)) for b in Y[idx]]
            lines.append("| " + " | ".join(row_vals) + " |")
        return "\n".join(lines)

    def generate_prompt_multi(self, X: np.ndarray, Y: np.ndarray) -> str:
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]

        prompt = (
            "You are performing a logic symbolic regression task. Based on the truth table's input-output relationships, find the simplest boolean expressions.\n\n"
        )
        prompt += "Task Description:\n"
        prompt += "1. You are given a complete truth table with inputs and outputs.\n"
        prompt += "2. For each output yK, produce a correct and as-simplified-as-possible boolean expression.\n"
        prompt += f"3. Use variables x1..x{n_inputs} for inputs. Do not use variables outside this range.\n"
        prompt += "4. Return exactly one line per output in the form 'yK = <expression>'.\n"
        prompt += "5. Use explicit parentheses to indicate grouping.\n"
        prompt += "6. If an output yK is a constant function, you may use the constant 0 or 1 for its expression.\n"
        prompt += "7. No extra commentary besides the lines 'yK = ...'.\n\n"

        prompt += "Optimization Goal (in order of priority):\n"
        prompt += "- (1) Exact correctness on the provided truth table.\n"
        prompt += "- (2) Minimal expression size (fewest gates / shortest simplified form).\n\n"

        config = get_method_config("gpt_4o")
        op_desc = []
        if config.has_and():
            op_desc.append("'and'")
        if config.has_or():
            op_desc.append("'or'")
        if config.has_not():
            op_desc.append("'not'")
        prompt += f"Allowed operators: {', '.join(op_desc)}.\n"
        prompt += "Use lowercase operator names. Only use the allowed operators above.\n\n"

        if config.has_or():
            prompt += "Use 'and', 'not', 'or' to find the relationship between y and x1, x2, ...\n\n"
        else:
            prompt += "Use 'and', 'not' to find the relationship between y and x1, x2, ...\n\n"

        prompt += "Few-shot examples (teaching sequences):\n"
        prompt += "Example 1 (2-input AND):\n"
        prompt += "| x1 | x2 | y1 |\n"
        prompt += "|----|----|----|\n"
        prompt += "| 0  | 0  | 0  |\n"
        prompt += "| 0  | 1  | 0  |\n"
        prompt += "| 1  | 0  | 0  |\n"
        prompt += "| 1  | 1  | 1  |\n"
        prompt += "Answer:\n"
        prompt += "y1 = x1 and x2\n\n"

        prompt += "Example 2 (2-input XOR):\n"
        prompt += "| x1 | x2 | y1 |\n"
        prompt += "|----|----|----|\n"
        prompt += "| 0  | 0  | 0  |\n"
        prompt += "| 0  | 1  | 1  |\n"
        prompt += "| 1  | 0  | 1  |\n"
        prompt += "| 1  | 1  | 0  |\n"
        prompt += "Answer:\n"
        prompt += "y1 = (x1 and not x2) or (not x1 and x2)\n\n"

        header = [f"x{i+1}" for i in range(n_inputs)
                  ] + [f"y{j+1}" for j in range(n_outputs)]
        prompt += "Complete truth table:\n"
        prompt += "| " + " | ".join(header) + " |\n"
        prompt += "|" + "|".join(["-" * 4 for _ in header]) + "|\n"

        for idx in range(len(X)):
            row_vals = [str(int(b))
                        for b in X[idx]] + [str(int(b)) for b in Y[idx]]
            prompt += "| " + " | ".join(row_vals) + " |\n"

        prompt += "\nReturn results now:"
        return prompt

    def generate_cot_prompt(self, X: np.ndarray, Y: np.ndarray) -> str:
        config = get_method_config("gpt_4o")
        table_md = self._format_truth_table_markdown(X, Y)
        allowed_ops = ", ".join(sorted(config.get_operators()))
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]
        prompt = textwrap.dedent(f"""
        You are performing a logic symbolic regression task. Based on the truth table's input-output relationships, analyze and find the simplest boolean expressions.

        Task Description:
        1. You are given a complete truth table with {n_inputs} inputs (x1, x2, ..., x{n_inputs}) and {n_outputs} outputs (y1, y2, ..., y{n_outputs}).
        2. Each row in the truth table represents one input combination and its corresponding output values.
        3. Your goal is to find the simplest boolean expression for each output yK that matches the truth table exactly.

        Truth Table Format:
        - Each row shows one complete input-output combination.
        - Columns x1..x{n_inputs} are input variables (0 or 1).
        - Columns y1..y{n_outputs} are output values (0 or 1).
        - Example: For a 2-input 1-output case, if x1=[1,1,0,0], x2=[1,0,1,0], y1=[1,0,0,0], this means:
          * When x1=1, x2=1 → y1=1
          * When x1=1, x2=0 → y1=0
          * When x1=0, x2=1 → y1=0
          * When x1=0, x2=0 → y1=0
          * So y1 = x1 and x2

        Constraints:
        - Use variables x1, x2, ..., x{n_inputs} for inputs.
        - Allowed operators: {allowed_ops}.
        - Expressions must match the truth table exactly for all rows.
        - Aim for the simplest expression possible.

        Provide a detailed Chain-of-Thought analysis that:
        1. Identifies minterms / maxterms for each output yK (which input combinations produce output 1).
        2. Explains grouping / simplification ideas (how to combine minterms into simpler expressions).
        3. Mentions any sanity checks you would perform (e.g., verifying all rows match).

        Do NOT output final expressions or code yet. Just provide your reasoning.
        End with a short "Reasoning Summary" describing the best candidate expression for each output.

        Truth table:
        {table_md}

        Chain-of-thought:""").strip()
        return prompt

    def _pot_few_shot_examples(self) -> str:
        example_and = textwrap.dedent("""\
        def solve():
            # Step 1: Use the provided truth table data (X_data, Y_data)
            # X_data shape: (n_samples, n_inputs) - each row is one input combination
            # Y_data shape: (n_samples, n_outputs) - each row is corresponding output values
            # For this AND example: X_data = [[0,0], [0,1], [1,0], [1,1]], Y_data = [[0], [0], [0], [1]]
            
            # Step 2: Identify minterms for y1 (rows where y1=1)
            minterms = []
            for i in range(len(X_data)):
                if Y_data[i, 0] == 1:  # y1 is in column 0
                    minterms.append(tuple(X_data[i]))
            # Result: minterms = [(1, 1)]

            # Step 3: Derive candidate expression from minterms
            # For AND: only minterm is (1,1), so y1 = x1 and x2
            expr = "(x1 and x2)"

            # Step 4: Verify expression against all truth table rows
            def f(x1, x2):
                return int(x1 and x2)

            for i in range(len(X_data)):
                x1, x2 = X_data[i]
                y_pred = f(x1, x2)
                y_true = Y_data[i, 0]
                assert y_pred == y_true, f"Row {i}: expected {y_true}, got {y_pred}"

            # Step 5: Print final expression
            print("y1 = (x1 and x2)")""")

        example_xor = textwrap.dedent("""\
        def solve():
            # XOR example: X_data = [[0,0], [0,1], [1,0], [1,1]], Y_data = [[0], [1], [1], [0]]
            
            # Step 1: Identify minterms for y1 (rows where y1=1)
            minterms = []
            for i in range(len(X_data)):
                if Y_data[i, 0] == 1:  # y1 is in column 0
                    minterms.append(tuple(X_data[i]))
            # Result: minterms = [(0, 1), (1, 0)]

            # Step 2: Construct expression from minterms
            # Minterm (0,1): not x1 and x2
            # Minterm (1,0): x1 and not x2
            # Combined: y1 = (not x1 and x2) or (x1 and not x2)
            expr = "((not x1 and x2) or (x1 and not x2))"

            # Step 3: Verify correctness against all rows
            def f(x1, x2):
                return int((not x1 and x2) or (x1 and not x2))

            for i in range(len(X_data)):
                x1, x2 = X_data[i]
                y_pred = f(x1, x2)
                y_true = Y_data[i, 0]
                assert y_pred == y_true, f"Row {i}: expected {y_true}, got {y_pred}"

            # Step 4: Print final expression
            print("y1 = " + expr)""")
        return f"{example_and}\n\n{example_xor}"

    def generate_pot_prompt(self,
                             X: np.ndarray,
                             Y: np.ndarray,
                             cot_text: str,
                             retry_feedback: Optional[str] = None) -> str:
        config = get_method_config("gpt_4o")
        allowed_ops = sorted(config.get_operators())
        x_list = X.astype(int).tolist()
        y_list = Y.astype(int).tolist()
        n_inputs = X.shape[1]
        n_outputs = Y.shape[1]
        table_md = self._format_truth_table_markdown(X, Y)
        few_shot = self._pot_few_shot_examples()

        retry_block = ""
        if retry_feedback:
            retry_block = textwrap.dedent(f"""

            Previous attempt details:
            {retry_feedback}

            You must fix every reported issue. Keep refining expressions until all assertions pass. Always print the best candidate expressions before raising errors.
            """).strip()

        prompt = textwrap.dedent(f"""
        You MUST perform Program-of-Thought reasoning and output ONLY executable Python code.

        Task: Implement a solve() function that finds boolean expressions for {n_outputs} outputs based on the truth table.

        Data Format (already provided in runtime context):
        - X_data: numpy array of shape ({len(X)}, {n_inputs}) - each row is one input combination [x1, x2, ..., x{n_inputs}]
        - Y_data: numpy array of shape ({len(X)}, {n_outputs}) - each row is corresponding output values [y1, y2, ..., y{n_outputs}]
        - For each row i: X_data[i] gives input values, Y_data[i] gives output values for all outputs
        - Example: X_data[0] = {x_list[0] if x_list else '[...]'}, Y_data[0] = {y_list[0] if y_list else '[...]'}

        Code Requirements:
        - Use the provided X_data and Y_data arrays (do NOT reconstruct them).
        - All reasoning must appear inside Python comments + executable logic.
        - For each output yK (K = 1 to {n_outputs}):
          * Identify minterms (rows where Y_data[:, K-1] == 1)
          * Derive the simplest boolean expression
          * Verify the expression matches ALL rows in the truth table
        - Use only allowed operators: {', '.join(allowed_ops)} (i.e., 'and', 'or', 'not').
        - Print exactly {n_outputs} lines in the format "yK = <expression>" (K from 1 to {n_outputs}).
        - Expressions must use lowercase operator names (and, or, not).
        - If an assertion fails, update your expressions and re-run verification until the assertions pass. If you still cannot satisfy the truth table, you must print the best candidate expressions before raising an AssertionError that clearly states the failing row.
        - IMPORTANT: When verifying expressions, extract variables from X_data like this:
          * For each row i: x1, x2, ... = X_data[i]  (unpack the row)
          * Then call your verification function with these values
          * Do NOT use eval() or exec() on expression strings - write the expression directly in your function
        - No natural-language text outside the code block.

        Runtime context (already available when your code executes):
        import numpy as np
        X_data = np.array({x_list}, dtype=int)  # Shape: ({len(X)}, {n_inputs})
        Y_data = np.array({y_list}, dtype=int)  # Shape: ({len(X)}, {n_outputs})
        n_inputs = {n_inputs}
        n_outputs = {n_outputs}
        allowed_ops = {allowed_ops}

        Truth table (for reference):
        {table_md}

        Chain-of-Thought analysis provided:
        {cot_text}

        {retry_block}

        Follow the Chain-of-Thought above and now produce the Program-of-Thought code.
        Use the following examples as style guidance (notice how they use X_data and Y_data):

        ```python
        {few_shot}
        ```

        Output ONLY the final Python code implementing solve(), nothing else.
        """).strip()
        return prompt

    def query_model(self, messages):
        """Send message list (or single prompt string) and return assistant content plus metadata."""
        print(f" Sending request to: {MODEL_CONFIG['endpoint']}")
        print(f" Using model: {MODEL_CONFIG['model']}")
        if isinstance(messages, str):
            messages = [{
                "role": "user",
                "content": messages
            }]
        conversation = messages[:]
        has_system = any(msg.get("role") == "system" for msg in conversation)
        if not has_system:
            conversation = [{
                "role": "system",
                "content": "You are a professional logic circuit expert."
            }] + conversation
        data = {"messages": conversation}
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

        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if seed is not None:
            data["seed"] = seed

        print(" Using decoding params (unset => server defaults):",
              {
                  "temperature": temperature,
                  "top_p": top_p,
                  "max_tokens": max_tokens,
                  "seed": seed,
              })
        print(" Sending prompt to GPT...")
        # Measure API response time (includes inference time + network overhead)
        # For API-based models, this is the end-to-end response time from request to completion
        start_time = time.perf_counter()
        response = requests.post(MODEL_CONFIG["endpoint"],
                                 json=data,
                                 headers=self.headers)
        response_time_s = time.perf_counter() - start_time
        response_json = response.json()
        content = response_json['choices'][0]['message']['content'].strip()

        usage = response_json.get('usage', {}) or {}
        prompt_tokens = int(usage.get('prompt_tokens', 0))
        completion_tokens = int(usage.get('completion_tokens', 0))
        total_tokens = int(usage.get('total_tokens', prompt_tokens + completion_tokens))

        metadata = {
            "api_response_time_s": response_time_s,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        return content, metadata

    def extract_code_block(self, text: str) -> str:
        # Try to extract code from markdown code blocks
        # Pattern 1: ```python ... ```
        pattern1 = r"```python\s*\n(.*?)```"
        match = re.search(pattern1, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code
        
        # Pattern 2: ``` ... ``` (any language)
        pattern2 = r"```[^\n]*\n(.*?)```"
        match = re.search(pattern2, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                # Remove any leading ```python or similar markers
                code = re.sub(r'^```[^\n]*\n', '', code, flags=re.MULTILINE)
                return code
        
        # If no fenced block found, assume entire text is code
        # But strip any leading/trailing markdown markers
        code = text.strip()
        # Remove leading ```python or ```
        code = re.sub(r'^```(?:python)?\s*\n?', '', code, flags=re.MULTILINE)
        # Remove trailing ```
        code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
        return code.strip()

    def extract_expressions_from_code_text(self, text: str,
                                           n_outputs: int) -> list:
        exprs = ["0"] * n_outputs
        found = set()
        line_patterns = [
            re.compile(r'print\s*\(\s*["\']y(\d+)\s*=\s*(.+?)["\']\s*\)'),
            re.compile(r'y(\d+)\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'"y(\d+)"\s*:\s*["\']([^"\']+)["\']')
        ]

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in line_patterns:
                match = pattern.search(stripped)
                if match:
                    idx = int(match.group(1)) - 1
                    expr_txt = match.group(2).strip()
                    if 0 <= idx < n_outputs:
                        exprs[idx] = self.validate_expression(expr_txt)
                        found.add(idx)
        return exprs if found else ["0"] * n_outputs

    def format_retry_feedback(self, attempt: int,
                              error: SolverExecutionError) -> str:
        parts = [f"Attempt {attempt} failed with error: {error}."]
        if error.stdout_text:
            preview = error.stdout_text.strip()
            if len(preview) > 400:
                preview = preview[-400:]
            parts.append("Captured stdout before failure:\n" + preview)
        if error.stderr_text:
            stderr_preview = error.stderr_text.strip()
            if len(stderr_preview) > 400:
                stderr_preview = stderr_preview[-400:]
            parts.append("Captured stderr:\n" + stderr_preview)
        return "\n".join(parts)

    def execute_solver_code(self,
                            code_text: str,
                            X: np.ndarray,
                            Y: np.ndarray,
                            allowed_ops) -> str:
        code = self.extract_code_block(code_text)
        if not code:
            raise ValueError("No executable code found in model response.")
        
        print(f"  Extracted code length: {len(code)} chars")
        if code:
            print(f"  Code preview (first 300 chars): {code[:300]}")
            # Check for common issues
            if code.startswith('```'):
                print(f"  Warning: Code still starts with markdown marker, first 50 chars: {repr(code[:50])}")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        # Create exec_globals with necessary variables and functions
        # Provide x1, x2, ... as lambda functions that extract from X_data (for eval compatibility)
        exec_globals = {
            '__builtins__': __builtins__,
            'np': np,
            'X_data': X.astype(int),
            'Y_data': Y.astype(int),
            'n_inputs': X.shape[1],
            'n_outputs': Y.shape[1],
            'allowed_ops': allowed_ops
        }
        
        # Add x1, x2, ... as global variables that can be used in eval()
        # These are set to None initially - actual values come from X_data when iterating
        # But we provide helper functions that extract values from X_data
        for i in range(X.shape[1]):
            var_name = f'x{i+1}'
            # Create a lambda that will be used if eval() is called
            # Note: actual usage should be from X_data[i], but this helps with eval()
            exec_globals[var_name] = None  # Will be set dynamically if needed
        exec_locals = {}

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(code, exec_globals, exec_locals)
                solve_fn = exec_locals.get('solve') or exec_globals.get('solve')
                if callable(solve_fn):
                    # Before calling solve(), ensure x1, x2, ... are available if needed
                    # If solve() uses eval(), it will need these variables in the global scope
                    # But typically, solve() should extract from X_data directly
                    solve_fn()
                else:
                    raise ValueError(f"solve() function not found in generated code. Available: {list(exec_locals.keys())}")
        except AssertionError as e:
            stdout_output = stdout_buffer.getvalue()
            stderr_msg = stderr_buffer.getvalue()
            if stdout_output:
                print(f"  Assertion failed but captured output before error: {len(stdout_output)} chars")
            raise SolverExecutionError(str(e), stdout_output, stderr_msg) from e
        except Exception as e:
            stdout_output = stdout_buffer.getvalue()
            stderr_msg = stderr_buffer.getvalue()
            if stdout_output:
                print(f"  Captured output before error: {stdout_output[:200]}")
            if stderr_msg:
                print(f"  Execution stderr: {stderr_msg[:200]}")
            raise SolverExecutionError(str(e), stdout_output, stderr_msg) from e

        stdout_output = stdout_buffer.getvalue()
        if not stdout_output:
            print("  Warning: solve() executed but produced no stdout output")
        return stdout_output


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
    """Find logic expressions using GPT model"""
    print("=" * 60)
    print(" GPT-4o (Large Language Model)")
    print("=" * 60)

    expressions = []
    metrics_list = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)
    evaluator = LLMLogicEvaluator()

    cot_prompt = evaluator.generate_cot_prompt(X_train, Y_train)
    conversation = [{
        "role": "user",
        "content": cot_prompt
    }]
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_response_time_s = 0.0

    cot_resp, cot_meta = evaluator.query_model(conversation)
    total_response_time_s += cot_meta.get("api_response_time_s", 0.0)
    usage = cot_meta.get("usage", {})
    total_prompt_tokens += usage.get("prompt_tokens", 0)
    total_completion_tokens += usage.get("completion_tokens", 0)

    config = get_method_config("gpt_4o")
    allowed_ops = sorted(config.get_operators())
    max_pot_attempts = int(os.getenv("POT_MAX_RETRIES", 2))
    execution_output = ""
    last_error = None
    last_code_resp = ""
    retry_feedback = None

    for attempt in range(1, max_pot_attempts + 1):
        pot_prompt = evaluator.generate_pot_prompt(X_train, Y_train,
                                                   cot_resp, retry_feedback)
        pot_conversation = [{
            "role": "user",
            "content": pot_prompt
        }]
        code_resp, pot_meta = evaluator.query_model(pot_conversation)
        last_code_resp = code_resp
        total_response_time_s += pot_meta.get("api_response_time_s", 0.0)
        usage = pot_meta.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        try:
            execution_output = evaluator.execute_solver_code(
                code_resp, X_train, Y_train, allowed_ops)
            print(
                f"  PoT execution output length: {len(execution_output)} chars")
            if execution_output:
                print(
                    f"  PoT execution output preview: {execution_output[:200]}")
            last_error = None
            break
        except SolverExecutionError as exc:
            last_error = exc
            retry_feedback = evaluator.format_retry_feedback(attempt, exc)
            print(f"  PoT execution failed on attempt {attempt}: {exc}")
            if attempt == max_pot_attempts:
                execution_output = exc.stdout_text or ""

    if not execution_output and last_error:
        execution_output = last_error.stdout_text or ""

    expr_list = evaluator.parse_multi_response(execution_output or "",
                                               Y_train.shape[1])
    if all(expr == "0" for expr in expr_list):
        print("  Parsing stdout yielded empty expressions, attempting code-text extraction.")
        expr_from_code = evaluator.extract_expressions_from_code_text(
            last_code_resp, Y_train.shape[1])
        if any(expr != "0" for expr in expr_from_code):
            expr_list = expr_from_code
        else:
            print(
                "  Code-text extraction failed, falling back to plain expression prompt."
            )
            prompt_multi = evaluator.generate_prompt_multi(X_train, Y_train)
            raw_resp, multi_meta = evaluator.query_model(prompt_multi)
            total_response_time_s += multi_meta.get("api_response_time_s", 0.0)
            usage = multi_meta.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            expr_list = evaluator.parse_multi_response(raw_resp,
                                                       Y_train.shape[1])

    if all(expr == "0" for expr in expr_list):
        print(f"  Warning: All expressions are '0', execution_output was: {repr(execution_output[:500])}")

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
    total_tokens = total_prompt_tokens + total_completion_tokens
    extra_info = {
        "all_vars_used": True,
        "aggregated_metrics": aggregated_metrics,
        "llm_metrics": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "api_response_time_s": total_response_time_s
        }
    }
    return expressions, metrics_list, extra_info
