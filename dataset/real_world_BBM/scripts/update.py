import os
import random
import re
import os.path as osp
from tqdm import tqdm
import ast
from itertools import product
from read_sbml import read_sbml

# ----- Safe Boolean Expression Evaluator (AST-based) -----
operators = {
    ast.And: all,
    ast.Or: any,
    ast.Not: lambda x: not x
}

def safe_eval_bool_expr(expr_str, local_vars):
    expr_ast = ast.parse(expr_str, mode='eval')
    return _eval_ast(expr_ast.body, local_vars)

def _eval_ast(node, local_vars):
    if isinstance(node, ast.BoolOp):
        op_func = operators[type(node.op)]
        return op_func([_eval_ast(v, local_vars) for v in node.values])
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return operators[ast.Not](_eval_ast(node.operand, local_vars))
    elif isinstance(node, ast.Name):
        return local_vars.get(node.id, 0)
    elif isinstance(node, ast.Constant):
        return node.value
    else:
        raise TypeError(f"Unsupported expression: {ast.dump(node)}")

# ----- Eval-based evaluator -----
def evaluate_function_eval(function, state):
    local_vars = {k: v for k, v in state.items()}
    function = function.replace('&', ' and ').replace('|', ' or ').replace('!', ' not ')
    result = eval(function, {}, local_vars)
    return int(result)

# ----- General evaluator -----
def evaluate_function(function, state, method='safe'):
    if method == 'safe':
        function = function.replace('!', 'not').replace('&', 'and').replace('|', 'or')
        return int(safe_eval_bool_expr(function, state))
    elif method == 'eval':
        return evaluate_function_eval(function, state)
    else:
        raise ValueError(f"Unknown evaluation method: {method}")

def read_bnet_file(bnet_file):
    with open(bnet_file, 'r') as file:
        lines = file.readlines()
    network = {}
    for line in lines[1:]:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split(',')
            node = parts[0].strip()
            function = parts[1].strip()
            network[node] = function
    return network

def extract_id_from_folder_name(folder_name):
    match = re.match(r'\[id-(\d+)]', folder_name)
    if match:
        return int(match.group(1))
    return None

def generate_initial_state(variables):
    return {var: random.choice([0, 1]) for var in variables}

def update_state(state, network, method):
    new_state = state.copy()
    for node in network:
        new_state[node] = evaluate_function(network[node], state, method)
    return new_state

def add_noise(state, noise_rate):
    noisy_state = state.copy()
    for var in noisy_state:
        if random.random() < noise_rate:
            noisy_state[var] = 1 - noisy_state[var]
    return noisy_state

def generate_all_combinations(variables):
    for combination in product([1, 0], repeat=len(variables)):
        yield combination

def generate_truth_tables(bnet_file, out_folder, model_id, input_variables, noise_rate=0.1, method='safe'):
    network = read_bnet_file(bnet_file)
    variables = list(network.keys())
    all_variables = input_variables + variables
    num_inputs = len(input_variables)
    total_variables = len(all_variables)

    target_dir = osp.join(out_folder, f"in{total_variables}_out{total_variables}") #in{num_inputs}
    os.makedirs(target_dir, exist_ok=True)
    out_file = osp.join(target_dir, f"{model_id}.truth")

    data = []

    if len(variables) < 12:
        for combination in tqdm(generate_all_combinations(all_variables), total=2 ** len(all_variables), desc="Exhaustive"):
            state = {var: val for var, val in zip(all_variables, combination)}
            new_state = update_state(state, network, method)
            noisy_state = add_noise(new_state, noise_rate)
            row = list(combination) + list(noisy_state.values())
            data.append(row)
        transposed = list(zip(*data))
        with open(out_file, 'w') as f:
            for row in transposed:
                f.write(''.join(map(str, row)) + '\n')

    else:
        current_state = generate_initial_state(all_variables)
        noisy_state = add_noise(current_state, noise_rate)
        num_steps = 10000
        states = []
        prev = list(noisy_state.values())
        cnt = 0

        for step in tqdm(range(1, num_steps + 1), desc="Simulation"):
            new_state = update_state(current_state, network, method)
            if list(new_state.values()) == list(current_state.values()):
                cnt += 1
                if cnt > 6:
                    new_state = generate_initial_state(all_variables)
                    cnt = 0
            else:
                cnt = 0
            if step % 200 == 0:
                new_state = generate_initial_state(all_variables)
                cnt = 0

            noisy_state = add_noise(new_state, noise_rate)
            curr = list(noisy_state.values())
            prev.extend(curr)
            states.append(prev)
            prev = curr
            current_state = new_state

        transposed = list(zip(*states))
        with open(out_file, 'w') as f:
            for row in transposed:
                f.write(''.join(map(str, row)) + '\n')

    print(f"Saved: {out_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default='safe', choices=['safe', 'eval'])
    args = parser.parse_args()

    models_path = 'models'
    noise_rates = [0.00, 0.01, 0.05]
    out_base_folder = f'./tseq_truth_tables'
    os.makedirs(out_base_folder, exist_ok=True)

    for noise_rate in noise_rates:
        out_folder = osp.join(out_base_folder, f"noise_{int(noise_rate * 100):02d}")
        os.makedirs(out_folder, exist_ok=True)

        cnt = 0
        for folder_name in os.listdir(models_path):
            cnt += 1
            if cnt > 20:  # debug limit
                break

            folder_path = osp.join(models_path, folder_name)
            if os.path.isdir(folder_path):
                model_id = extract_id_from_folder_name(folder_name)
                if model_id is None:
                    continue
                bnet_file = osp.join(folder_path, 'model.bnet')
                sbml_file = osp.join(folder_path, 'model.sbml')
                input_variables = read_sbml(sbml_file)
                generate_truth_tables(bnet_file, out_folder, model_id, input_variables, noise_rate, method=args.eval_mode)

