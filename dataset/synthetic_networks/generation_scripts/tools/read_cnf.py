import itertools
import csv

def parse_cnf_file(path):
    """
    Parse a CNF file in DIMACS format.

    Args:
        path (str): Path to the CNF file.

    Returns:
        tuple: (num_vars, clauses)
            - num_vars (int): Number of variables.
            - clauses (list): List of clauses, each clause is a list of literals.
    """
    num_vars = 0
    clauses = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue  # Skip comments and empty lines
            if line.startswith('p cnf'):
                _, _, num_vars_str, _ = line.split()
                num_vars = int(num_vars_str)
                continue
            # Read clause (ends with 0)
            clause = list(map(int, line.split()))
            if clause[-1] == 0:
                clause.pop()
            clauses.append(clause)

    return num_vars, clauses


def is_clause_satisfied(clause, assignment):
    """
    Check if a single clause is satisfied under the given assignment.

    Args:
        clause (list): List of integers representing literals.
        assignment (dict): Variable assignment {var_index: bool}.

    Returns:
        bool: True if the clause is satisfied, False otherwise.
    """
    for literal in clause:
        var = abs(literal)
        value = assignment[var]
        if (literal > 0 and value) or (literal < 0 and not value):
            return True
    return False


def is_formula_satisfied(clauses, assignment):
    """
    Check if all clauses are satisfied under the given assignment.

    Args:
        clauses (list): List of clauses.
        assignment (dict): Variable assignment.

    Returns:
        bool: True if the formula is satisfied, False otherwise.
    """
    return all(is_clause_satisfied(clause, assignment) for clause in clauses)


def generate_truth_table(num_vars, clauses):
    """
    Generate the truth table for the formula.

    Args:
        num_vars (int): Number of variables.
        clauses (list): List of clauses.

    Returns:
        list: Each row is a list of Boolean variable values followed by the satisfaction result.
    """
    truth_table = []
    for values in itertools.product([False, True], repeat=num_vars):
        assignment = {i + 1: val for i, val in enumerate(values)}
        result = is_formula_satisfied(clauses, assignment)
        truth_table.append(list(values) + [result])
    return truth_table


def save_truth_table(truth_table, output_path):
    """
    Save the truth table to a CSV file.

    Args:
        truth_table (list): The truth table to save.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'Var{i + 1}' for i in range(len(truth_table[0]) - 1)] + ['Satisfies'])
        writer.writerows(truth_table)


# ===== Example usage =====
cnf_file = '../data/c139.cnf'  # Replace with your CNF file path
output_csv = 'truth_table.csv'

num_vars, clauses = parse_cnf_file(cnf_file)
print(len(clauses))

truth_table = generate_truth_table(num_vars, clauses)
save_truth_table(truth_table, output_csv)

print(f"Saved truth table with {len(truth_table)} entries to {output_csv}")
