import itertools


def read_cnf_file(file_path):
    """
    Reads a CNF file and extracts the clauses.

    Args:
    file_path (str): The path to the CNF file.

    Returns:
    list: A list of clauses, where each clause is a list of literals.
    """
    clauses = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c') or line.startswith('p'):
                continue  # Skip comment and problem description lines
            clause = [int(literal) for literal in line.split() if int(literal) != 0]
            clauses.append(clause)
    return clauses


def evaluate_clause(clause, assignment):
    """
    Evaluates a clause based on the given assignment.

    Args:
    clause (list): A list of literals representing the clause.
    assignment (list): A list of boolean values representing the variable assignment.

    Returns:
    int: 1 if the clause is true, 0 otherwise.
    """
    for literal in clause:
        var = abs(literal)
        if (literal > 0 and assignment[var - 1]) or (literal < 0 and not assignment[var - 1]):
            return 1  # Clause is true
    return 0  # Clause is false


def generate_truth_table(clauses, num_vars):
    """
    Generates the truth table for the given clauses.

    Args:
    clauses (list): A list of clauses.
    num_vars (int): The number of variables.

    Returns:
    list: The truth table, where each row is a list of variable assignments followed by clause evaluations.
    """
    truth_table = []
    print(f"Number of clauses: {len(clauses)}")
    # Generate all possible assignments of input variables
    for assignment in itertools.product([0, 1], repeat=num_vars):
        row = list(assignment)  # Values of input variables
        for clause in clauses:
            row.append(evaluate_clause(clause, assignment))
        truth_table.append(row)
    return truth_table


def save_truth_table(truth_table, filename):
    """
    Saves the truth table to a file.

    Args:
    truth_table (list): The truth table to save.
    filename (str): The path to the output file.
    """
    with open(filename, 'w') as f:
        for row in truth_table:
            f.write(''.join(map(str, row)) + '\n')


def main():
    """
    Main function to read a CNF file, generate the truth table, and save it to a file.
    """
    cnf_file_path = "../data/c139.cnf"  # Replace with the actual CNF file path
    output_file_path = "tt.truth"  # Output file path

    # Read clauses from the CNF file
    clauses = read_cnf_file(cnf_file_path)
    # Calculate the number of variables (assuming variable indices start from 1)
    num_vars = max(max(abs(literal) for literal in clause) for clause in clauses)

    # Generate the truth table
    truth_table = generate_truth_table(clauses, num_vars)
    # Save the truth table to a file
    save_truth_table(truth_table, output_file_path)
    print(f"Truth table saved to {output_file_path}")


if __name__ == "__main__":
    main()