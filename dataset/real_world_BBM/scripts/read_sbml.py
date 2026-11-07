import os
from biodivine_aeon import *


def read_sbml(file):
    total_regulations = 0
    total_variables = 0
    total_non_inputs = 0
    total_inputs = 0
    total_outputs = 0

    input_variables = set()
    output_variables = set()

    try:
        bn = BooleanNetwork.from_file('./' + file)
        print("Successfully loaded.")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return []

    for var in bn.variables():
        total_variables += 1
        regs = bn.predecessors(var)  # Get the predecessors (regulators) of the variable
        total_regulations += len(regs)

        # If the variable has no regulators, it is considered an input variable
        if len(regs) == 0:
            total_inputs += 1
            name = bn.get_variable_name(var)
            input_variables.add(name)
        else:
            total_non_inputs += 1

        # If the variable has no successors, it is considered an output variable
        if len(bn.successors(var)) == 0:
            total_outputs += 1
            name = bn.get_variable_name(var)
            output_variables.add(name)

    # Print a summary of the Boolean network
    print(
        f"{file}: {total_variables} variables "
        f"({total_inputs} inputs, {total_non_inputs} non-inputs, {total_outputs} outputs), "
        f"{total_regulations} total regulations."
    )

    return list(input_variables)


if __name__ == '__main__':
    for file in os.listdir(''):
        if not file.endswith('.sbml'):
            continue
        inputs = read_sbml(file)
        print('Input variables:')
        for name in sorted(inputs):
            print(name)

