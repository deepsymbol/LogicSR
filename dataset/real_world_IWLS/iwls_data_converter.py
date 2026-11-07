#!/usr/bin/env python3
"""
IWLS Truth Table Format Converter
Convert IWLS contest truth table format to our format and automatically infer input/output dimensions
"""

import math
import argparse
from pathlib import Path


def analyze_iwls_truth_table(file_path):
    """Analyze IWLS truth table file and infer input/output dimensions"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return None, None, None

    # Each line represents a truth table for one output
    num_outputs = len(lines)
    truth_table_length = len(lines[0])

    # Infer input dimension from truth table length: 2^n = truth_table_length
    num_inputs = int(math.log2(truth_table_length))

    if 2**num_inputs != truth_table_length:
        print(f"Warning: Truth table length {truth_table_length} is not a power of 2")
        return None, None, None

    print(f"File: {file_path}")
    print(f"  Input dimension: {num_inputs}")
    print(f"  Output dimension: {num_outputs}")
    print(f"  Truth table length: {truth_table_length}")

    return num_inputs, num_outputs, lines


def convert_iwls_to_our_format(iwls_lines, num_inputs, num_outputs):
    """Convert IWLS format to our format - input order with 1 at the beginning"""
    truth_table_length = len(iwls_lines[0])

    # Generate input variable rows (1 at the beginning order)
    input_lines = []
    for i in range(num_inputs):
        input_line = ""
        for j in range(truth_table_length):
            # Forward arrangement: from small to large
            # The i-th input variable (from high to low bit)
            cycle_length = 2**(num_inputs - 1 - i)
            bit_value = (j // cycle_length) % 2
            input_line += str(bit_value)
        # Reverse the binary string to put 1 at the beginning
        reversed_input_line = input_line[::-1]
        input_lines.append(reversed_input_line)

    # Output rows also need to be reversed to put 1 at the beginning
    output_lines = []
    for output_line in iwls_lines:
        # Reverse the output row to put 1 at the beginning
        reversed_output_line = output_line[::-1]
        output_lines.append(reversed_output_line)

    # Combine input and output rows
    result_lines = input_lines + output_lines

    return result_lines


def convert_iwls_directory(input_dir, output_dir):
    """Convert entire IWLS directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Group by input/output dimensions
    dimension_groups = {}

    # Recursively find all .truth files
    truth_files = list(input_path.rglob("*.truth"))
    print(f"Found {len(truth_files)} truth table files")

    for truth_file in truth_files:
        num_inputs, num_outputs, iwls_lines = analyze_iwls_truth_table(
            truth_file)

        if num_inputs is None:
            print(f"Skipping file: {truth_file}")
            continue

        # Convert format
        converted_lines = convert_iwls_to_our_format(iwls_lines, num_inputs,
                                                     num_outputs)

        # Group by dimensions
        key = f"in{num_inputs}_out{num_outputs}"
        if key not in dimension_groups:
            dimension_groups[key] = []

        dimension_groups[key].append({
            'original_name': truth_file.name,
            'converted_lines': converted_lines
        })

    # Save grouped files
    for key, files in dimension_groups.items():
        group_dir = output_path / key
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing group: {key} ({len(files)} files)")

        for idx, file_info in enumerate(files, 1):
            output_file = group_dir / f"{idx}.truth"
            with open(output_file, 'w') as f:
                for line in file_info['converted_lines']:
                    f.write(line + '\n')

            print(f"  {file_info['original_name']} -> {output_file}")

    print(f"\nConversion completed! Output directory: {output_dir}")
    print("Dimension grouping statistics:")
    for key, files in dimension_groups.items():
        print(f"  {key}: {len(files)} files")


def convert_existing_data_to_new_format(data_dir):
    """Convert existing data to new format (change from 0 at beginning to 1 at beginning)"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Data directory does not exist: {data_dir}")
        return

    # Traverse all truth files
    for truth_file in data_path.rglob("*.truth"):
        print(f"Converting file: {truth_file}")

        # Read original content
        with open(truth_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Reverse each line (change from 0 at beginning to 1 at beginning)
        reversed_lines = [line[::-1] for line in lines]

        # Write back to file
        with open(truth_file, 'w') as f:
            for line in reversed_lines:
                f.write(line + '\n')

    print("Existing data conversion completed!")


def main():
    parser = argparse.ArgumentParser(description='IWLS Truth Table Format Converter')
    parser.add_argument('input_dir', help='IWLS truth table file directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--convert-existing',
                        action='store_true',
                        help='Convert existing data format (change from 0 at beginning to 1 at beginning)')

    args = parser.parse_args()

    if args.convert_existing:
        convert_existing_data_to_new_format(args.input_dir)
    else:
        convert_iwls_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
