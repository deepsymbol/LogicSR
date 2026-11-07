import argparse
import struct
import gzip
import os

def read_aiger_ascii(file_path):
    """
    read AIGER files in ASCII format.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    header = lines[0].strip().split()
    if header[0] != "aag":
        raise ValueError("Invalid AIGER header")

    M, I, L, O, A = map(int, header[1:])
    inputs = []
    latches = []
    outputs = []
    and_gates = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("i"):
            inputs.append(line)
        elif line.startswith("l"):
            latches.append(line)
        elif line.startswith("o"):
            outputs.append(line)
        elif line.startswith("a"):
            and_gates.append(line)

    return {
        "M": M,
        "I": I,
        "L": L,
        "O": O,
        "A": A,
        "inputs": inputs,
        "latches": latches,
        "outputs": outputs,
        "and_gates": and_gates
    }

def write_aiger_ascii(data, file_path):
    """
    write AIGER data to files in ASCII format.
    """
    with open(file_path, "w") as file:
        file.write(f"aag {data['M']} {data['I']} {data['L']} {data['O']} {data['A']}\n")
        for line in data["inputs"]:
            file.write(line + "\n")
        for line in data["latches"]:
            file.write(line + "\n")
        for line in data["outputs"]:
            file.write(line + "\n")
        for line in data["and_gates"]:
            file.write(line + "\n")

def read_aiger_binary(file_path):
    """
    Read an AIGER file in binary format.
    """
    with open(file_path, "rb") as file:
        header = file.readline().decode("utf-8").strip().split()
        if header[0] != "aig":
            raise ValueError("Invalid AIGER header")

        M, I, L, O, A = map(int, header[1:])
        inputs = []
        latches = []
        outputs = []
        and_gates = []

        for _ in range(I):
            inputs.append(struct.unpack('!I', file.read(4))[0])
        for _ in range(L):
            latches.append(struct.unpack('!I', file.read(4))[0])
        for _ in range(O):
            outputs.append(struct.unpack('!I', file.read(4))[0])
        for _ in range(A):
            delta0 = struct.unpack('!I', file.read(4))[0]
            delta1 = struct.unpack('!I', file.read(4))[0]
            lhs = delta0 + delta1
            rhs0 = delta1
            rhs1 = rhs0 - delta1
            and_gates.append((lhs, rhs0, rhs1))

    return {
        "M": M,
        "I": I,
        "L": L,
        "O": O,
        "A": A,
        "inputs": inputs,
        "latches": latches,
        "outputs": outputs,
        "and_gates": and_gates
    }

def write_aiger_binary(data, file_path):
    """
    write AIG data to binary files.
    """
    with open(file_path, "wb") as file:
        file.write(f"aig {data['M']} {data['I']} {data['L']} {data['O']} {data['A']}\n".encode("utf-8"))
        for input_ in data["inputs"]:
            file.write(struct.pack('!I', input_))
        for latch in data["latches"]:
            file.write(struct.pack('!I', latch))
        for output in data["outputs"]:
            file.write(struct.pack('!I', output))
        for lhs, rhs0, rhs1 in data["and_gates"]:
            delta0 = lhs - rhs0
            delta1 = rhs0 - rhs1
            file.write(struct.pack('!I', delta0))
            file.write(struct.pack('!I', delta1))

def strip_symbols_and_comments(data):
    """
    Remove symbol tables and comments.
    """
    data["inputs"] = [line for line in data["inputs"] if not line.startswith("c")]
    data["latches"] = [line for line in data["latches"] if not line.startswith("c")]
    data["outputs"] = [line for line in data["outputs"] if not line.startswith("c")]
    data["and_gates"] = [line for line in data["and_gates"] if not line.startswith("c")]

def remove_outputs(data):
    """
    Remove outputs.
    """
    data["outputs"] = []
    data["O"] = 0

def main():
    parser = argparse.ArgumentParser(description="AIGER file converter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-s", "--strip", action="store_true", help="Strip symbols and comments")
    parser.add_argument("-a", "--ascii", action="store_true", help="Output in ASCII AIGER format")
    parser.add_argument("-b", "--binary", action="store_true", help="Output in binary AIGER format")
    parser.add_argument("-r", "--remove-outputs", action="store_true", help="Remove outputs")
    parser.add_argument("src", help="Source file")
    parser.add_argument("dst", nargs="?", help="Destination file (optional)")

    args = parser.parse_args()

    if args.ascii and args.binary:
        raise ValueError("Cannot specify both --ascii and --binary")

    if args.dst is None:
        args.dst = args.src

    if args.src.endswith(".gz"):
        with gzip.open(args.src, "rt") as file:
            content = file.read()
        src_data = read_aiger_ascii(content)
    else:
        src_data = read_aiger_ascii(args.src)

    if args.strip:
        strip_symbols_and_comments(src_data)

    if args.remove_outputs:
        remove_outputs(src_data)

    if args.dst.endswith(".gz"):
        with gzip.open(args.dst, "wt") as file:
            if args.ascii:
                write_aiger_ascii(src_data, file)
            else:
                write_aiger_binary(src_data, file)
    else:
        if args.ascii:
            write_aiger_ascii(src_data, args.dst)
        else:
            write_aiger_binary(src_data, args.dst)

    if args.verbose:
        print(f"Converted {args.src} to {args.dst}")


if __name__ == "__main__":
    main()


