from methods.net import AIG,AIGNode
from tools.vis_net import vis_aig
import os
import os.path as osp


def read_aag(file_path):
    """
    Read an ASCII AIGER file and convert it into an AIG object.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the AIGER file header.
    header = lines[0].strip().split()
    if header[0].lower() != 'aag':
        raise ValueError("Invalid AIGER file format. Expected 'aag' at the beginning.")

    max_index = int(header[1])  # Maximum variable index.
    I = int(header[2])  # Number of input nodes.
    L = int(header[3])  # Number of latch nodes (not currently supported).
    O = int(header[4])  # Number of output nodes.
    A = int(header[5])  # Number of AND gates.

    # Initialize the AIG object.
    # print(max_index,I,L,O,A)
    aig = AIG(k=I, l=O)
    aig.var_count = max_index//2
    out_ids = []
    outs = dict()

    for line in lines[1:1+I]:
        id = int(line.strip())//2 -1
        node = AIGNode(id,'INPUT')
        aig.nodes.append(node)

    for line in lines[1+I:O+I+1]:
        parts = line.strip().split()[0]
        id = int(parts)//2 -1
        out_ids.append(id)
        if id in outs.keys():
            outs[id] += (int(parts)%2 + 1) # 1 for normal, 2 for inverted, and 3 for both.
        else:
            outs[id] = int(parts)%2 + 1

    aig.outs = outs
    and_ids = []
    and_ids = list(set(range(I,max_index-O)) - set(and_ids))
    for line in lines[1 + I + O:]:
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid AND gate definition: {line.strip()}")

        fanout = int(parts[0]) // 2 - 1
        fanin1 = int(parts[1])
        fanin2 = int(parts[2])
        node = AIGNode(fanout,'AND')
        node.add_fanin(fanin1)
        node.add_fanin(fanin2)
        # print(fanin2//2-1,fanin1//2-1,fanin2//2-1,node.id,len(aig.nodes))
        aig.nodes[fanin1//2-1].add_fanout(node.id+fanin1%2)
        aig.nodes[fanin2//2-1].add_fanout(node.id+fanin2%2)
        aig.nodes.append(node)
        # Create the AND node and add its fanins; if it is an output, connect the AND node to that output.
        if fanout in out_ids:
            node.out = True
    return aig


def read_aon(file_path):
    """
    Read an ASCII AIGER file and convert it into an AIG object.
    Supports four-element gate records: output, input 1, input 2, gate_type (0=AND, 1=OR).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the AIGER file header.
    header = lines[0].strip().split()
    if header[0].lower() != 'aag':
        raise ValueError("Invalid AIGER file format. Expected 'aag' at the beginning.")

    max_index = int(header[1])  # Maximum variable index.
    I = int(header[2])          # Number of input nodes.
    L = int(header[3])          # Number of latch nodes (not currently supported).
    O = int(header[4])          # Number of output nodes.
    A = int(header[5])          # Number of AND/OR gates.

    # Initialize the AIG object.
    aig = AIG(k=I, l=O)
    aig.var_count = max_index // 2
    out_ids = []
    outs = {}

    # Read input nodes.
    for line in lines[1:1 + I]:
        id = int(line.strip()) // 2 - 1
        node = AIGNode(id,'INPUT')
        aig.nodes.append(node)

    # Read output information.
    for line in lines[1 + I:1 + I + O]:
        lit = int(line.strip())
        id = lit // 2 - 1
        inv = lit % 2

        out_ids.append(id)
        if id in outs:
            outs[id] += (inv + 1)
        else:
            outs[id] = inv + 1
    aig.outs = outs

    # Read AND/OR gate definitions in four-element format.
    for line in lines[1 + I + O:]:
        parts = line.strip().split(' ')
        if len(parts) != 4:
            raise ValueError(f"Expected 4 elements per gate line, got {len(parts)}: {line.strip()}")

        fanout = int(parts[0]) // 2 - 1
        fanin1 = int(parts[1])
        fanin2 = int(parts[2])
        gate_type_flag = int(parts[3])

        node = AIGNode(fanout)
        node.gate_type = "OR" if gate_type_flag == 1 else "AND"
        node.add_fanin(fanin1)
        node.add_fanin(fanin2)

        # Add fanout information to the corresponding input nodes.
        aig.nodes[fanin1 // 2 - 1].add_fanout(node.id + fanin1 % 2)
        aig.nodes[fanin2 // 2 - 1].add_fanout(node.id + fanin2 % 2)

        aig.nodes.append(node)

        if fanout in out_ids:
            node.out = True

    return aig


def main():
    dir_path = "generated_aigs/in10_out10/and40"
    aag_path = os.path.join(dir_path,'aag')
    pic_dir = osp.join(dir_path, "pic")
    os.makedirs(pic_dir,exist_ok=True)
    for aig_file_path in os.listdir(aag_path):
        if not aig_file_path.endswith(".aag"):
            continue
        file_path = osp.join(aag_path,aig_file_path)
        pic_name = os.path.basename(aig_file_path).split(".")[0]
        pic_path = osp.join(pic_dir,pic_name)

        aag = read_aag(file_path)
        vis_aig(aag,pic_path,view=True,fmt='png')
        break


if __name__=='__main__':
    main()

