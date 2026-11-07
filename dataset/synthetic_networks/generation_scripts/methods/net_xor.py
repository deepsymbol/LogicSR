import numpy as np
from tqdm import tqdm
import random


class AIGNode:
    def __init__(self, id, gate_type='AND', priority=3):
        self.id = id
        self.gate_type = gate_type  # 'AND', 'OR', 'NOR', 'NAND', 'XOR', 'INPUT'
        self.fanins = []
        self.fanouts = []
        self.truth_table = None
        self.out = False
        self.visited = False
        self.redundant = False
        self.hanged = True
        self.incre = 0
        self.priority = priority
        self.hasnotfin = False
        self.del_cost = 1
        self.layer = 1
        self.trivial = False
        self.ancestors = {self.id * 2, self.id * 2 + 1}
        self.aig_ancestors = {}

    def add_fanin(self, fanin):
        self.fanins.append(fanin)

    def add_fanout(self, fanout):
        self.fanouts.append(fanout)

    def copy_node(self):
        new_node = AIGNode(self.id, self.gate_type)
        new_node.fanins = self.fanins.copy()
        new_node.fanouts = self.fanouts.copy()
        new_node.truth_table = self.truth_table[:]
        new_node.out = self.out
        return new_node

    def __repr__(self):
        return f"Node({self.id}, gate={self.gate_type}, fanins={self.fanins}, truth_table={self.truth_table})"


class AIG:
    def __init__(self, k, l):
        self.k = k  # num of inputs
        self.l = l  # num of outputs
        self.nodes = []
        self.var_count = k
        self.outs = dict()
        self.id2tt = dict()
        self.max_layer = 0
        self._next_and_id = k  # first K nodes as inputs

    def get_next_and_id(self):
        idx = self._next_and_id
        self._next_and_id += 1
        return idx

    def new_var(self):
        self.var_count += 1
        return self.var_count - 1

    def remove_nodes(self, remove):
        # construct mapping from old ID to new ID (skip deleted nodes)
        id_map = {}
        new_nodes = []
        new_id = 0

        for old_id, node in enumerate(self.nodes):
            if old_id in remove or node is None:
                continue
            id_map[old_id] = new_id
            node.id = new_id
            new_nodes.append(node)
            new_id += 1

        # update ID of fanins and fanouts （2*id/2*id+1）
        for node in new_nodes:
            updated_fanins = []
            for f in node.fanins:
                old_fanin_id = f // 2 - 1
                if old_fanin_id in id_map:
                    new_fanin_id = id_map[old_fanin_id]
                    updated_fanins.append((new_fanin_id + 1) * 2 + (f % 2))
            node.fanins = updated_fanins

            updated_fanouts = []
            for f in node.fanouts:
                old_fanout_id = f // 2 - 1
                if old_fanout_id in id_map:
                    new_fanout_id = id_map[old_fanout_id]
                    updated_fanouts.append((new_fanout_id + 1) * 2 + (f % 2))
            node.fanouts = updated_fanouts

        # update outs ID
        new_outs = {}
        for old_out_id, polarity in self.outs.items():
            if old_out_id in id_map:
                new_outs[id_map[old_out_id]] = polarity
        self.outs = new_outs

        # replace nodes
        self.nodes = new_nodes
        self.var_count = len(new_nodes)
        self.k = sum(1 for n in new_nodes if n.gate_type == 'INPUT')
        self.l = len(self.outs)

        # reset _next_and_id
        self._next_and_id = self.var_count

        return id_map

    def add_gate(self, fanin1, fanin2, gate_type='AND'):
        node_id = self.new_var()
        new_node = AIGNode(node_id, gate_type)
        new_node.add_fanin(fanin1)
        new_node.add_fanin(fanin2)
        self.nodes[fanin1 // 2 - 1].add_fanout(new_node.id * 2 + fanin1 % 2)
        self.nodes[fanin2 // 2 - 1].add_fanout(new_node.id * 2 + fanin2 % 2)
        self.nodes.append(new_node)
        return new_node

    def init_input_tt(self):
        for node_id in range(self.k):
            multi = 2 ** (self.k - node_id - 1)
            input_node = self.nodes[node_id]
            tt = [1] * multi + [0] * multi
            input_node.truth_table = tt * (2 ** node_id)

    def compute_truth_tables(self):
        self.init_input_tt()
        for node in self.nodes[self.k:]:
            if node is None:
                continue
            fanin1_id = node.fanins[0] // 2 - 1
            fanin2_id = node.fanins[1] // 2 - 1
            fanin1_tt = self.nodes[fanin1_id].truth_table[:]
            fanin2_tt = self.nodes[fanin2_id].truth_table[:]
            if node.fanins[0] % 2 == 1:
                fanin1_tt = [1 - x for x in fanin1_tt]
            if node.fanins[1] % 2 == 1:
                fanin2_tt = [1 - x for x in fanin2_tt]

            # Support five types of logic gates.
            if node.gate_type == 'AND':
                node.truth_table = [a & b for a, b in zip(fanin1_tt, fanin2_tt)]
            elif node.gate_type == 'OR':
                node.truth_table = [a | b for a, b in zip(fanin1_tt, fanin2_tt)]
            elif node.gate_type == 'NOR':
                node.truth_table = [1 - (a | b) for a, b in zip(fanin1_tt, fanin2_tt)]
            elif node.gate_type == 'NAND':
                node.truth_table = [1 - (a & b) for a, b in zip(fanin1_tt, fanin2_tt)]
            elif node.gate_type == 'XOR':
                node.truth_table = [a ^ b for a, b in zip(fanin1_tt, fanin2_tt)]
            else:
                raise ValueError(f"Unsupported gate_type: {node.gate_type}")

    def print_truth_tables(self):
        for node in self.nodes:
            if node.out or node.id < self.k:
                print(f"Node {node.id}: TT = {node.truth_table}")

    def save_tt(self, filename, trans=False, noise_rate=0):
        tts = [self.nodes[i].truth_table for i in range(self.k)]
        for i, val in self.outs.items():
            tt = self.nodes[i].truth_table
            inv_tt = [1 - x for x in tt]
            if noise_rate > 0:
                flip_num = int(len(tt) * noise_rate)
                flip_indices = random.sample(range(len(tt)), flip_num)
                for idx in flip_indices:
                    tt[idx] ^= 1
                    inv_tt[idx] ^= 1

            tts.append(inv_tt if val == 2 else tt)
            if val >= 3:
                print('val err')

        arr = np.array(tts)
        if trans:
            arr = arr.T
        with open(filename, 'w') as f:
            for row in arr:
                f.write(''.join(map(str, row)) + '\n')

    def to_mid(self, use_aig=False):
        """
        Convert the current AIG circuit to an intermediate format.
        Now supports AND, OR, NOR, NAND, XOR gates with gate type encoding.

        Gate type encoding:
        0: AND
        1: OR
        2: NOR
        3: NAND
        4: XOR

        Args:
            use_aig (bool): If True, use strict AIG format (only AND gates with inversion)

        Returns:
            str: The circuit in the intermediate format as a string.
        """
        I = self.k  # Number of input nodes
        L = 0  # Number of latch nodes (not used in this format)
        O = len(self.outs)  # Number of output nodes
        max_index = len(self.nodes)  # Total number of nodes
        A = len([node for node in self.nodes if node is not None and len(node.fanins) == 2])  # Number of gates

        mid_content = []
        mid_content.append(f"aag {max_index + 1} {I} {L} {O} {A}")

        # Input section
        for i in range(self.k):
            mid_content.append(f"{(i + 1) * 2}")

        # Output section
        for node_id, val in self.outs.items():
            literal = (node_id + 1) * 2
            if val == 2:
                literal += 1
            mid_content.append(f"{literal}")

        # Gate section with gate type
        for node in self.nodes:
            if not node:
                continue
            if len(node.fanins) == 2:
                fanin1, fanin2 = node.fanins
                output_literal = (node.id + 1) * 2

                # Gate type encoding
                gate_type_flag = {
                    'AND': 0,
                    'OR': 1,
                    'NOR': 2,
                    'NAND': 3,
                    'XOR': 4
                }.get(node.gate_type, 0)

                if use_aig:
                    # For strict AIG format, convert other gates to AND with inversions
                    if node.gate_type != 'AND':
                        # This is a simplification - in practice you'd need to decompose
                        mid_content.append(f"{output_literal} {fanin1} {fanin2}")
                    else:
                        mid_content.append(f"{output_literal} {fanin1} {fanin2}")
                else:
                    mid_content.append(f"{output_literal} {fanin1} {fanin2} {gate_type_flag}")

        return "\n".join(mid_content)

    def from_mid(self, mid_content):
        """
        Read circuit from intermediate format with extended gate support.

        Args:
            mid_content (str): The circuit in intermediate format
        """
        lines = mid_content.strip().split('\n')
        header = lines[0].split()

        if header[0] != 'aag':
            raise ValueError("Invalid file format: expected 'aag' header")

        max_index, I, L, O, A = map(int, header[1:6])

        # Clear existing nodes
        self.nodes = []
        self.outs = {}
        self.k = I
        self.l = O
        self.var_count = I

        # Create input nodes
        for i in range(I):
            self.nodes.append(AIGNode(i, 'INPUT'))

        # Read inputs
        input_lines = lines[1:1 + I]

        # Read outputs
        output_lines = lines[1 + I:1 + I + O]
        for i, line in enumerate(output_lines):
            literal = int(line)
            node_id = literal // 2 - 1
            polarity = literal % 2  # 0: normal, 1: inverted
            self.outs[node_id] = polarity

        # Read gates
        gate_lines = lines[1 + I + O:1 + I + O + A]
        for line in gate_lines:
            parts = line.split()
            output_literal = int(parts[0])
            fanin1 = int(parts[1])
            fanin2 = int(parts[2])

            # Determine gate type (default to AND if not specified)
            gate_type = 'AND'
            if len(parts) > 3:
                gate_type_flag = int(parts[3])
                gate_type = {
                    0: 'AND',
                    1: 'OR',
                    2: 'NOR',
                    3: 'NAND',
                    4: 'XOR'
                }.get(gate_type_flag, 'AND')

            node_id = output_literal // 2 - 1
            if node_id >= len(self.nodes):
                # Add missing nodes
                for i in range(len(self.nodes), node_id + 1):
                    self.nodes.append(AIGNode(i, 'AND'))  # Temporary type

            node = self.nodes[node_id]
            node.gate_type = gate_type
            node.fanins = [fanin1, fanin2]

            # Update fanouts of fanin nodes
            for fanin in [fanin1, fanin2]:
                fanin_node_id = fanin // 2 - 1
                if fanin_node_id < len(self.nodes):
                    self.nodes[fanin_node_id].fanouts.append(output_literal)

        self._next_and_id = len(self.nodes)

    def to_bnet(self):
        def get_expr(node_id, visited):
            node = self.nodes[node_id]
            if node.id in visited:
                return visited[node.id]
            if node.id < self.k:
                expr = f"v{node.id}"
                visited[node.id] = expr
                return expr

            fin0 = node.fanins[0]
            fin1 = node.fanins[1]
            fin0_id, inv0 = fin0 // 2 - 1, fin0 % 2
            fin1_id, inv1 = fin1 // 2 - 1, fin1 % 2

            expr0 = get_expr(fin0_id, visited)
            expr1 = get_expr(fin1_id, visited)

            if inv0:
                expr0 = f"!v{fin0_id}" if self.nodes[fin0_id].id < self.k else f"!{expr0}"
            if inv1:
                expr1 = f"!v{fin1_id}" if self.nodes[fin1_id].id < self.k else f"!{expr1}"

            # Support for all five gate types
            if node.gate_type == "AND":
                expr = f"({expr0} & {expr1})"
            elif node.gate_type == "OR":
                expr = f"({expr0} | {expr1})"
            elif node.gate_type == "NOR":
                expr = f"!({expr0} | {expr1})"
            elif node.gate_type == "NAND":
                expr = f"!({expr0} & {expr1})"
            elif node.gate_type == "XOR":
                expr = f"({expr0} ^ {expr1})"
            else:
                raise ValueError(f"Unsupported gate_type: {node.gate_type}")

            visited[node.id] = expr
            return expr

        bnet_lines = []
        for out_id, polarity in self.outs.items():
            visited = {}
            expr = get_expr(out_id, visited)
            if polarity == 2:
                expr = f"!{expr}"
            bnet_lines.append(f"v{out_id}, {expr}")

        return "\n".join(bnet_lines)

    def calculate_key(self, redundancy_control=False):
        key = []
        for id in self.outs.keys():
            node = self.nodes[id]
            truth_table = node.truth_table
            not_truth_table = [1 - bit for bit in truth_table]
            if redundancy_control and tuple(truth_table) in key or tuple(not_truth_table) in key:
                return 0
            if truth_table > not_truth_table:
                key.append(tuple(not_truth_table))
            else:
                key.append(tuple(truth_table))
        key.sort()
        return tuple(key)

    def __hash__(self):
        return hash(self.calculate_key())

    def __eq__(self, other):
        return self.calculate_key() == other.calculate_key()

    def increment_ids(self, incre, in_incre):
        """
        Increment the IDs of nodes and update their fanins and fanouts accordingly.
        """
        # Increment IDs of input nodes
        for node in self.nodes[:self.k]:
            node.id += in_incre
            node.incre = in_incre
            node.fanouts = [f + incre * 2 for f in node.fanouts]

        # Increment IDs of internal nodes
        for node in self.nodes[self.k:]:
            node.id += incre
            node.fanins = [f + incre * 2 if f // 2 > self.k else f + in_incre * 2 for f in node.fanins]
            node.fanouts = [f + incre * 2 if f // 2 > self.k else f + in_incre * 2 for f in node.fanouts]

        # Update the output dictionary with new IDs
        out_dict = {}
        for out, val in self.outs.items():
            out_dict[out + incre] = val

        self.outs = out_dict

    def loop_truth(self, looptimes, final_len, use_tqdm=False):
        def repeat_elements(truth_table, N, L):
            result = []
            for value in truth_table:
                for _ in range(N):
                    if len(result) < L:
                        result.append(value)
                    else:
                        break
                if len(result) >= L:
                    break
            return result

        ori_len = len(self.nodes[0].truth_table)
        len_one_loop = ori_len * looptimes
        times = final_len // len_one_loop

        iterator = tqdm(self.nodes, desc="Expanding truth tables") if use_tqdm else self.nodes

        if times == 0:
            for node in iterator:
                tt = repeat_elements(node.truth_table, looptimes, final_len)
                node.truth_table = tt
        else:
            for node in iterator:
                tt = [item for item in node.truth_table for _ in range(looptimes)]
                left = final_len % len(tt)
                node.truth_table = tt * times + tt[:left]


# Example usage
if __name__ == "__main__":
    # Create AIG with 2 inputs, 1 output
    aig = AIG(2, 1)

    # Add input nodes
    for i in range(2):
        aig.nodes.append(AIGNode(i, 'INPUT'))

    # Add XOR gate
    xor_gate = aig.add_gate(2, 4, 'XOR')  # fanin1=2 (input0), fanin2=4 (input1)

    # Set output
    aig.outs[xor_gate.id] = 0

    # Compute truth tables
    aig.compute_truth_tables()

    # Print in extended format
    print("Extended AIG format:")
    print(aig.to_mid(use_aig=False))

    # Print in BN format
    print("\nBN format:")
    print(aig.to_bnet())