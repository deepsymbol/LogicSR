import random
import numpy as np
from net_xor import AIGNode, AIG
import math
import json
import time
import os
from datetime import datetime


def record_time(subdir, config_key, elapsed_time, config_type, time_file='time_record.json'):
    """Record experiment runtime in a JSON file."""
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes} min {seconds} sec"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(time_file):
        with open(time_file, 'r', encoding='utf-8') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = {}
    else:
        records = {}

    # Ensure the subdirectory exists.
    if subdir not in records:
        records[subdir] = {}

    # Append a record instead of overwriting when the config key already exists.
    if config_key in records[subdir]:
        # Convert the existing single record to a list first, if needed.
        if not isinstance(records[subdir][config_key], list):
            records[subdir][config_key] = [records[subdir][config_key]]

        # Append the new record to the list.
        records[subdir][config_key].append({
            'time': time_str,
            'seconds': round(elapsed_time, 2),
            'timestamp': timestamp,
            'folder': config_type
        })
    else:
        # Create a new record if none exists.
        records[subdir][config_key] = {
            'time': time_str,
            'seconds': round(elapsed_time, 2),
            'timestamp': timestamp,
            'folder': config_type
        }

    with open(time_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"✓ Time record saved: {time_str}")


def filter_pool(pool, priorities, k, step, Mstep):
    """
    Dynamic layer control:
    - Early: restrict layer <= MAX_LAYER
    - Later: gradually relax the limit as step → Mstep
    """
    # Minimal depth ~ log2(k)
    min_depth = math.ceil(math.log2(k + 1))
    # Reasonable max depth ~ sqrt(Mstep) or log2(k+Mstep)
    max_depth_est = int(min(math.sqrt(Mstep), math.log2(k + Mstep)))
    MAX_LAYER = max(min_depth, max_depth_est)

    # Relaxation factor: 0 → strict, 1 → fully relaxed
    relax_ratio = step / Mstep
    dynamic_max = MAX_LAYER + int(relax_ratio * MAX_LAYER)

    new_pool, new_pri = [], []
    for n, p in zip(pool, priorities):
        if n.layer <= dynamic_max:
            new_pool.append(n)
            new_pri.append(p)

    return new_pool, new_pri


def compute_truth(node, fanin1, fanin2, inv1, inv2):
    fanin1_tt = np.array(fanin1.truth_table, dtype=np.uint8)
    fanin2_tt = np.array(fanin2.truth_table, dtype=np.uint8)

    # Apply inversion
    if inv1 == 1:
        fanin1_tt = 1 - fanin1_tt
    if inv2 == 1:
        fanin2_tt = 1 - fanin2_tt

    # Compute the truth table, supporting five gate types.
    if node.gate_type == 'AND':
        truth_table = fanin1_tt & fanin2_tt
    elif node.gate_type == 'OR':
        truth_table = fanin1_tt | fanin2_tt
    elif node.gate_type == 'XOR':
        truth_table = fanin1_tt ^ fanin2_tt
    elif node.gate_type == 'NOR':
        truth_table = 1 - (fanin1_tt | fanin2_tt)
    elif node.gate_type == 'NAND':
        truth_table = 1 - (fanin1_tt & fanin2_tt)
    else:
        raise ValueError(f"Unsupported gate type: {node.gate_type}")

    node.truth_table = truth_table.tolist()


def generate_ops(total_gates, use_aig, gate_ratios=None):
    """
    Generate a list of gate types with specified ratios for four gate types.
    """
    if use_aig:
        return ["AND"] * total_gates
    else:
        # Use equal proportions by default.
        if gate_ratios is None:
            gate_ratios = {'AND': 0.2, 'OR': 0.2, 'NOR': 0.2, 'NAND': 0.2, "XOR":0.2}

        gate_types = list(gate_ratios.keys())
        ratios = list(gate_ratios.values())

        # Generate gate types with the specified proportions using random.choices.
        ops = random.choices(gate_types, weights=ratios, k=total_gates)
        random.shuffle(ops)
        return ops

def trivial_check(tt):
    if tt is None or sum(tt) == 0 or sum(tt) == len(tt):
        return False
    return True


def allocate_output(aig, start_id, out_left, clear_outs=False):
    """
    Allocate outputs for an AIG (And-Inverter Graph).
    """
    if clear_outs:
        for id in aig.outs:
            aig.nodes[id].out = False
        aig.outs = {}

    # Identify nontrivial and trivial nodes
    nontrivial_ids = [id for id in range(start_id, len(aig.nodes))
                      if trivial_check(aig.nodes[id].truth_table)]
    trivial_ids_set = set(range(start_id, len(aig.nodes))) - set(nontrivial_ids)

    # Identify dangling nodes among nontrivial ones
    dangling_nontrivial = [id for id in nontrivial_ids if aig.nodes[id].hanged]
    needed_nondang = out_left - len(dangling_nontrivial)

    if len(nontrivial_ids) < out_left:
        print('[ERROR] not enough nontrivial nodes')

    if needed_nondang < 0:
        # More dangling nodes than needed - select subset by priority
        dangling_priorities = [max(aig.nodes[id].priority + min((aig.nodes[id].layer - aig.max_layer / 6) * 2, 0),
                                   1e-6) for id in dangling_nontrivial]
        chosen_out_ids = np.random.choice(
            dangling_nontrivial,
            size=out_left,
            replace=False,
            p=np.array(dangling_priorities) / np.sum(dangling_priorities)
        ).tolist()
    else:
        # Use all dangling nodes, then supplement with nondangling nodes
        nondangling_nontrivial = list(set(nontrivial_ids) - set(dangling_nontrivial))
        nondangling_priorities = [max(aig.nodes[id].priority + min((aig.nodes[id].layer - aig.max_layer / 6) * 2, 0),
                                      1e-6) for id in nondangling_nontrivial]

        # Calculate how many trivial nodes we might need
        needed_trivial = needed_nondang - len(nondangling_nontrivial)
        trivial_ids = []

        if needed_trivial > 0:
            # Find dangling trivial nodes
            dangling_trivial = [id for id in range(start_id, len(aig.nodes)) if
                                not trivial_check(aig.nodes[id].truth_table) and aig.nodes[id].hanged]
            random.shuffle(dangling_trivial)

            # Find nondangling trivial nodes
            nondangling_trivial = list(trivial_ids_set - set(dangling_trivial))

            # Combine and select required number of trivial nodes
            dangling_trivial.extend(nondangling_trivial)
            trivial_ids = dangling_trivial[:needed_trivial]

        # Select nondangling nontrivial nodes based on priority
        selected_nondangling = np.random.choice(
            nondangling_nontrivial,
            size=min(needed_nondang, len(nondangling_nontrivial)),
            replace=False,
            p=np.array(nondangling_priorities) / np.sum(nondangling_priorities)
        ).tolist()

        chosen_out_ids = dangling_nontrivial + selected_nondangling + trivial_ids

    # Mark selected nodes as outputs
    for id in chosen_out_ids:
        aig.nodes[id].out = True
        aig.outs.update({id: random.randint(1, 2)})


def random_logic_net(k, l, Mstep, use_aig, p0=3):
    """
    Generate a random logic network with specified parameters.

    Args:
        k (int): Number of input variables
        l (int): Number of output functions
        Mstep (int): Total number of logic gates to generate
        use_aig (bool): True if AIG else extended gates

    Returns:
        AIG: The generated random logic network
    """
    MAX_TRIES = 20000

    # Create input nodes
    inputs = [AIGNode(i, 'INPUT', priority=3) for i in range(k)]
    aig = AIG(k, l)
    aig.nodes.extend(inputs)
    aig.init_input_tt()
    priority = [p0 for _ in range(k)]
    out_left = l

    # Generate random sequence of gate operations
    custom_ratios = {'AND': 0.0, 'OR': 0.0, 'NOR': 0.5, 'NAND': 0.5, "XOR":0.0}
    ops = generate_ops(Mstep, use_aig=use_aig) #,gate_ratios=custom_ratios)
    danglings = inputs.copy()  # Nodes that can be connected to
    num_dangling = k

    def postprocess(num_dangling, priority, hanged, c0, c1, new_node):
        '''
        work after choosing c0 c1,
        to update neighbourhood relationships and update priority.
        '''
        new_node.add_fanin((c0.id + 1) * 2 if not invert_c0 else (c0.id + 1) * 2 + 1)
        new_node.add_fanin((c1.id + 1) * 2 if not invert_c1 else (c1.id + 1) * 2 + 1)
        c0.add_fanout((new_node.id + 1) * 2 + invert_c0)
        c1.add_fanout((new_node.id + 1) * 2 + invert_c1)
        new_node.ancestors = (c0.ancestors | c1.ancestors |
                              {c0.id * 2 + int(invert_c0), c1.id * 2 + int(invert_c1)})

        # Update node priorities and deletion costs
        # new_node.priority = p0 # max(c0.priority, c1.priority) + 1
        new_node.layer = max(c0.layer, c1.layer) + 1
        new_node.priority = p0 + math.sqrt(new_node.layer / p0)
        # new_node.priority = p0#+(new_node.layer/p0)
        c0.priority = c0.priority / p0
        c1.priority = c1.priority / p0
        priority.append(new_node.priority + math.sqrt(len(new_node.ancestors) / p0))

        # Update hanging status of connected nodes
        if c0.hanged: num_dangling -= 1
        if c1.hanged: num_dangling -= 1
        c0.hanged = False
        c1.hanged = False
        hanged.append(new_node)
        num_dangling += 1

        # Add the new node to working sets
        aig.nodes.append(new_node)
        aig.max_layer = max(aig.max_layer, new_node.layer)  # todo: ,max_layer)

        return num_dangling

    for i in range(Mstep):
        gate_type = ops[i]
        new_node = AIGNode(aig.new_var(), gate_type=gate_type)
        only_connect_hanged = num_dangling >= out_left - 1

        # Randomly determine input inversions (if allowed)
        invert_c0 = random.choice([True, False])  # False if c0.hasnotfin else
        invert_c1 = random.choice([True, False])  # False if c1.hasnotfin else
        new_node.hasnotfin = invert_c0 or invert_c1

        # Select valid nodes for connection based on hanging status
        pool = []  # never < 2, no need to judge
        priorities = []
        for p, h in zip(priority, danglings):
            if h.hanged or not only_connect_hanged:
                pool.append(h)
                priorities.append(p)
        Pool = pool.copy()
        Priorities = priorities.copy()

        # Find first candidate node (c0) that doesn't create cycles
        c0 = random.choices(pool, weights=priorities, k=1)[0]
        c0_index = pool.index(c0)
        pool.remove(c0)  # inplace
        priorities.pop(c0_index)
        c1_range = [node for node in pool]
        c1_pri = []

        for node in c1_range:
            id = node.id
            if id * 2 in c0.ancestors and id * 2 + 1 in c0.ancestors:
                c1_pri.append(0)
            elif id * 2 in c0.ancestors:
                invert_c1 = False
                c1_pri.append(node.priority / p0)
            elif id * 2 + 1 in c0.ancestors:
                invert_c1 = True  # todo: store inv
                c1_pri.append(node.priority / p0)
            else:
                c1_pri.append(node.priority)

        c1 = random.choices(c1_range, weights=c1_pri)[0]
        compute_truth(new_node, c0, c1, invert_c0, invert_c1)
        cnt = 0
        while not trivial_check(new_node.truth_table):
            cnt += 1
            if cnt >= MAX_TRIES:
                new_node.trivial = True
                break
            # Fallback if no valid pairs found after attempts
            c = np.random.choice(
                Pool,
                size=2,
                replace=False,
                p=np.array(Priorities) / np.sum(Priorities)
            ).tolist()
            c0, c1 = c[0], c[1]  # Pool[idx[0]], Pool[idx[1]]
            invert_c0 = random.choice([True, False])
            invert_c1 = random.choice([True, False])
            compute_truth(new_node, c0, c1, invert_c0, invert_c1)

        # Connect the new node to its inputs
        num_dangling = postprocess(num_dangling, priority, danglings, c0, c1, new_node)

    allocate_output(aig, start_id=aig.k, out_left=aig.l)

    return aig
