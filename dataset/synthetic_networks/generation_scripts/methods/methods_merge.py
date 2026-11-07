from tqdm import tqdm
import time
import os
from collections import defaultdict
from methods.methods_plain import *


def priority_weight(node, out_penalty=1, alpha=1.5, divider=3):
    """
    Compute a power-law weight for a node based on priority.
    Higher priority -> higher weight.
    """
    p = node.priority + math.sqrt(len(node.aig_ancestors)/divider)

    weight = p ** alpha
    if node.out:
        weight *= out_penalty
    if node.trivial:
        weight = 0
    return weight

def select_aig_list(aigs, aig_selected_count, aig_connect_count, size=1, node=None, p0=3):
    zero_count_aigs = [aig for aig in aigs if aig_connect_count[id(aig)] == 0]
    if len(zero_count_aigs) >= size:
        return np.random.choice(zero_count_aigs, size=size, replace=False).tolist()

    # first choose from count=0
    sele = zero_count_aigs.copy()
    remaining_size = size - len(sele)

    # others use the previous weight sampling strategy
    non_zero_aigs = [aig for aig in aigs if aig_selected_count[id(aig)] > 0]
    aig_weights = []
    for aig in non_zero_aigs:
        weight = 1 / (1 + aig_selected_count[id(aig)])
        if node and id(aig) in node.aig_ancestors.keys():
            weight /= p0
        aig_weights.append(weight)

    if remaining_size > 0 and non_zero_aigs:
        more = np.random.choice(
            non_zero_aigs, size=remaining_size, replace=False,
            p=np.array(aig_weights) / np.sum(aig_weights)
        ).tolist()
        sele.extend(more)

    return sele

def manipulate(aigs, k, l, in_left, tt_size, p0):
    # create new inputs
    new_inputs = [AIGNode(i, 'INPUT', priority=p0) for i in range(k - in_left, k)]
    new_aig = AIG(k, l)

    # fill input truth tables
    for node_id in range(in_left):
        input_node = new_inputs[node_id]
        tt = [(i >> (in_left - node_id - 1)) & 1 for i in reversed(range(2 ** in_left))]
        input_node.truth_table = tt * int(tt_size / len(tt)) + tt[:(tt_size % len(tt))]

    # add small network inputs
    for aig in aigs:
        new_aig.nodes.extend(aig.nodes[:aig.k])
    new_aig.nodes.extend(new_inputs)

    # add internal nodes
    for aig in aigs:
        new_aig.nodes.extend(aig.nodes[aig.k:])
        new_aig.outs.update(aig.outs)

    new_aig.var_count = len(new_aig.nodes)
    return new_inputs, new_aig

def generate_merged_aig(use_aig, aigs, k, l, in_left, out_left, Msteps, tt_size,
                     use_tqdm=False, p0 = 3, REPEAT_outer = 20, REPEAT_inner = 25):
    random.seed(time.time() + os.getpid())

    new_inputs,new_aig = manipulate(aigs, k, l, in_left, tt_size, p0)
    only_connect_dangling = False
    num_dangling = in_left
    aig_selected_count = defaultdict(int)
    aig_connect_count = defaultdict(int)
    all_nodes = new_inputs.copy()   # []

    def postprocess(node1,node2,gate_node,invert1,invert2,aig0,aig1,new_inputs,connect_to_new):
        gate_node.add_fanin((node1.id + 1) * 2 + int(invert1))
        gate_node.add_fanin((node2.id + 1) * 2 + int(invert2))
        node1.add_fanout(gate_node.id * 2 + int(invert1))
        node2.add_fanout(gate_node.id * 2 + int(invert2))
        gate_node.layer = max(node1.layer, node2.layer)+1

        all_nodes.append(gate_node)
        node1.hanged = False
        node2.hanged = False
        node1.priority = node1.priority/p0
        node2.priority = node2.priority/p0
        gate_node.priority = p0+math.sqrt(gate_node.layer/p0)

        if len(new_inputs)>0:
            if node1 in new_inputs: new_inputs.remove(node1)
            if node2 in new_inputs: new_inputs.remove(node2)

        if aig0:
            aig_selected_count[id(aig0)] += 1
            if connect_to_new:
                aig_connect_count[id(aig0)] += 1
            gate_node.aig_ancestors[id(aig0)] = 1
        if aig1:
            aig_selected_count[id(aig1)] += 1
            gate_node.aig_ancestors[id(aig1)] = 1

        new_aig.max_layer = max(new_aig.max_layer,gate_node.layer)
        new_aig.nodes.append(gate_node)

    for i in tqdm(range(Msteps), disable=not use_tqdm):
        gate_type = 'AND' if use_aig else random.choice(['AND', 'OR'])
        gate_node = AIGNode(new_aig.new_var(), gate_type)
        cnt = 0

        Pool, Priorities = [], []
        for node in all_nodes:
            if node.hanged or not only_connect_dangling:
                Pool.append(node)
                Priorities.append(priority_weight(node))

        only_use_aigs = (len(aigs)-len(aig_connect_count.keys())) >= (Msteps-1-i)-1

        while not trivial_check(gate_node.truth_table):
            cnt += 1
            if cnt > REPEAT_outer:
                gate_node.trivial = True
                break


            connect_to_new = False

            for _ in range(REPEAT_inner):
                if trivial_check(gate_node.truth_table):
                    break
                invert1, invert2 = random.choice([True, False]), random.choice([True, False])
                aig0, aig1 = None, None
                pool = Pool.copy()
                priorities = Priorities.copy()
                p_use_pool = len(aig_selected_count.keys())/len(aigs)
                connect_to_new = False

                if (pool and random.random() < p_use_pool) or only_use_aigs:
                    # one from new gate and one from prev aigs
                    node1 = random.choices(pool, weights=priorities)[0]
                    index = pool.index(node1)
                    pool.remove(node1)
                    priorities.pop(index)
                    dangling_degrade = int(node1.hanged)
                    if pool and random.random() > max(1e-6,math.sqrt(len(new_inputs)/in_left)) and not only_use_aigs: # 0.06 for 40
                        node2 = random.choices(pool, weights=priorities)[0]
                        dangling_degrade += int(node2.hanged)
                    else: # if has new inputs, will tend to use a prev aig, just limit to case 2.
                        connect_to_new = True
                        aig0 = select_aig_list(aigs, aig_selected_count,aig_connect_count, size = 1, node = node1)[0]
                        node2 = random.choices(aig0.nodes, weights=[priority_weight(node) for node in aig0.nodes])[0]
                else:
                    # both from small aigs
                    sele = select_aig_list(aigs, aig_selected_count,aig_connect_count,size = 2)
                    aig0, aig1 = sele[0], sele[1]
                    node1 = random.choices(aig0.nodes, weights=[priority_weight(node) for node in aig0.nodes])[0]  # in aigs[0].nodes
                    node2 = random.choices(aig1.nodes, weights=[priority_weight(node) for node in aig1.nodes])[0]
                    dangling_degrade = 0

                compute_truth(gate_node, node1, node2, invert1, invert2)

        postprocess(node1,node2,gate_node,invert1,invert2,aig0,aig1,new_inputs,connect_to_new)
        num_dangling += 1 - dangling_degrade
        only_connect_dangling = (num_dangling >= out_left)

    # if len(aig_connect_count.keys()) != len(aigs):
    #     print("[WARN] some net not connected to main structure")

    return new_aig
