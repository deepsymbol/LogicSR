from pyvis.network import Network
import os
from graphviz import Digraph


def vis_aig(aig, filename="aig", view=False, fmt="both"):
    """
    Visualize the structure of an AIG circuit.
    Args:
        aig: The AIG object.
        filename: Output file path without an extension.
        view: Whether to open the generated image automatically.
        fmt: Output format; can be 'png' or 'html'.
    """
    if fmt=='png':
        vis_aig_png(aig,filename,view)
    elif fmt =='html':
        vis_aig_html(aig,filename,view)
    else: # both
        vis_aig_png(aig,filename,view)
        vis_aig_html(aig,filename,view)


def vis_aig_png_icon(aig, filename="aig", view=False):
    fmt = 'png'
    dot = Digraph(comment="AIG Visualization", format=fmt)

    # Use a vertical layout with small spacing between layers and wider spacing within each layer.
    dot.attr(rankdir='TB', size="4,6!", nodesep="0.8", ranksep="0.3")

    levels = {}
    outputs = []

    # ---------- Compute the level of each node ----------
    def compute_level(node_id):
        if node_id in levels:
            return levels[node_id]
        node = aig.nodes[node_id]
        if node.gate_type == "INPUT":
            levels[node_id] = 0
        else:
            level = 0
            for fin in node.fanins:
                fid = (fin >> 1) - 1
                level = max(level, compute_level(fid) + 1)
            levels[node_id] = level
        return levels[node_id]

    for node in aig.nodes:
        if node is not None:
            compute_level(node.id)

    max_level = max(levels.values()) if levels else 0

    # ---------- Add nodes ----------
    for node in aig.nodes:
        if node is None:
            continue
        nid = str(node.id)
        style = {
            "shape": "circle",
            "width": "0.4",
            "height": "0.4",
            "fixedsize": "true",
            "label": ""
        }

        if node.gate_type == "INPUT":
            dot.node(nid, fillcolor="limegreen", style="filled", **style)
        else:
            dot.node(nid, fillcolor="white", style="filled", **style)

    # ---------- Add output nodes ----------
    out_offset = max(n.id for n in aig.nodes if n) + 1
    for out_id, typ in aig.outs.items():
        virtual_id = f"out_{out_id}"
        outputs.append(virtual_id)
        dot.node(virtual_id, fillcolor="salmon", style="filled",
                 shape="circle", width="0.4", height="0.4",
                 fixedsize="true", label="")

        literal = (out_id + 1) * 2
        if typ == 2:
            literal += 1
        src_id = (literal >> 1) - 1
        is_inv = literal & 1
        style = "dashed" if is_inv else "solid"
        dot.edge(str(src_id), virtual_id, style=style, color="black")

    # ---------- Add regular edges ----------
    for node in aig.nodes:
        if node is None:
            continue
        for fanin in node.fanins:
            src = (fanin >> 1) - 1
            is_inv = fanin & 1
            style = "dashed" if is_inv else "solid"
            dot.edge(str(src), str(node.id), style=style, color="black")

    # ---------- Control the layout of each layer ----------
    layer_nodes = [[] for _ in range(max_level + 2)]
    for node in aig.nodes:
        if node is None:
            continue
        lvl = levels.get(node.id, 0)
        layer_nodes[lvl].append(str(node.id))

    for virtual_out in outputs:
        layer_nodes[-1].append(virtual_out)

    # --------- Widen the middle layers for a narrow-wide-narrow layout ---------
    max_width = max(len(layer) for layer in layer_nodes)
    for i, layer in enumerate(layer_nodes):
        pad_needed = max_width - len(layer)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        for _ in range(pad_left):
            dummy_id = f"dummy_{i}_l_{_}"
            dot.node(dummy_id, style="invis", shape="point")
            layer.insert(0, dummy_id)
        for _ in range(pad_right):
            dummy_id = f"dummy_{i}_r_{_}"
            dot.node(dummy_id, style="invis", shape="point")
            layer.append(dummy_id)

    # ---------- Apply rank=same to each layer ----------
    for layer in layer_nodes:
        with dot.subgraph() as s:
            s.attr(rank='same')
            for nid in layer:
                s.node(nid)

    # ---------- Render ----------
    dot.render(filename=filename, format='png', view=view, cleanup=True)


def vis_aig_png(aig, filename="aig", view=False):
    fmt = 'png'
    dot = Digraph(comment="AIG Visualization", format='png' if fmt == 'png' else 'svg')
    dot.attr(rankdir='LR')  # Arrange the graph from left to right.

    # --- Process layers ---
    # Place all inputs in the same rank.
    inputs = []
    outputs = []
    levels = {}

    # Precompute node levels.
    def compute_level(node_id):
        if node_id in levels:
            return levels[node_id]
        node = aig.nodes[node_id]
        if node.gate_type == "INPUT":
            levels[node_id] = 0
        else:
            level = 0
            for fin in node.fanins:
                fid = (fin >> 1) - 1
                level = max(level, compute_level(fid) + 1)
            levels[node_id] = level
        return levels[node_id]

    for node in aig.nodes:
        if node is not None:
            compute_level(node.id)

    max_level = max(levels.values()) if levels else 0

    # --- Add nodes ---
    for node in aig.nodes:
        if node is None:
            continue
        nid = str(node.id)
        if node.gate_type == "INPUT":
            dot.node(
                nid, label=f"IN {node.id}", shape="circle",
                style="filled", fillcolor="limegreen",
                width="1", height="1", fixedsize="true"
            )
        elif node.gate_type == "AND":
            dot.node(nid, label=f"{node.id}", shape="box",
                     style="filled", fillcolor="white")
        elif node.gate_type == "OR":
            dot.node(nid, label=f"{node.id}", shape="box",
                     style="filled", fillcolor="lightgray")
        else:
            dot.node(nid, label=str(node.id), shape="box", style="filled", fillcolor="gray")

    # --- Add output nodes ---
    out_offset = max(n.id for n in aig.nodes if n) + 1
    for out_id, typ in aig.outs.items():
        virtual_id = f"out_{out_id}"
        outputs.append(virtual_id)
        dot.node(virtual_id, label=f"OUT {out_id}", shape="doublecircle",
                 style="filled", fillcolor="salmon",
                 width="1", height="1", fixedsize="true")
        literal = (out_id + 1) * 2
        if typ == 2:
            literal += 1
        src_id = (literal >> 1) - 1
        is_inv = literal & 1
        style = "dashed" if is_inv else "solid"
        dot.edge(str(src_id), virtual_id, style=style, color="red")

        # --- Add edges ---
    for node in aig.nodes:
        if node is None:
            continue
        for fanin in node.fanins:
            src = (fanin >> 1) - 1
            is_inv = fanin & 1
            style = "dashed" if is_inv else "solid"
            dot.edge(str(src), str(node.id), style=style)

    # --- Control the layers ---
    layer_nodes = [[] for _ in range(max_level + 2)]  # The last layer contains outputs.
    for node in aig.nodes:
        if node is None:
            continue
        lvl = levels.get(node.id, 0)
        layer_nodes[lvl].append(str(node.id))
    for virtual_out in outputs:
        layer_nodes[-1].append(virtual_out)

    for layer in layer_nodes:
        with dot.subgraph() as s:
            s.attr(rank='same')
            for nid in layer:
                s.node(nid)

    dot.render(filename=filename, format='png', view=view, cleanup=True)


def vis_aig_html(aig, filename="aig", view=False):
    # -------------------
    # pyvis HTML output
    # -------------------
    fmt = 'html'
    net = Network(height='750px', width='100%', directed=True, notebook=False)

    colors = {
        'INPUT': '#90EE90',
        'AND': '#FFFFFF',
        'OR': '#D3D3D3',
        'OUTPUT': '#FFA07A',
        'UNKNOWN': '#D3D3D3'
    }

    levels = {}

    def compute_level(node_id):
        if node_id in levels:
            return levels[node_id]
        node = aig.nodes[node_id]
        if node is None:
            return 0
        if node.gate_type == 'INPUT':
            levels[node_id] = 0
            return 0
        max_level = 0
        for fin in node.fanins:
            fin_id = (fin >> 1) - 1
            max_level = max(max_level, compute_level(fin_id))
        levels[node_id] = max_level + 1
        return levels[node_id]

    for node in aig.nodes:
        if node is not None:
            compute_level(node.id)

    max_level = max(levels.values()) if levels else 0

    # Add the main nodes.
    for node in aig.nodes:
        if node is None:
            continue
        nid = node.id
        lvl = levels.get(nid, 0)
        label = str(nid)
        gate_type = node.gate_type or 'UNKNOWN'
        color = colors.get(gate_type, '#D3D3D3')

        net.add_node(
            nid,
            label=label,
            title=f"ID: {nid}\nType: {gate_type}\nLevel: {lvl}",
            color=color,
            shape='box',
            level=lvl,
            font={'size': 40},
            size=60,
        )

    # Add virtual output nodes.
    out_virtual_base = max(n.id for n in aig.nodes if n is not None) + 1
    for i, (out_id, typ) in enumerate(aig.outs.items()):
        node_id = out_virtual_base + i
        lvl = max_level + 1
        net.add_node(
            node_id,
            label=f"{out_id}",
            title=f"Output ID: {out_id}",
            color=colors['OUTPUT'],
            shape='box',
            level=lvl,
            font={'size': 40},  # Enlarge the text.
            size=60
        )
        literal = (out_id + 1) * 2
        if typ == 2:
            literal += 1
        src = (literal >> 1) - 1
        is_inv = literal & 1
        net.add_edge(
            src, node_id,
            color='red', #if is_inv else 'black',
            dashes= bool(is_inv),
            title=f"{src} → OUT_{out_id} ({'inv' if is_inv else 'direct'})",
            width=2,
            arrows="to"
        )

    # Add edges.
    for node in aig.nodes:
        if node is None:
            continue
        for fanin in node.fanins:
            src = (fanin >> 1) - 1
            is_inv = fanin & 1
            net.add_edge(
                src,
                node.id,
                color='black',
                dashes=bool(is_inv),  # Explicitly convert to a Boolean.
                title=f"{src} → {node.id} ({'inv' if is_inv else 'direct'})",
                width=2,
                arrows="to"
            )

    net.set_options("""
    {
        "layout": {
            "hierarchical": {
                  "enabled": true,
                  "direction": "LR",
                  "sortMethod": "directed",
                  "levelSeparation": 125, 
                  "nodeSpacing": 80,   
                  "treeSpacing": 120
            }
        },
        "interaction": {
            "dragNodes": true,
            "hover": true,
            "multiselect": true,
            "selectable": true
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true
                }
            },
            "smooth": {
                "enabled": false
            }
        },
        "physics": {
            "enabled": false
        },
        "nodes": {
            "font": {
                "size": 20,
                "face": "monospace"
            }
        }
    }
    """)

    html_path = filename + ".html"
    net.write_html(html_path)
    if view:
        os.system(f'start {html_path}' if os.name == 'nt' else f'open {html_path}')

