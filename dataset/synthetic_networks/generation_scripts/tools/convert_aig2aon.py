import os
import re
import json
# from bs4 import BeautifulSoup

# Mapping between gate type strings and binary values
TYPE_MAP = {
    'AND': 0,
    'OR': 1
}

def extract_node_type_from_html(html_path):
    """
    Extract gate types (AND/OR) from node titles in HTML file.
    Returns a dictionary mapping node IDs to their corresponding binary values (0/1).

    Args:
        html_path (str): Path to the HTML file containing the circuit visualization

    Returns:
        dict: {node_id: 0 or 1} mapping
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Extract nodes definition from script tags
    script_tags = soup.find_all("script")
    for tag in script_tags:
        if "new vis.DataSet" in tag.text and "nodes" in tag.text:
            script = tag.text
            match = re.search(r"nodes\s*=\s*new vis\.DataSet\((\[.*?\])\);", script, re.DOTALL)
            if match:
                nodes_json = match.group(1)
                nodes_data = json.loads(nodes_json)
                node_type_map = {}
                for node in nodes_data:
                    node_id = int(node['id'])
                    title = node.get('title', '')
                    m = re.search(r'Type:\s*(AND|OR)', title)
                    if m:
                        gate_type = m.group(1)
                        node_type_map[node_id] = TYPE_MAP[gate_type]
                return node_type_map
    return {}


def patch_aag_file(aag_path, node_type_map, save_path=None):
    """
    Append gate type (0/1) to each AND/OR gate line in the AAG file.

    Args:
        aag_path (str): Path to the input AAG file
        node_type_map (dict): Mapping of node IDs to gate types
        save_path (str, optional): Output file path. If None, overwrites original.
    """
    with open(aag_path, 'r') as f:
        lines = f.readlines()

    # Parse AAG header (M I L O A values)
    header = lines[0].strip().split()
    M, I, L, O, A = map(int, header[1:])

    # Keep header + inputs/latches/outputs unchanged
    new_lines = lines[:1 + I + L + O]

    # Process AND/OR gate lines
    for line in lines[1 + I + L + O:]:
        parts = line.strip().split()
        if not parts:
            continue
        node_id = int(parts[0]) // 2 - 1
        gate_type = node_type_map.get(node_id, 0)  # Default to AND (0)
        new_lines.append(f"{line.strip()} {gate_type}\n")

    # Write output (overwrite original if no save_path provided)
    out_path = save_path if save_path else aag_path
    with open(out_path, 'w') as f:
        f.writelines(new_lines)


def process_all(path):
    """
    Process all HTML/AAG file pairs in the given directory structure.

    Args:
        path (str): Base directory containing 'pic' and 'aag' subdirectories
    """
    folder_html = os.path.join(path, 'pic')
    folder_aag = os.path.join(path, 'aag')
    for fname in os.listdir(folder_html):
        if not fname.endswith('.html'):
            continue
        id_base = fname[:-5]  # Remove .html extension
        html_path = os.path.join(folder_html, f"{id_base}.html")
        aag_path = os.path.join(folder_aag, f"{id_base}.aag")

        if not os.path.exists(aag_path):
            print(f"[!] Missing AAG for {id_base}")
            continue

        node_type_map = extract_node_type_from_html(html_path)
        patch_aag_file(aag_path, node_type_map)
        print(f"[✓] Patched {aag_path}")


def patch_aag_add_zero(base_dir="."):
    """
    Find directories matching the inX_outY format (excluding those containing tt)
    and append ' 0' to each gate line in their nested andX/aag/*.aag files.
    """
    # Match in + digits + _out + digits, with no trailing characters.
    pattern = re.compile(r"^in\d+_out\d+$")

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # Keep only directories that strictly match the inX_outY format.
        if not os.path.isdir(folder_path) or not pattern.match(folder_name):
            continue

        print(f"Processing target folder: {folder_name}")

        # Iterate through the nested andX directories.
        for subfolder in os.listdir(folder_path):
            aag_dir = os.path.join(folder_path, subfolder, 'aag')
            if not os.path.isdir(aag_dir):
                continue

            # Process all .aag files in this directory.
            for fname in os.listdir(aag_dir):
                if not fname.endswith('.aag'):
                    continue

                aag_file_path = os.path.join(aag_dir, fname)

                # --- Process one AAG file ---
                with open(aag_file_path, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                # Parse the AAG header (M I L O A).
                header = lines[0].strip().split()
                try:
                    M, I, L, O, A = map(int, header[1:])
                except (ValueError, IndexError):
                    print(f"[!] Invalid AAG header in {aag_file_path}")
                    continue

                # Preserve the header and input/output definition lines.
                new_lines = lines[:1 + I + L + O]

                # Process the remaining gate lines.
                for line in lines[1 + I + L + O:]:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    parts = stripped_line.split()
                    # Append 0 if the original line has only three numbers (standard gate format).
                    if len(parts) == 3:
                        new_lines.append(f"{stripped_line} 0\n")
                    else:
                        new_lines.append(f"{stripped_line}\n")

                # Write the updated content back to the original file.
                with open(aag_file_path, 'w') as f:
                    f.writelines(new_lines)

                print(f"[✓] Patched: {aag_file_path}")


if __name__ == '__main__':
    # folder = ('in80_out80_tt15')
    # for subfolder in os.listdir(folder):
    #     path = os.path.join(folder, subfolder)
    #     process_all(path)
    patch_aag_add_zero("../AN")


