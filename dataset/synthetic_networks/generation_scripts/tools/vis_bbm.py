import os
import re
import matplotlib.pyplot as plt
from collections import Counter


def parse_folder_metadata(folder_name):
    """Extract the variable count, input count, and problem type from a folder name."""
    keywords = ['cell', 'cycle', 'gene', 'immune', 'signal', 'cancer', 'mapk', 'grn']

    var_match = re.search(r'\[var-(\d+)\]', folder_name)
    in_match = re.search(r'\[in-(\d+)\]', folder_name)

    num_var = int(var_match.group(1)) if var_match else None
    num_in = int(in_match.group(1)) if in_match else None

    parts = folder_name.split('__')
    last_part = parts[-1] if parts else ''
    type_match = re.search(r'\[(.+?)\]', last_part)

    if type_match:
        raw_type = type_match.group(1).replace('-', '').lower()
        for kw in keywords:
            if kw in raw_type:
                problem_type = kw
                break
        else:
            problem_type = 'others'
    else:
        problem_type = 'others'

    return problem_type, num_var, num_in


def visualize_bar(counter, xlabel, ylabel, title, save_path):
    sorted_items = sorted(counter.items(), key=lambda x: -x[1])
    labels, values = zip(*sorted_items)

    plt.figure(figsize=(8, 4))  # Use a slightly narrower and shorter figure.
    bar_width = 0.4  # Use narrower bars.

    bars = plt.bar(labels, values, color='orange', edgecolor='black', width=bar_width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')

    # Add value labels above the bars.
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,  # Position labels slightly above the bars.
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_models(models_root):
    type_counter = Counter()
    var_list = []
    in_list = []

    for folder in os.listdir(models_root):
        folder_path = os.path.join(models_root, folder)
        if os.path.isdir(folder_path):
            prob_type, var, in_num = parse_folder_metadata(folder)
            type_counter[prob_type] += 1
            if var is not None:
                var_list.append(var)
            if in_num is not None:
                in_list.append(in_num)

    return type_counter, var_list, in_list


def visualize_hist(data, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=20, color='steelblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main():
    models_dir = "bio/models"  # Set this to the path of your models directory.

    type_counter, var_list, in_list = analyze_models(models_dir)

    # Visualize the distribution of problem types.
    visualize_bar(type_counter,
                  xlabel="Problem Type",
                  ylabel="Model Count",
                  title="Distribution of Problem Types",
                  save_path="./v1_dangling/vis/problem_type_distribution.png")

    # Visualize the variable counts.
    visualize_hist(var_list,
                   xlabel="Number of Variables",
                   ylabel="Frequency",
                   title="Variable Count Distribution",
                   save_path="./v1_dangling/vis/variable_count_distribution.png")

    # Visualize the input counts.
    visualize_hist(in_list,
                   xlabel="Number of Inputs",
                   ylabel="Frequency",
                   title="Input Count Distribution",
                   save_path="./v1_dangling/vis/input_count_distribution.png")

    print("✅ Plots saved to 'vis/' folder")


if __name__ == "__main__":
    os.makedirs("./v1_dangling/vis", exist_ok=True)
    main()
