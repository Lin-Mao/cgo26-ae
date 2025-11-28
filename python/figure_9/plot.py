import os
import re
import ast
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


models = {
    "alexnet": "AlexNet",
    "resnet18": "RN-18",
    "resnet34": "RN-34",
    "bert": "BERT",
    "gpt2": "GPT-2",
    "whisper": "Whisper",
}

def parse_result_log(file_path):
    """Parse result.log file and extract time dictionaries for both 3060 and A100."""
    orig_time_3060 = {}
    gpu_time_3060 = {}
    cpu_time_3060 = {}
    nvbit_time_3060 = {}
    orig_time_a100 = {}
    gpu_time_a100 = {}
    cpu_time_a100 = {}
    nvbit_time_a100 = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse lines like "orig_time_3060 = {'alexnet': 20, ...}"
            if '=' in line:
                var_name, dict_str = line.split('=', 1)
                var_name = var_name.strip()
                dict_str = dict_str.strip()
                
                try:
                    # Safely evaluate the dictionary string
                    parsed_dict = ast.literal_eval(dict_str)
                    
                    if var_name == 'orig_time_3060':
                        orig_time_3060 = parsed_dict
                    elif var_name == 'gpu_time_3060':
                        gpu_time_3060 = parsed_dict
                    elif var_name == 'cpu_time_3060':
                        cpu_time_3060 = parsed_dict
                    elif var_name == 'nvbit_time_3060':
                        nvbit_time_3060 = parsed_dict
                    elif var_name == 'orig_time_a100':
                        orig_time_a100 = parsed_dict
                    elif var_name == 'gpu_time_a100':
                        gpu_time_a100 = parsed_dict
                    elif var_name == 'cpu_time_a100':
                        cpu_time_a100 = parsed_dict
                    elif var_name == 'nvbit_time_a100':
                        nvbit_time_a100 = parsed_dict
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return (orig_time_3060, gpu_time_3060, cpu_time_3060, nvbit_time_3060,
            orig_time_a100, gpu_time_a100, cpu_time_a100, nvbit_time_a100)

def time_list(data, model_keys):
    """Extract time values for given model keys."""
    return [data.get(b, 0) for b in model_keys]

def normalize(data, base):
    """Normalize data by base, handling infinity values."""
    return [float('inf') if data[i] == float('inf') else round(data[i] / base[i], 2) for i in range(len(base))]

def geo_mean(data):
    """Calculate geometric mean ignoring infinity values."""
    data = [d for d in data if d != float('inf')]
    return round(np.exp(np.mean(np.log(data))), 2)

def main(orig_time_3060, gpu_time_3060, cpu_time_3060, nvbit_time_3060,
         orig_time_a100, gpu_time_a100, cpu_time_a100, nvbit_time_a100, output_folder):
    models_list = list(models.keys())
    model_names = [models[model] for model in models_list]

    # Use np.inf for missing Whisper NVBIT entries
    if 'whisper' not in nvbit_time_3060:
        nvbit_time_3060['whisper'] = np.inf
    if 'whisper' not in nvbit_time_a100:
        nvbit_time_a100['whisper'] = np.inf

    # Extract time lists
    orig_time_a100_list = time_list(orig_time_a100, models_list)
    gpu_time_a100_list = time_list(gpu_time_a100, models_list)
    cpu_time_a100_list = time_list(cpu_time_a100, models_list)
    nvbit_time_a100_list = time_list(nvbit_time_a100, models_list)

    orig_time_3060_list = time_list(orig_time_3060, models_list)
    gpu_time_3060_list = time_list(gpu_time_3060, models_list)
    cpu_time_3060_list = time_list(cpu_time_3060, models_list)
    nvbit_time_3060_list = time_list(nvbit_time_3060, models_list)

    # Normalize times
    gpu_time_normalized_a100 = normalize(gpu_time_a100_list, orig_time_a100_list)
    cpu_time_normalized_a100 = normalize(cpu_time_a100_list, orig_time_a100_list)
    nvbit_time_normalized_a100 = normalize(nvbit_time_a100_list, orig_time_a100_list)

    gpu_time_normalized_3060 = normalize(gpu_time_3060_list, orig_time_3060_list)
    cpu_time_normalized_3060 = normalize(cpu_time_3060_list, orig_time_3060_list)
    nvbit_time_normalized_3060 = normalize(nvbit_time_3060_list, orig_time_3060_list)

    # Calculate geometric means
    gpu_time_geo_mean_a100 = geo_mean(gpu_time_normalized_a100)
    cpu_time_geo_mean_a100 = geo_mean(cpu_time_normalized_a100)
    nvbit_time_geo_mean_a100 = geo_mean(nvbit_time_normalized_a100)

    gpu_time_geo_mean_3060 = geo_mean(gpu_time_normalized_3060)
    cpu_time_geo_mean_3060 = geo_mean(cpu_time_normalized_3060)
    nvbit_time_geo_mean_3060 = geo_mean(nvbit_time_normalized_3060)

    print("gpu_time_geo_mean_a100 =", gpu_time_geo_mean_a100)
    print("cpu_time_geo_mean_a100 =", cpu_time_geo_mean_a100)
    print("nvbit_time_geo_mean_a100 =", nvbit_time_geo_mean_a100)
    print("gpu_time_geo_mean_3060 =", gpu_time_geo_mean_3060)
    print("cpu_time_geo_mean_3060 =", cpu_time_geo_mean_3060)
    print("nvbit_time_geo_mean_3060 =", nvbit_time_geo_mean_3060)

    # Add geometric mean to lists
    model_names.append("Geo.")
    gpu_time_normalized_a100.append(gpu_time_geo_mean_a100)
    cpu_time_normalized_a100.append(cpu_time_geo_mean_a100)
    nvbit_time_normalized_a100.append(nvbit_time_geo_mean_a100)
    gpu_time_normalized_3060.append(gpu_time_geo_mean_3060)
    cpu_time_normalized_3060.append(cpu_time_geo_mean_3060)
    nvbit_time_normalized_3060.append(nvbit_time_geo_mean_3060)

    # Prepare data for plotting
    data = [gpu_time_normalized_a100, cpu_time_normalized_a100, nvbit_time_normalized_a100,
            gpu_time_normalized_3060, cpu_time_normalized_3060, nvbit_time_normalized_3060]
    labels = ['CS-GPU-A100', 'CS-CPU-A100', 'NVBIT-CPU-A100', 'CS-GPU-3060', 'CS-CPU-3060', 'NVBIT-CPU-3060']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#e377c2', '#bcbd22']

    global_font_size = 16
    # Plotting
    fig, ax = plt.subplots(figsize=(9, 3.5))

    x = np.arange(len(model_names))  # the label locations
    total_width = 0.8
    width = total_width / len(data)
    bars = []
    for i in range(len(data)):
        bars.append(
            ax.bar(x + i * width - width * (len(data) - 1) / 2,
                   [1e6 if val == float('inf') else val for val in data[i]],
                   width,
                   label=labels[i],
                   color=colors[i],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=0.1)
        )

    # Annotate ∞ for the Whisper bar (index = 5)
    for i, d in enumerate(data):
        if d[5] == float('inf'):
            bar_x = x[5] + i * width - width * (len(data) - 1) / 2
            ax.text(bar_x, 1e6 * 0.67, '∞', ha='center', va='bottom', fontsize=global_font_size)

    ax.set_yscale('log')
    ax.set_ylabel('Overhead (log scale)\n(v.s. Model Execution Time)', fontsize=global_font_size, y=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0, fontsize=global_font_size)
    ax.tick_params(axis='both', which='major', labelsize=global_font_size)

    # Reorder legend
    all_labels_row_major = labels
    index_order = [0, 3, 1, 4, 2, 5]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.8, edgecolor='black', linewidth=0.1) for c in colors]
    labels_reordered = [all_labels_row_major[i] for i in index_order]
    handles_reordered = [handles[i] for i in index_order]
    ax.legend(handles=handles_reordered,
              labels=labels_reordered,
              loc='lower center',
              bbox_to_anchor=(0.5, 0.92),
              ncol=3,
              fontsize=global_font_size-1,
              frameon=False,
              handletextpad=1,
              columnspacing=2,
              labelspacing=0.3)

    fig.tight_layout()

    fig_name = os.path.join(output_folder, "overhead")
    # fmt = 'png'
    # # Show the plot
    # fig_filename = f"{fig_name}.{fmt}"
    # plt.savefig(fig_filename, format=f'{fmt}')

    fmt = 'pdf'
    # Show the plot
    fig_filename = f"{fig_name}.{fmt}"
    plt.savefig(fig_filename, format=f'{fmt}', dpi=600, bbox_inches='tight', pad_inches=0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot overhead comparison from result.log")
    parser.add_argument(
        "--result-log",
        type=str,
        required=True,
        help="Path to result.log file"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder"
    )
    args = parser.parse_args()
    
    (orig_time_3060, gpu_time_3060, cpu_time_3060, nvbit_time_3060,
     orig_time_a100, gpu_time_a100, cpu_time_a100, nvbit_time_a100) = parse_result_log(args.result_log)
    main(orig_time_3060, gpu_time_3060, cpu_time_3060, nvbit_time_3060,
         orig_time_a100, gpu_time_a100, cpu_time_a100, nvbit_time_a100, args.output_folder)