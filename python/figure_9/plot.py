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
    """Parse result.log file and extract time dictionaries."""
    orig_time = {}
    gpu_time = {}
    cpu_time = {}
    nvbit_time = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse lines like "orig_time = {'alexnet': 20, ...}"
            if '=' in line:
                var_name, dict_str = line.split('=', 1)
                var_name = var_name.strip()
                dict_str = dict_str.strip()
                
                try:
                    # Safely evaluate the dictionary string
                    parsed_dict = ast.literal_eval(dict_str)
                    
                    if var_name == 'orig_time':
                        orig_time = parsed_dict
                    elif var_name == 'gpu_time':
                        gpu_time = parsed_dict
                    elif var_name == 'cpu_time':
                        cpu_time = parsed_dict
                    elif var_name == 'nvbit_time':
                        nvbit_time = parsed_dict
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return orig_time, gpu_time, cpu_time, nvbit_time

def main(orig_time, gpu_time, cpu_time, nvbit_time, output_folder):
    models_list = list(models.keys())
    model_names = [models[model] for model in models_list]

    orig_time_list = []
    gpu_time_list = []
    cpu_time_list = []
    nvbit_time_list = []

    for b in models_list:
        orig_time_list.append(orig_time.get(b, 0))
        gpu_time_list.append(gpu_time.get(b, 0))
        cpu_time_list.append(cpu_time.get(b, 0))
        nvbit_time_list.append(nvbit_time.get(b, 0))


    # orig_time = [orig_time_list[i] / orig_time_list[i] for i in range(len(orig_time_list))]
    gpu_time_normalized = [round(gpu_time_list[i] / orig_time_list[i], 2) for i in range(len(orig_time_list))]
    cpu_time_normalized = [round(cpu_time_list[i] / orig_time_list[i], 2) for i in range(len(orig_time_list))]
    nvbit_time_normalized = [round(nvbit_time_list[i] / orig_time_list[i], 2) for i in range(len(orig_time_list))]


    print("model =", model_names)
    print("gpu_time =", gpu_time_normalized)
    print("cpu_time =", cpu_time_normalized)
    print("nvbit_time =", nvbit_time_normalized)

    # get geometric mean
    gpu_time_geo_mean = round(np.exp(np.mean(np.log(gpu_time_normalized))), 2)
    cpu_time_geo_mean = round(np.exp(np.mean(np.log(cpu_time_normalized))), 2)
    nvbit_time_geo_mean = round(np.exp(np.mean(np.log([x for x in nvbit_time_normalized if x != 0]))), 2)

    print("gpu_time_geo_mean =", gpu_time_geo_mean)
    print("cpu_time_geo_mean =", cpu_time_geo_mean)
    print("nvbit_time_geo_mean =", nvbit_time_geo_mean)

    model_names.append("Geo-mean")
    gpu_time_normalized.append(gpu_time_geo_mean)
    cpu_time_normalized.append(cpu_time_geo_mean)
    nvbit_time_normalized.append(nvbit_time_geo_mean)


    data = [gpu_time_normalized, cpu_time_normalized, nvbit_time_normalized]
    labels = ['CS-GPU', 'CS-CPU', 'NVBIT-CPU']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


    global_font_size = 14
    # Plotting
    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(model_names))  # the label locations
    total_width = 0.5
    width = total_width / len(data)  # width of each individual bar
    for i in range(len(data)):
        ax.bar(x + i * width - width * (len(data) - 1) / 2, data[i], width, label=labels[i], color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.1)

    ax.set_yscale('log')
    ax.set_ylabel('Normalized Overhead (log scale)\n(v.s. Model Execution Time)', fontsize=global_font_size, y=0.4)
    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', labelsize=global_font_size)
    ax.set_xticklabels(model_names, rotation=45, fontsize=global_font_size)

    # ax.legend(loc="lower left", fontsize=global_font_size)
    handles = [plt.Rectangle((0,0),1,1,facecolor=c,alpha=0.8,edgecolor='black',linewidth=0.1) for c in colors]
    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=len(labels),
        fontsize=global_font_size-1,
        frameon=False,
        handletextpad=1,
        columnspacing=5.0
    )

    fig.tight_layout()

    fig_name = os.path.join(output_folder, "overhead")
    # fmt = 'png'
    # # Show the plot
    # fig_filename = f"{fig_name}.{fmt}"
    # plt.savefig(fig_filename, format=f'{fmt}')

    fmt = 'pdf'
    # Show the plot
    fig_filename = f"{fig_name}.{fmt}"
    plt.savefig(fig_filename, format=f'{fmt}')

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
    
    orig_time, gpu_time, cpu_time, nvbit_time = parse_result_log(args.result_log)
    main(orig_time, gpu_time, cpu_time, nvbit_time, args.output_folder)