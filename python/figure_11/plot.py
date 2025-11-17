import os
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
    """Parse result.log file and extract prefetch dictionaries."""
    no_prefetch = {}
    object_level = {}
    tensor_level = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse lines like "no_prefetch = {'alexnet': 0.84, ...}"
            if '=' in line:
                var_name, dict_str = line.split('=', 1)
                var_name = var_name.strip()
                dict_str = dict_str.strip()
                
                try:
                    # Safely evaluate the dictionary string
                    parsed_dict = ast.literal_eval(dict_str)
                    
                    if var_name == 'no_prefetch':
                        no_prefetch = parsed_dict
                    elif var_name == 'object_level':
                        object_level = parsed_dict
                    elif var_name == 'tensor_level':
                        tensor_level = parsed_dict
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return no_prefetch, object_level, tensor_level

def main(no_prefetch, object_level, tensor_level, output_folder):
    models_list = list(models.keys())
    model_names = [models[model] for model in models_list]

    object_norm = [object_level[m] / no_prefetch[m] for m in models_list]
    tensor_norm = [tensor_level[m] / no_prefetch[m] for m in models_list]

    # get average of each model
    object_norm_avg = np.mean(object_norm)
    tensor_norm_avg = np.mean(tensor_norm)

    # print("object_norm_avg:", object_norm_avg)
    # print("tensor_norm_avg:", tensor_norm_avg)

    model_names.append("Avg.")
    object_norm.append(object_norm_avg)
    tensor_norm.append(tensor_norm_avg)
    data = [object_norm, tensor_norm]
    labels = ['Object-Level Prefetch', 'Tensor-Level Prefetch']
    colors = ['#1f77b4', '#ff7f0e']

    x = np.arange(len(model_names))
    total_width = 0.75
    width = total_width / len(data)

    # Plot
    global_font_size = 16
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i in range(len(data)):
        bars = ax.bar(x + i * width - width * (len(data) - 1) / 2,
                   [1e6 if val == float('inf') else val for val in data[i]],
                   width,
                   label=labels[i],
                   color=colors[i],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=0.1
                   )

    # Labels and formatting
    ax.set_ylabel('Execution Time\n(Normalized to No Prefetch)', fontsize=global_font_size, y=0.55)
    # ax.set_title('Prefetching Performance (Normalized to No Prefetch) (3060)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0, fontsize=global_font_size)
    ax.tick_params(axis='both', which='major', labelsize=global_font_size)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    ax.legend(
              loc='lower center',
              bbox_to_anchor=(0.5, 0.93),
              ncol=2,
              fontsize=global_font_size-1,
              frameon=False,
              handletextpad=0.5,
              columnspacing=0.5,
              labelspacing=0.2)

    plt.tight_layout()

    fig_name = os.path.join(output_folder, "uvm_speedup")
    # fmt = 'png'
    # # Show the plot
    # fig_filename = f"{fig_name}.{fmt}"
    # plt.savefig(fig_filename, format=f'{fmt}')

    fmt = 'pdf'
    # Show the plot
    fig_filename = f"{fig_name}.{fmt}"
    plt.savefig(fig_filename, format=f'{fmt}', dpi=600, bbox_inches='tight', pad_inches=0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot prefetching performance from result.log")
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
    
    no_prefetch, object_level, tensor_level = parse_result_log(args.result_log)
    main(no_prefetch, object_level, tensor_level, args.output_folder)
