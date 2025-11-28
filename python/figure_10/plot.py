import os
import re
import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
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


def parse_result_log(result_log_path):
    """Parse result.log file and return dictionaries, setting missing model keys to 0."""
    result_dicts = {}
    
    with open(result_log_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match dictionary assignments: var_name = {...}
        match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if match:
            var_name = match.group(1)
            dict_str = match.group(2).strip()
            
            try:
                # Parse the dictionary string
                parsed_dict = ast.literal_eval(dict_str)
                result_dicts[var_name] = parsed_dict
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse {var_name}: {e}")
                continue
    
    return result_dicts


def ensure_all_models(d, model_keys):
    """Ensure all model keys exist in dictionary, setting missing ones to 0."""
    for model in model_keys:
        d.setdefault(model, 0.0)
    return d


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result-log', type=str, required=True, help='Path to result.log file')
parser.add_argument('--output-folder', type=str, required=True, help='Output folder for plots')
args = parser.parse_args()

# Read data from result.log
result_data = parse_result_log(args.result_log)

# Extract dictionaries for A100
orig_time_a100 = ensure_all_models(result_data.get('orig_time_a100', {}), models.keys())
gpu_collection_a100 = ensure_all_models(result_data.get('gpu_collection_a100', {}), models.keys())
gpu_transfer_a100 = ensure_all_models(result_data.get('gpu_transfer_a100', {}), models.keys())
gpu_analysis_a100 = ensure_all_models(result_data.get('gpu_analysis_a100', {}), models.keys())
gpu_total_a100 = ensure_all_models(result_data.get('gpu_total_a100', {}), models.keys())

cpu_collection_a100 = ensure_all_models(result_data.get('cpu_collection_a100', {}), models.keys())
cpu_transfer_a100 = ensure_all_models(result_data.get('cpu_transfer_a100', {}), models.keys())
cpu_analysis_a100 = ensure_all_models(result_data.get('cpu_analysis_a100', {}), models.keys())
cpu_total_a100 = ensure_all_models(result_data.get('cpu_total_a100', {}), models.keys())

nvbit_collection_a100 = ensure_all_models(result_data.get('nvbit_collection_a100', {}), models.keys())
nvbit_transfer_a100 = ensure_all_models(result_data.get('nvbit_transfer_a100', {}), models.keys())
nvbit_analysis_a100 = ensure_all_models(result_data.get('nvbit_analysis_a100', {}), models.keys())
nvbit_total_a100 = ensure_all_models(result_data.get('nvbit_total_a100', {}), models.keys())

# Handle missing whisper entries in nvbit data for A100
for d in (nvbit_collection_a100, nvbit_transfer_a100, nvbit_analysis_a100, nvbit_total_a100):
    d.setdefault('whisper', 0.0)

# Extract dictionaries for 3060
orig_time_3060 = ensure_all_models(result_data.get('orig_time_3060', {}), models.keys())
gpu_collection_3060 = ensure_all_models(result_data.get('gpu_collection_3060', {}), models.keys())
gpu_transfer_3060 = ensure_all_models(result_data.get('gpu_transfer_3060', {}), models.keys())
gpu_analysis_3060 = ensure_all_models(result_data.get('gpu_analysis_3060', {}), models.keys())
gpu_total_3060 = ensure_all_models(result_data.get('gpu_total_3060', {}), models.keys())

cpu_collection_3060 = ensure_all_models(result_data.get('cpu_collection_3060', {}), models.keys())
cpu_transfer_3060 = ensure_all_models(result_data.get('cpu_transfer_3060', {}), models.keys())
cpu_analysis_3060 = ensure_all_models(result_data.get('cpu_analysis_3060', {}), models.keys())
cpu_total_3060 = ensure_all_models(result_data.get('cpu_total_3060', {}), models.keys())

nvbit_collection_3060 = ensure_all_models(result_data.get('nvbit_collection_3060', {}), models.keys())
nvbit_transfer_3060 = ensure_all_models(result_data.get('nvbit_transfer_3060', {}), models.keys())
nvbit_analysis_3060 = ensure_all_models(result_data.get('nvbit_analysis_3060', {}), models.keys())
nvbit_total_3060 = ensure_all_models(result_data.get('nvbit_total_3060', {}), models.keys())

# Handle missing whisper entries in nvbit data for 3060
for d in (nvbit_collection_3060, nvbit_transfer_3060, nvbit_analysis_3060, nvbit_total_3060):
    d.setdefault('whisper', 0.0)

# orig_time_rate_a100 = []
# gpu_collection_rate_a100 = []
# gpu_transfer_rate_a100 = []
# gpu_analysis_rate_a100 = []

# for model in models.keys():
#     orig_time_rate_a100.append(orig_time_a100[model] / gpu_total_a100[model])
#     gpu_collection_rate_a100.append(gpu_collection_a100[model] / gpu_total_a100[model])
#     gpu_transfer_rate_a100.append(gpu_transfer_a100[model] / gpu_total_a100[model])
#     gpu_analysis_rate_a100.append(gpu_analysis_a100[model] / gpu_total_a100[model])


# orig_time_rate_3060 = []
# gpu_collection_rate_3060 = []
# gpu_transfer_rate_3060 = []
# gpu_analysis_rate_3060 = []

# for model in models.keys():
#     orig_time_rate_3060.append(orig_time_3060[model] / gpu_total_3060[model])
#     gpu_collection_rate_3060.append(gpu_collection_3060[model] / gpu_total_3060[model])
#     gpu_transfer_rate_3060.append(gpu_transfer_3060[model] / gpu_total_3060[model])
#     gpu_analysis_rate_3060.append(gpu_analysis_3060[model] / gpu_total_3060[model])


# print("orig_time_rate_a100 =", orig_time_rate_a100)
# print("gpu_collection_rate_a100 =", gpu_collection_rate_a100)
# print("gpu_transfer_rate_a100 =", gpu_transfer_rate_a100)
# print("gpu_analysis_rate_a100 =", gpu_analysis_rate_a100)

# print()
# print("orig_time_rate_3060 =", orig_time_rate_3060)
# print("gpu_collection_rate_3060 =", gpu_collection_rate_3060)
# print("gpu_transfer_rate_3060 =", gpu_transfer_rate_3060)
# print("gpu_analysis_rate_3060 =", gpu_analysis_rate_3060)


# ---------- Helpers ----------
def to_rates(orig, collection, transfer, analysis, total, ordered_models):
    """Return array with columns [orig, collection, transfer, analysis] as fractions."""
    out = []
    for m in ordered_models:
        denom = total[m]
        if denom == 0:
            out.append([0, 0, 0, 0])
        else:
            out.append([
                orig[m] / denom,
                collection[m] / denom,
                transfer[m] / denom,
                analysis[m] / denom,
            ])
    return np.array(out)


# Calculate rates for A100
a100_gpu_rates = to_rates(orig_time_a100, gpu_collection_a100, gpu_transfer_a100, gpu_analysis_a100, gpu_total_a100, models.keys())
a100_cpu_rates = to_rates(orig_time_a100, cpu_collection_a100, cpu_transfer_a100, cpu_analysis_a100, cpu_total_a100, models.keys())
a100_nvbit_rates = to_rates(orig_time_a100, nvbit_collection_a100, nvbit_transfer_a100, nvbit_analysis_a100, nvbit_total_a100, models.keys())

# Calculate rates for 3060
rtx3060_gpu_rates = to_rates(orig_time_3060, gpu_collection_3060, gpu_transfer_3060, gpu_analysis_3060, gpu_total_3060, models.keys())
rtx3060_cpu_rates = to_rates(orig_time_3060, cpu_collection_3060, cpu_transfer_3060, cpu_analysis_3060, cpu_total_3060, models.keys())
rtx3060_nvbit_rates = to_rates(orig_time_3060, nvbit_collection_3060, nvbit_transfer_3060, nvbit_analysis_3060, nvbit_total_3060, models.keys())


fig, ax = plt.subplots(figsize=(9, 3.5))
global_font_size = 14
bar_width = 0.12
x = np.arange(len(models))

# ---------- Styling ----------
labels = ['CS-GPU-A100', 'CS-CPU-A100', 'NVBIT-CPU-A100', 'CS-GPU-3060', 'CS-CPU-3060', 'NVBIT-CPU-3060']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#e377c2', '#bcbd22']
# hatches = ['////', '\\\\\\\\', 'xxxx', '....']  # orig / collection / transfer / analysis
hatches = ['////', '\\\\', 'xx', '..']


# 6 bars per model: [A100 GPU, A100 CPU, A100 NVBit, 3060 GPU, 3060 CPU, 3060 NVBit]
offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * bar_width

def stack_bars(xpos, rates, color, label):
    bottoms = np.zeros(len(models))
    for i in range(4):
        ax.bar(xpos, rates[:, i], bar_width, bottom=bottoms,
               color=color, hatch=hatches[i], edgecolor='black', linewidth=0.4)
        bottoms += rates[:, i]
    ax.bar([], [], bar_width, color=color, label=label)

stack_bars(x + offsets[0], a100_gpu_rates,   colors[0], labels[0])
stack_bars(x + offsets[1], a100_cpu_rates,   colors[1], labels[1])
stack_bars(x + offsets[2], a100_nvbit_rates, colors[2], labels[2])
stack_bars(x + offsets[3], rtx3060_gpu_rates,   colors[3], labels[3])
stack_bars(x + offsets[4], rtx3060_cpu_rates,   colors[4], labels[4])
stack_bars(x + offsets[5], rtx3060_nvbit_rates, colors[5], labels[5])


ax.set_xticks(x)
ax.set_xticklabels(models.values())
ax.set_ylim(0, 1.02)
ax.set_ylabel("Fraction of Total Time", fontsize=global_font_size)
# ax.set_title("Per-Model Breakdown (6 bars each)\nEach bar stacks: orig, collection, transfer, analysis", fontsize=global_font_size)
ax.tick_params(axis='both', which='major', labelsize=global_font_size)

ax.grid(axis="y", linestyle="--", alpha=0.5)

# Legends: one for color (backends) and one for hatches (segments)
color_handles = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(6)]
hatch_handles = [Patch(facecolor='white', edgecolor='black', hatch=hatches[i],
                       label=['Execution','Collection','Transfer','Analysis'][i]) for i in range(4)]

# first_legend = ax.legend(handles=color_handles, ncol=3, handletextpad=0.1, columnspacing=0.1, labelspacing=0.2, loc="upper left", bbox_to_anchor=(-0.1, -0.12), title="Backend (color)", title_fontsize=global_font_size, fontsize=global_font_size)
# ax.add_artist(first_legend)
# second_legend = ax.legend(handles=hatch_handles, ncol=2, handletextpad=0.1, columnspacing=0.1, labelspacing=0.2, loc="upper right", bbox_to_anchor=(1.01, -0.12), title="Segment (hatch)", title_fontsize=global_font_size, fontsize=global_font_size)

first_legend = ax.legend(handles=color_handles, ncol=3, handletextpad=0.1, columnspacing=0.1, labelspacing=0.2, loc="upper left", bbox_to_anchor=(-0.1, -0.1), fontsize=global_font_size)
ax.add_artist(first_legend)
second_legend = ax.legend(handles=hatch_handles, ncol=2, handletextpad=0.1, columnspacing=0.1, labelspacing=0.2, loc="upper right", bbox_to_anchor=(1.01, -0.1), fontsize=global_font_size)



fig.tight_layout()

fig_name = "overhead_breakdown"
# fmt = 'png'
# fig_filename = f"{fig_name}.{fmt}"
# plt.savefig(fig_filename, format=f'{fmt}', dpi=300)

fmt = 'pdf'
os.makedirs(args.output_folder, exist_ok=True)
fig_filename = os.path.join(args.output_folder, f"{fig_name}.{fmt}")
plt.savefig(fig_filename, format=f'{fmt}', dpi=600, bbox_inches='tight', pad_inches=0.01)
