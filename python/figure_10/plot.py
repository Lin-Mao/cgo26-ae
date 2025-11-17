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
parser.add_argument('--tag', type=str, default='', help='GPU type tag (empty string for A100, "3060" for RTX 3060)')
args = parser.parse_args()

# Determine GPU type from tag (default "" means A100)
gpu_name = args.tag

# Read data from result.log
result_data = parse_result_log(args.result_log)

# Build suffix for dictionary keys
# When tag is "" (default), use keys without suffix (A100 default)
# When tag is "3060", use keys with "_3060" suffix
suffix = f"_{args.tag}" if args.tag != "" else ""

def get_dict(key_base):
    """Get dictionary from result_data, trying with suffix first if tag is set, otherwise without suffix."""
    if args.tag != "":
        # Try with suffix first, then without suffix as fallback
        return result_data.get(f'{key_base}{suffix}', result_data.get(key_base, {}))
    else:
        # Try without suffix first, then with "_a100" as fallback
        return result_data.get(key_base, result_data.get(f'{key_base}_a100', {}))

# Extract dictionaries and ensure all models exist
orig_time = ensure_all_models(get_dict('orig_time'), models.keys())
gpu_collection = ensure_all_models(get_dict('gpu_collection'), models.keys())
gpu_transfer = ensure_all_models(get_dict('gpu_transfer'), models.keys())
gpu_analysis = ensure_all_models(get_dict('gpu_analysis'), models.keys())
gpu_total = ensure_all_models(get_dict('gpu_total'), models.keys())

cpu_collection = ensure_all_models(get_dict('cpu_collection'), models.keys())
cpu_transfer = ensure_all_models(get_dict('cpu_transfer'), models.keys())
cpu_analysis = ensure_all_models(get_dict('cpu_analysis'), models.keys())
cpu_total = ensure_all_models(get_dict('cpu_total'), models.keys())

nvbit_collection = ensure_all_models(get_dict('nvbit_collection'), models.keys())
nvbit_transfer = ensure_all_models(get_dict('nvbit_transfer'), models.keys())
nvbit_analysis = ensure_all_models(get_dict('nvbit_analysis'), models.keys())
nvbit_total = ensure_all_models(get_dict('nvbit_total'), models.keys())

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


gpu_rates = to_rates(orig_time, gpu_collection, gpu_transfer, gpu_analysis, gpu_total, models.keys())
cpu_rates = to_rates(orig_time, cpu_collection, cpu_transfer, cpu_analysis, cpu_total, models.keys())
nvbit_rates = to_rates(orig_time, nvbit_collection, nvbit_transfer, nvbit_analysis, nvbit_total, models.keys())


fig, ax = plt.subplots(figsize=(9, 3.5))
global_font_size = 14
bar_width = 0.12
x = np.arange(len(models))

# ---------- Styling ----------
labels = [f'CS-GPU-{gpu_name}', f'CS-CPU-{gpu_name}', f'NVBIT-CPU-{gpu_name}']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# hatches = ['////', '\\\\\\\\', 'xxxx', '....']  # orig / collection / transfer / analysis
hatches = ['////', '\\\\', 'xx', '..']


# 3 bars per model: [GPU, CPU, NVBit]
offsets = np.array([-1, 0, 1]) * bar_width

def stack_bars(xpos, rates, color, label):
    bottoms = np.zeros(len(models))
    for i in range(4):
        ax.bar(xpos, rates[:, i], bar_width, bottom=bottoms,
               color=color, hatch=hatches[i], edgecolor='black', linewidth=0.4)
        bottoms += rates[:, i]
    ax.bar([], [], bar_width, color=color, label=label)

stack_bars(x + offsets[0], gpu_rates,   colors[0], labels[0])
stack_bars(x + offsets[1], cpu_rates,   colors[1], labels[1])
stack_bars(x + offsets[2], nvbit_rates, colors[2], labels[2])


ax.set_xticks(x)
ax.set_xticklabels(models.values())
ax.set_ylim(0, 1.02)
ax.set_ylabel("Fraction of Total Time", fontsize=global_font_size)
# ax.set_title("Per-Model Breakdown (6 bars each)\nEach bar stacks: orig, collection, transfer, analysis", fontsize=global_font_size)
ax.tick_params(axis='both', which='major', labelsize=global_font_size)

ax.grid(axis="y", linestyle="--", alpha=0.5)

# Legends: one for color (backends) and one for hatches (segments)
color_handles = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(3)]
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
