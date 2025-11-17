import os
import argparse
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42   # embed TrueType; searchable/selectable text
mpl.rcParams['ps.fonttype']  = 42


def parse_log_file(file_path):
    """Parse log file and extract memory sizes from Malloc/Free tensor lines."""
    idx = -3  # Index to extract reserved size (third-to-last number)
    memory_sizes = []
    
    with open(file_path, "r") as f:
        line = f.readline()
        while line:
            if "Malloc tensor" in line or "Free tensor" in line:
                nums = re.findall(r"\d+\.?\d*", line)
                if len(nums) > abs(idx):
                    memory_sizes.append(int(nums[idx]))
            line = f.readline()
    
    return np.array(memory_sizes)


# --- helpers ------------------------------------------------
def bytes_to_mb(x, _):
    return f"{int(x/1024/1024)}"


def main(log_file, output_folder, label=None):
    """Plot memory usage over time from a single log file."""
    # Parse the log file
    memory_sizes = parse_log_file(log_file)
    
    if len(memory_sizes) == 0:
        print(f"Warning: No memory data found in {log_file}")
        return
    
    # Create time axis
    t = np.arange(len(memory_sizes))
    
    # Get label from filename if not provided
    if label is None:
        label = os.path.basename(log_file).replace('.log', '')
    
    # --- figure -----------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    def k_formatter(x, pos):
        return f"{int(x/1000)}K"

    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

    # Plot memory usage
    ax.plot(t, memory_sizes, label=label, color="tab:blue", lw=1.0, alpha=0.95, rasterized=True)
    ax.fill_between(t, 0, memory_sizes, color="tab:blue", alpha=0.2, rasterized=True)

    ax.yaxis.set_major_formatter(FuncFormatter(bytes_to_mb))
    ax.set_ylabel("Memory Usage\n(MB)")
    ax.set_xlabel("Logical Timestamp (Tensor Allocation/Deallocation Event Index)")
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.85)

    plt.tight_layout()

    fig_name = os.path.join(output_folder, "memory_usage")
    fmt = 'pdf'
    fig_filename = f"{fig_name}.{fmt}"
    plt.savefig(fig_filename, format=f'{fmt}', dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"Plot saved to {fig_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPU memory usage over time from log file")
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to log file containing Malloc/Free tensor lines"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for the plot"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default=None,
        help="Label for the plot line (default: filename without extension)"
    )
    args = parser.parse_args()
    
    main(args.log_file, args.output_folder, args.label)
