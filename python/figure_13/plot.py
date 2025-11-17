import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import argparse
from matplotlib.colors import LogNorm
import os
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_hotness(filename, fig_name, output_folder):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    # Parse data
    block_indices = list(map(int, lines[0].split()))
    data = [list(map(int, line.split())) for line in lines[1:]]

    # Automatically compute the offset (e.g., 1.338e8)
    offset = np.floor(np.min(block_indices) / 10**np.floor(np.log10(np.ptp(block_indices)))) * 10**np.floor(np.log10(np.ptp(block_indices)))

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=block_indices)

    # Remove blocks with all-zero access counts
    non_zero_rows = (df.T.sum(axis=1) > 0)
    df_filtered = df.T[non_zero_rows].T
    block_indices_filtered = np.array(block_indices)[non_zero_rows.values]

    # Create figure and axes
    global_font_size = 16
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot with log scale and tighter range
    im = ax.imshow(df_filtered.T, aspect='auto', interpolation='nearest',
            cmap='coolwarm', origin='lower', norm=LogNorm(vmin=1, vmax=df_filtered.values.max()))

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Access Count (log scale)', fontsize=global_font_size)

    # Set labels
    ax.set_xlabel('Timestamps (Every 1 Million Memory Accesses)', fontsize=global_font_size)
    ax.set_ylabel('Memory Block Index (2MB Blocks)', fontsize=global_font_size)
    # ax.get_yaxis().set_label_coords(-0.02, 0.5)
    # ax.set_title(fig_name, fontsize=global_font_size)

    # Show only min and max (in scientific notation)
    min_idx = 0
    max_idx = df_filtered.shape[1] - 1
    min_label = f"{block_indices_filtered[min_idx] - offset:.1e}"
    max_label = f"{block_indices_filtered[max_idx] - offset:.1e}"

    # ax.set_yticks(ticks=[min_idx+20, max_idx-5])
    # ax.set_yticklabels([min_label, max_label], fontsize=global_font_size, rotation=90)

    fig_name = os.path.join(output_folder, fig_name)

    fmt = 'pdf'
    fig_filename = f"{fig_name}.{fmt}"
    fig.savefig(fig_filename, format=f'{fmt}', dpi=600, bbox_inches='tight')
    plt.close(fig)


def main(result_log, output_folder):
    # Read from file
    fig_name = f'hotness'
    plot_hotness(filename=result_log, fig_name=fig_name, output_folder=output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hotness from result.log")
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
    main(args.result_log, args.output_folder)