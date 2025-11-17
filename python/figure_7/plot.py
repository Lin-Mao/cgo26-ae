import re
import os
import circlify
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def _get_prefix_short(line):
    prefix = line.split('<')[0]
    return prefix

def _get_prefix_long(line):
    prefix = ''
    stack = []
    first_tag = False
    for i in line:
        prefix += i
        if i == '<':
            if not first_tag:
                first_tag = True
            stack.append(i)
        elif i == '>':
            if stack:
                stack.pop()
        if first_tag and not stack:
            break

    return prefix


def plot_buble(kernel_dict, output_folder):
    counts = []
    kernels = []

    for kernel, count in kernel_dict.items():
        counts.append(count)
        kernels.append(kernel)

    # ---------------------------
    # Step 2: Circlify the data FIRST (DO NOT SORT FIRST!)
    circle_data = [{"id": k, "datum": c} for k, c in zip(kernels, counts)]

    circles = circlify.circlify(
        [item["datum"] for item in circle_data],
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    # ---------------------------
    # Step 3: After packing, sort circles by radius
    circle_items = list(zip(circles, circle_data))
    circle_items_sorted = sorted(circle_items, key=lambda x: x[0].r, reverse=True)

    # ---------------------------
    # Step 4: Assign colors based on radius order (biggest bubble gets first color)
    top_N = 20
    if len(circle_items_sorted) < top_N:
        top_N = len(circle_items_sorted)
    color_palette = mpl.colormaps.get_cmap('tab20')

    # Color assignment for each circle (by circle order in layout)
    colors_by_circle = {}
    for idx, (circle, item) in enumerate(circle_items_sorted):
        if idx < top_N:
            colors_by_circle[item["id"]] = color_palette(idx / top_N)
        else:
            colors_by_circle[item["id"]] = (0.85, 0.85, 0.85)  # gray

    # ---------------------------
    # Step 5: Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    # ax.set_title("Kernel Invocation Frequency (Packed Bubble Chart)", fontsize=16)

    # Plot circles in original layout order (circles list)
    for circle, item in zip(circles, circle_data):
        x, y, r = circle.x, circle.y, circle.r
        color = colors_by_circle[item["id"]]
        ax.add_patch(plt.Circle((x, y), r, alpha=0.7, linewidth=1.5,
                                edgecolor='black', facecolor=color))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # ---------------------------
    # Step 6: Legend — build from sorted circles (biggest first)
    data_sorted = sorted(zip(counts, kernels), reverse=True)
    sorted_counts, sorted_kernels = zip(*data_sorted)
    sorted_colors = []
    for idx in range(len(sorted_counts)):
        if idx < top_N:
            color = color_palette(idx / top_N)
        else:
            color = (0.85, 0.85, 0.85)  # gray
        sorted_colors.append(color)
    legend_handles = []
    for idx in range(top_N):
        label = f"{sorted_kernels[idx][:35]}{'...' if len(sorted_kernels[idx]) > 35 else ''} ({sorted_counts[idx]:,})"
        legend_handles.append(Patch(facecolor=sorted_colors[idx], label=label))

    # count of other kernels
    other_count = sum(sorted_counts[top_N:])
    other_label = f"Other kernels ({other_count:,})"
    legend_handles.append(Patch(facecolor=(0.85, 0.85, 0.85), label=other_label))

    ax.legend(handles=legend_handles,
              title=f"Top {top_N} Kernels (Ordered by invocation frequency)",
              bbox_to_anchor=(1.05, 0.5), loc='center left')

    # ax.set_aspect('equal')
    plt.tight_layout()
    # plt.show()

    fmt = 'pdf'
    # Show the plot
    fig_filename = f"{output_folder}/figure7.{fmt}"
    plt.savefig(fig_filename, format=f'{fmt}')
    plt.close(fig)


def draw_buble_chart_all_models(log_folder, output_folder):
    log_files = [f for f in os.listdir(log_folder) if f.endswith(".txt")]

    kernel_dict_all = {}
    for fn in log_files:
        print(fn)
        file = open(os.path.join(log_folder, fn), "r")
        line = file.readline().strip()

        
        while line:
            prefix = _get_prefix_short(line)
            prefix = prefix.replace("void ", "")
            kernel_dict_all[prefix] = kernel_dict_all.get(prefix, 0) + 1
            line = file.readline().strip()

    plot_buble(kernel_dict_all, output_folder)

def main(log_folder, output_folder):
    draw_buble_chart_all_models(log_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-folder",
        type=str,
        required=True,
        help="Log folder path"
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder path"
    )
    args = parser.parse_args()
    main(args.log_folder, args.output_folder)
