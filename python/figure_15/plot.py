import matplotlib.pyplot as plt
import re
import os
import argparse
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42   # embed TrueType; searchable/selectable text
mpl.rcParams['ps.fonttype']  = 42


def read_mem_data(fname):
    mem = []
    with open(fname, "r") as f:
        for line in f:
            m = re.search(r"\d+", line)
            if m:
                mem.append(int(m.group(0)))
    return mem

def draw_parallelism(path, file_name, output_folder):
    gpu0 = read_mem_data(f"{path}/{file_name}_0.txt")
    gpu1 = read_mem_data(f"{path}/{file_name}_1.txt")

    # --- align by common prefix + tails -------------------------
    n_common = min(len(gpu0), len(gpu1))
    t_common  = np.arange(n_common)
    gpu0_c, gpu1_c = np.array(gpu0[:n_common]), np.array(gpu1[:n_common])

    gpu0_tail = np.array(gpu0[n_common:])
    gpu1_tail = np.array(gpu1[n_common:])
    t0_tail   = np.arange(n_common, n_common + len(gpu0_tail))
    t1_tail   = np.arange(n_common, n_common + len(gpu1_tail))

    # --- helpers ------------------------------------------------
    def bytes_to_mb(x, _):
        # return f"{x/1024/1024:.1f}"
        return f"{int(x/1024/1024)}"

    to_MB = lambda arr: arr / (1024*1024)

    diff_mb_common = to_MB(gpu1_c - gpu0_c)

    # --- figure (short + stackable) -----------------------------
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

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 3.8), sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1], hspace=0.1)
    )
    ax1.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax2.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # ---------- Panel 1: Absolute usage (unchanged) ----------
    ax1.plot(t_common, gpu0_c, label="GPU 0", color="tab:blue", lw=1.0, alpha=0.95, rasterized=True)
    ax1.plot(t_common, gpu1_c, label="GPU 1", color="tab:orange", lw=1.0, alpha=0.95, ls="--", rasterized=True)
    ax1.fill_between(t_common, gpu0_c, gpu1_c, color="gray", alpha=0.1, rasterized=True)

    if gpu0_tail.size:
        ax1.plot(t0_tail, gpu0_tail, color="tab:blue", lw=0.9, ls=":", alpha=0.9, label="GPU 0 (tail)", rasterized=True)
    if gpu1_tail.size:
        ax1.plot(t1_tail, gpu1_tail, color="tab:orange", lw=0.9, ls=":", alpha=0.9, label="GPU 1 (tail)", rasterized=True)

    ax1.yaxis.set_major_formatter(FuncFormatter(bytes_to_mb))
    ax1.set_ylabel("Memory Usage\n(MB)")
    # ax1.set_title(f"GPU Memory Usage Over Time ({path})")
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.legend(loc="lower center", ncol=3, frameon=True, fancybox=True, framealpha=0.85)

    # ---------- Panel 2: Difference ----------
    pos_mask = (diff_mb_common >= 0)
    neg_mask = ~pos_mask

    # Build NaN-masked arrays so lines don't bridge gaps at sign flips
    diff_pos = np.where(pos_mask, diff_mb_common, np.nan)
    diff_neg = np.where(neg_mask, diff_mb_common, np.nan)

    # Positive region (GPU1 > GPU0)
    ax2.plot(t_common, diff_pos, color="green", lw=0.9, label="GPU1 > GPU0 (Δ)", rasterized=True)
    ax2.fill_between(t_common, 0, diff_mb_common, where=pos_mask, interpolate=True, color="green", alpha=0.15, rasterized=True)

    # Negative region (GPU0 > GPU1)
    ax2.plot(t_common, diff_neg, color="red", lw=0.9, label="GPU0 > GPU1 (Δ)", rasterized=True)
    ax2.fill_between(t_common, 0, diff_mb_common, where=neg_mask, interpolate=True, color="red", alpha=0.15, rasterized=True)

    # Tail differences with shade
    if gpu0_tail.size:  # GPU0 longer
        tail_diff = -to_MB(gpu0_tail)
        ax2.fill_between(t0_tail, 0, tail_diff, color="red", alpha=0.15, rasterized=True)
        ax2.plot(t0_tail, tail_diff, color="red", lw=0.9, ls=":", rasterized=True)
    if gpu1_tail.size:  # GPU1 longer
        tail_diff = to_MB(gpu1_tail)
        ax2.fill_between(t1_tail, 0, tail_diff, color="green", alpha=0.15, rasterized=True)
        ax2.plot(t1_tail, tail_diff, color="green", lw=0.9, ls=":", rasterized=True)

    ax2.axhline(0, color="black", lw=0.8, ls="--")
    # ax2.set_ylabel("GPU1−GPU0\nΔ (MB)")
    ax2.set_ylabel("Δ (MB)")
    ax2.xaxis.set_label_coords(1.0, -0.12)
    # ax2.set_xlabel("Time Step")
    ax2.set_xlabel("(Time Stamp)", loc="right", labelpad=1, fontsize=8)

    ax2.grid(True, ls="--", alpha=0.25)
    ax2.set_ylabel("Δ Memory (MB)\n(GPU1 − GPU0)")
    ax2.legend(loc="upper center", frameon=True, fancybox=True, framealpha=0.85, ncol=3)

    # for ax in (ax1, ax2):
    #     ax.margins(x=0)

    # plt.tight_layout()

    fig_name = f"{output_folder}/memory_over_time_{os.path.basename(path)}"
    fmt = 'pdf'
    fig_filename = f"{fig_name}.{fmt}"
    plt.savefig(fig_filename, format=f'{fmt}', dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()


def main(log_path, output_folder):
    parallelisms = [f"{log_path}/tp", f"{log_path}/dp", f"{log_path}/pp"]
    # parallelisms = ["tp"]
    for path in parallelisms:
        print(f"Drawing {path}...")
        file_name = "tensor_gpu"
        draw_parallelism(path, file_name, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder for the plot"
    )
    args = parser.parse_args()

    main(args.log_path, args.output_folder)

