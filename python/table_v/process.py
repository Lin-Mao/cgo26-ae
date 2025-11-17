import os
import re
import numpy as np
import sys
import argparse

def _format_size(size):
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size / 1024 / 1024:.2f} MB"


# parse the log file output by the app_analysis tool
def parse_log_file(file_path):
    # {kernel_id: [kernel_name, access_count, tensor_working_set_size, memory_working_set_size, tensor_footprint_size, memory_footprint_size]}
    kernel_dict = {}

    file = open(file_path, "r")
    line = file.readline()
    while line:
        line = line.strip()
        current_kid = None
        current_data = []
        if line.startswith("Kernel ID:"):
            nums = re.findall(r"\d+\.?\d*", line)
            current_kid = int(nums[0])
            line = file.readline()  # Kernel Name
            current_data.append(line.replace("  Kernel Name:", "").strip())
            line = file.readline()  # Access Count
            current_data.append(int(re.findall(r"\d+\.?\d*", line)[0]))
            line = file.readline()  # Tensor Working Set Size
            current_data.append(int(re.findall(r"\d+\.?\d*", line)[0]))
            line = file.readline()  # Memory Working Set Size
            current_data.append(int(re.findall(r"\d+\.?\d*", line)[0]))
            line = file.readline()  # Tensor Footprint Size
            current_data.append(int(re.findall(r"\d+\.?\d*", line)[0]))
            line = file.readline()  # Memory Footprint Size
            current_data.append(int(re.findall(r"\d+\.?\d*", line)[0]))
        if current_kid is not None:
            kernel_dict[current_kid] = current_data
        line = file.readline()
    return kernel_dict


def print_kernel_data(all_kernel_dicts):
    for key, value in all_kernel_dicts.items():
        print(key, len(value.keys()))
        tensor_working_set = []
        memory_working_set = []
        tensor_footprint = []
        memory_footprint = []
        for kernel_id, kernel_data in value.items():
            tensor_working_set.append(kernel_data[2])
            memory_working_set.append(kernel_data[3])
            tensor_footprint.append(kernel_data[4])
            memory_footprint.append(kernel_data[5])
        print(f"{key} ------------------------------------------------------------")
        print("tensor working set == ", tensor_working_set)
        print("memory working set == ", memory_working_set)
        print("tensor footprint == ", tensor_footprint)
        print("memory footprint == ", memory_footprint)


def get_kernel_mean_max_min(all_kernel_dicts, output_folder):
    result_dict = {}
    for key, value in all_kernel_dicts.items():
        tensor_working_set = []
        memory_working_set = []
        tensor_footprint = []
        memory_footprint = []
        kernel_count = 0
        for kernel_id, kernel_data in value.items():
            tensor_working_set.append(kernel_data[2])
            memory_working_set.append(kernel_data[3])
            tensor_footprint.append(kernel_data[4])
            memory_footprint.append(kernel_data[5])
            kernel_count += 1
        # calculate the average, max, min, median, std, and 90th percentile via numpy
        print(f"{key} ------------------------------------------------------------")
        print(f"Kernel count: {kernel_count}")
        # one line for each , one per line
        print(f"Tensor work set - size: {len(tensor_working_set)}")
        print(f"  Max: {np.max(tensor_working_set)} ({_format_size(np.max(tensor_working_set))})")
        print(f"  Min: {np.min(tensor_working_set)} ({_format_size(np.min(tensor_working_set))})")
        print(f"  Average: {np.mean(tensor_working_set)} ({_format_size(np.mean(tensor_working_set))})")
        print(f"  Median: {np.median(tensor_working_set)} ({_format_size(np.median(tensor_working_set))})")
        print(f"  90th percentile: {np.percentile(tensor_working_set, 90)} ({_format_size(np.percentile(tensor_working_set, 90))})")
        # print(f"Memory work set - size: {len(memory_working_set)}")
        # print(f"  Average: {np.mean(memory_working_set)} ({_format_size(np.mean(memory_working_set))})")
        # print(f"  Max: {np.max(memory_working_set)} ({_format_size(np.max(memory_working_set))})")
        # print(f"  Min: {np.min(memory_working_set)} ({_format_size(np.min(memory_working_set))})")
        # print(f"  Median: {np.median(memory_working_set)} ({_format_size(np.median(memory_working_set))})")
        # print(f"  90th percentile: {np.percentile(memory_working_set, 90)} ({_format_size(np.percentile(memory_working_set, 90))})")
        # print(f"Tensor footprint - size: {len(tensor_footprint)}")
        # print(f"  Average: {np.mean(tensor_footprint)} ({_format_size(np.mean(tensor_footprint))})")
        # print(f"  Max: {np.max(tensor_footprint)} ({_format_size(np.max(tensor_footprint))})")
        # print(f"  Min: {np.min(tensor_footprint)} ({_format_size(np.min(tensor_footprint))})")
        # print(f"  Median: {np.median(tensor_footprint)} ({_format_size(np.median(tensor_footprint))})")
        # print(f"  90th percentile: {np.percentile(tensor_footprint, 90)} ({_format_size(np.percentile(tensor_footprint, 90))})")
        print(f"Memory footprint - size: {len(memory_footprint)}")
        print(f"  Average: {np.mean(memory_footprint)} ({_format_size(np.mean(memory_footprint))})")
        # print(f"  Max: {np.max(memory_footprint)} ({_format_size(np.max(memory_footprint))})")
        # print(f"  Min: {np.min(memory_footprint)} ({_format_size(np.min(memory_footprint))})")
        # print(f"  Median: {np.median(memory_footprint)} ({_format_size(np.median(memory_footprint))})")
        # print(f"  90th percentile: {np.percentile(memory_footprint, 90)} ({_format_size(np.percentile(memory_footprint, 90))})")
        print("")

def main(log_folder, output_folder):
    log_files = [f for f in os.listdir(log_folder) if f.endswith(".log")]
    # print(log_files)

    all_kernel_dicts = {}
    for file in log_files:
        kernel_dict = parse_log_file(os.path.join(log_folder, file))
        all_kernel_dicts[file.rstrip("_app_analysis.log")] = kernel_dict
        # print(file.rstrip("_app_analysis.log"),len(kernel_dict.keys()))
    
    get_kernel_mean_max_min(all_kernel_dicts, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-folder",
        type=str,
        required=True,
        help="Log folder path"
    )

    # parser.add_argument(
    #     "--output-folder",
    #     type=str,
    #     required=True,
    #     help="Output folder path"
    # )
    args = parser.parse_args()
    main(args.log_folder, "")
