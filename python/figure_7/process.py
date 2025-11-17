import os
import re
import sys
import argparse



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


def _get_kernel_name(kernel_dict, out_filename=""):
    # write to file or print to console
    if out_filename == "":
        std_out = sys.stdout
    else:
        std_out = open(out_filename, "w")

    for _, kernel_data in kernel_dict.items():
        std_out.write(kernel_data[0] + "\n")
    std_out.close()


def get_all_kernel_name(all_kernel_dicts, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for key, value in all_kernel_dicts.items():
        print(f"Processing {output_folder}/{key}_kernel_name.txt... Done")
        _get_kernel_name(value, f"{output_folder}/{key}_kernel_name.txt")

def main(root_folder, output_folder):
    # log_folder = f"{root_folder}/train_raw"
    log_folder = f"{root_folder}"
    log_files = [f for f in os.listdir(log_folder) if f.endswith(".log")]
    # print(log_files)

    all_kernel_dicts = {}
    for file in log_files:
        kernel_dict = parse_log_file(os.path.join(log_folder, file))
        all_kernel_dicts[file.rstrip("_app_analysis.log")] = kernel_dict
        # print(file.rstrip("_app_analysis.log"),len(kernel_dict.keys()))
    
    get_all_kernel_name(all_kernel_dicts, output_folder)

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
