import os
import re
import sys
import argparse


def main(log_file, output_folder):
    # open two files, tensor_gpu_0.txt and tensor_gpu_1.txt
    file0 = open(f"{output_folder}/tensor_gpu_0.txt", "w")
    file1 = open(f"{output_folder}/tensor_gpu_1.txt", "w")

    file = open(log_file, "r")
    line = file.readline()
    while line:
        if line.startswith("Malloc tensor") or line.startswith("Free tensor"):
            nums = re.findall(r"\d+\.?\d*", line)
            device = int(nums[-1])
            allocated_size = int(nums[-3])
            if device == 0:
                file0.write(f"{allocated_size}\n")
            elif device == 1:
                file1.write(f"{allocated_size}\n")
        line = file.readline()

    file0.close()
    file1.close()
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Log file path"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder path"
    )

    args = parser.parse_args()
    main(args.log_file, args.output_folder)
