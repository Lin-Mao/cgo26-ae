import re
import json
from copy import deepcopy
import os
import argparse


def parse_trace_to_dict(trace_text):
    with open(trace_text, "r") as file:
        line = file.readline()
        global_dict = dict()
        section_dict = None
        current_model = None
        current_section = None
        while line:
            if "NO PREFETCH" in line and "---" in line:
                # Save previous section if it exists
                if current_section and section_dict:
                    global_dict[current_section] = deepcopy(section_dict)
                current_section = "No_Prefetch"
                section_dict = {"alexnet": [], "resnet18": [], "resnet34": [], "bert": [], "gpt2": [], "whisper": []}
            elif "OBJECT LEVEL PREFETCH" in line and "---" in line:
                # Save previous section if it exists
                if current_section and section_dict:
                    global_dict[current_section] = deepcopy(section_dict)
                current_section = "Object-Level"
                section_dict = {"alexnet": [], "resnet18": [], "resnet34": [], "bert": [], "gpt2": [], "whisper": []}
            elif "TENSOR LEVEL PREFETCH" in line and "---" in line:
                # Save previous section if it exists
                if current_section and section_dict:
                    global_dict[current_section] = deepcopy(section_dict)
                current_section = "Tensor-Level"
                section_dict = {"alexnet": [], "resnet18": [], "resnet34": [], "bert": [], "gpt2": [], "whisper": []}

            if "Running" in line:
                current_model = line.split(" ")[1]

            if "All time taken" in line and section_dict and current_model:
                elapsed_time = re.search(r"All time taken.*?([\d.]+) seconds", line)
                if elapsed_time:
                    section_dict[current_model].append(float(elapsed_time.group(1)))
            line = file.readline()
        
        # Save the last section
        if current_section and section_dict:
            global_dict[current_section] = deepcopy(section_dict)

    return global_dict


def main(log_folder, suffix):
    path = f"{log_folder}/uvm_advisor.log"

    result = parse_trace_to_dict(path)

    processed_result = dict()

    for key, value in result.items():
        model_dict = dict()
        for model, time in value.items():
            model_dict[model] = round(sum(time) / len(time), 2)
        processed_result[key] = model_dict

    # print(processed_result)

    suffix = f"_{suffix}" if suffix else ""
    for key, value in processed_result.items():
        print(f"{key.lower().replace('-', '_')}{suffix} =", value)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-folder",
        type=str,
        required=True,
        help="Log folder path"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        help="Suffix",
        default=""
    )
    args = parser.parse_args()
    main(args.log_folder, args.suffix)
