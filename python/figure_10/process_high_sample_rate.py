import os
import re
import argparse

def parse_elapsed_time_to_seconds(time_str):
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600 + m * 60 + s

def extract_elapsed_time_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1]
        if "[ACCELPROF INFO] ELAPSED TIME" in last_line:
            time_part = last_line.strip().split(":")[-3:]  # get last 3 parts
            return parse_elapsed_time_to_seconds(":".join(time_part))
    return None

# List of files to parse
orig_file_list = [
    "run_alexnet.accelprof.log",
    "run_resnet18.accelprof.log",
    "run_resnet34.accelprof.log",
    "run_bert.accelprof.log",
    "run_gpt2.accelprof.log",
    "run_whisper.accelprof.log",
]

gpu_file_list = [
    "test_gpu_alexnet.accelprof.log",
    "test_gpu_resnet18.accelprof.log",
    "test_gpu_resnet34.accelprof.log",
    "test_gpu_bert.accelprof.log",
    "test_gpu_gpt2.accelprof.log",
    "test_gpu_whisper.accelprof.log",
]

# List of file contain breakdown
cpu_file_list = [
    "test_cpu_alexnet.accelprof.log",
    "test_cpu_resnet18.accelprof.log",
    "test_cpu_resnet34.accelprof.log",
    "test_cpu_bert.accelprof.log",
    "test_cpu_gpt2.accelprof.log",
    "test_cpu_whisper.accelprof.log",
]

nvbit_file_list = [
    "test_nvbit_alexnet.accelprof.log",
    "test_nvbit_resnet18.accelprof.log",
    "test_nvbit_resnet34.accelprof.log",
    "test_nvbit_bert.accelprof.log",
    "test_nvbit_gpt2.accelprof.log",
    "test_nvbit_whisper.accelprof.log",
]

models = [
    "alexnet",
    "resnet18",
    "resnet34",
    "bert",
    "gpt2",
    "whisper",
]

sample_rate={
    "alexnet": 60,
    "resnet18": 60,
    "resnet34": 100,
    "bert": 20,
    "gpt2": 20,
    "whisper": 20,
}


def get_orig_time(path):
    result = {}

    for file_name in orig_file_list:
        if os.path.exists(f"{path}/{file_name}"):
            seconds = extract_elapsed_time_from_file(f"{path}/{file_name}")
            if seconds is not None:
                file_name = file_name.replace(".accelprof.log", "")
                file_name = file_name.replace("test_", "")
                result[file_name] = seconds
            else:
                print(f"Warning: No valid elapsed time found in {file_name}")
        else:
            print(f"Warning: File not found: {file_name}")

    # Print the result dictionary
    # print(result)
    # print()

    orig_time = {}
    for model in models:
        orig_time[model] = result["run_" + model]
    return orig_time


def get_time_breakdown(path, file_list, prefix_str, is_sample):
    trace_collection = {}
    trace_transfer = {}
    analysis = {}
    for file_name in file_list:
        if os.path.exists(f"{path}/{file_name}"):
            with open(f"{path}/{file_name}", 'r') as f:
                lines = f.readlines()
                if not lines:
                    return None
                for i in range(-1, -21, -1):
                    line = lines[i]
                    if "Trace collection time" in line:
                        file_name = file_name.replace(".accelprof.log", "")
                        file_name = file_name.replace(prefix_str, "")
                        nums = re.findall(r"\d+\.?\d*", line)
                        time = float(nums[0])
                        if is_sample:
                            time *= sample_rate[file_name]
                        trace_collection[file_name] = time
                    elif "Trace transfer time" in line:
                        file_name = file_name.replace(".accelprof.log", "")
                        file_name = file_name.replace(prefix_str, "")
                        nums = re.findall(r"\d+\.?\d*", line)
                        time = float(nums[0])
                        if is_sample:
                            time *= sample_rate[file_name]
                        trace_transfer[file_name] = time
                    elif "Trace analysis time" in line:
                        file_name = file_name.replace(".accelprof.log", "")
                        file_name = file_name.replace(prefix_str, "")
                        nums = re.findall(r"\d+\.?\d*", line)
                        time = float(nums[0])
                        if is_sample:
                            time *= sample_rate[file_name]
                        analysis[file_name] = time

    return trace_collection, trace_transfer, analysis


def main(path, suffix):

    orig_time = get_orig_time(path)

    gpu_collection, gpu_transfer, gpu_analysis = get_time_breakdown(path, gpu_file_list, "test_gpu_", False)

    cpu_collection, cpu_transfer, cpu_analysis = get_time_breakdown(path, cpu_file_list, "test_cpu_", True)

    nvbit_collection, nvbit_transfer, nvbit_analysis = get_time_breakdown(path, nvbit_file_list, "test_nvbit_", True)

    gpu_total = {}
    for model in models:
        if  model in gpu_collection and model in gpu_transfer and model in gpu_analysis and model in orig_time:
            gpu_total[model] = gpu_collection[model] + gpu_transfer[model] + gpu_analysis[model] + orig_time[model]

    cpu_total = {}
    for model in models:
        if  model in cpu_collection and model in cpu_transfer and model in cpu_analysis and model in orig_time:
            cpu_total[model] = cpu_collection[model] + cpu_transfer[model] + cpu_analysis[model] + orig_time[model]
    
    nvbit_total = {}
    for model in models:
        if  model in nvbit_collection and model in nvbit_transfer and model in nvbit_analysis and model in orig_time:
            nvbit_total[model] = nvbit_collection[model] + nvbit_transfer[model] + nvbit_analysis[model] + orig_time[model]

    suffix = f"_{suffix}" if suffix != "" else ""

    print(f"orig_time{suffix} =", orig_time)

    print()
    print(f"gpu_collection{suffix} =", gpu_collection)
    print(f"gpu_transfer{suffix} =", gpu_transfer)
    print(f"gpu_analysis{suffix} =", gpu_analysis)
    print(f"gpu_total{suffix} =", gpu_total)

    print()
    print(f"cpu_collection{suffix} =", cpu_collection)
    print(f"cpu_transfer{suffix} =", cpu_transfer)
    print(f"cpu_analysis{suffix} =", cpu_analysis)
    print(f"cpu_total{suffix} =", cpu_total)

    print()
    print(f"nvbit_collection{suffix} =", nvbit_collection)
    print(f"nvbit_transfer{suffix} =", nvbit_transfer)
    print(f"nvbit_analysis{suffix} =", nvbit_analysis)
    print(f"nvbit_total{suffix} =", nvbit_total)


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
