import os
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
file_list = [
    "run_alexnet.accelprof.log",
    "run_bert.accelprof.log",
    "run_gpt2.accelprof.log",
    "run_resnet18.accelprof.log",
    "run_resnet34.accelprof.log",
    "run_whisper.accelprof.log",
    "test_gpu_alexnet.accelprof.log",
    "test_gpu_bert.accelprof.log",
    "test_gpu_gpt2.accelprof.log",
    "test_gpu_resnet18.accelprof.log",
    "test_gpu_resnet34.accelprof.log",
    "test_gpu_whisper.accelprof.log",
    "test_cpu_alexnet.accelprof.log",
    "test_cpu_bert.accelprof.log",
    "test_cpu_gpt2.accelprof.log",
    "test_cpu_resnet18.accelprof.log",
    "test_cpu_resnet34.accelprof.log",
    "test_cpu_whisper.accelprof.log",
    "test_nvbit_alexnet.accelprof.log",
    "test_nvbit_bert.accelprof.log",
    "test_nvbit_gpt2.accelprof.log",
    "test_nvbit_resnet18.accelprof.log",
    "test_nvbit_resnet34.accelprof.log",
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
    "alexnet": 30,
    "resnet18": 30,
    "resnet34": 50,
    "bert": 10,
    "gpt2": 10,
    "whisper": 10,
}

def main(path, suffix):

    result = {}

    for file_name in file_list:
        if os.path.exists(f"{path}/{file_name}"):
            seconds = extract_elapsed_time_from_file(f"{path}/{file_name}")
            if seconds is not None:
                file_name = file_name.replace(".accelprof.log", "")
                file_name = file_name.replace("test_", "")
                result[file_name] = seconds
            else:
                # print(f"Warning: No valid elapsed time found in {file_name}")
                pass
        else:
            print(f"Warning: File not found: {file_name}")

    # Print the result dictionary
    # print(result)
    # print()

    orig_time = {}
    for model in models:
        orig_time[model] = result["run_" + model]

    gpu_time = {}
    for model in models:
        gpu_time[model] = result["gpu_" + model]

    cpu_time = {}
    for model in models:
        cpu_time[model] = result["cpu_" + model] * sample_rate[model]

    nvbit_time = {}
    for model in models:
        if model == "whisper":
            continue
        nvbit_time[model] = result["nvbit_" + model] * sample_rate[model]


    if suffix != "":
        print(f"orig_time_{suffix} =", orig_time)
        print(f"gpu_time_{suffix} =", gpu_time)
        print(f"cpu_time_{suffix} =", cpu_time)
        print(f"nvbit_time_{suffix} =", nvbit_time)
    else:
        print(f"orig_time =", orig_time)
        print(f"gpu_time =", gpu_time)
        print(f"cpu_time =", cpu_time)
        print(f"nvbit_time =", nvbit_time)


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

