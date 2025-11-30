import os
import re
import sys
import argparse


def main(log_file):
    file = open(log_file, "r")
    line = file.readline()
    while line:
        if line.startswith("[SANITIZER INFO] Malloc tensor") or line.startswith("[SANITIZER INFO] Free tensor"):
            print(line.strip().replace("[SANITIZER INFO] ", ""))
        line = file.readline()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Log file path"
    )

    args = parser.parse_args()
    main(args.log_file)
