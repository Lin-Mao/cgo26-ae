import os
import re
import sys
import argparse


def main(log_folder):
    file = open(log_folder, "r")
    line = file.readline()
    while line:
        if line.startswith("[SANITIZER INFO] Malloc tensor") or line.startswith("[SANITIZER INFO] Free tensor"):
            print(line.strip())
        line = file.readline()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-folder",
        type=str,
        required=True,
        help="Log folder path"
    )

    args = parser.parse_args()
    main(args.log_folder)
