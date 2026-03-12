import argparse
import os
import subprocess
import sys


def run_cmd(cmd, cwd):
    print(f"[RUN] {' '.join(cmd)} (cwd={cwd})")
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["401", "402", "403", "404"])
    parser.add_argument("--with_generalization", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable

    for exp in args.experiments:
        exp_dir = os.path.join(root, exp)
        run_cmd([py, "train.py", "--exp_dir", exp_dir], cwd=exp_dir)
        if args.with_generalization:
            run_cmd([py, "generalization.py", "--exp_dir", exp_dir], cwd=exp_dir)


if __name__ == "__main__":
    main()

