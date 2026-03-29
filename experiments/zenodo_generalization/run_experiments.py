import argparse
import os
import subprocess
import sys


def run_cmd(cmd, cwd):
    print(f"[RUN] {' '.join(cmd)} (cwd={cwd})")
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["401", "402", "403", "404", "501", "502", "503"])
    parser.add_argument("--with_generalization", action="store_true")
    parser.add_argument(
        "--generalization_experiments",
        nargs="*",
        default=[],
        help="Optional subset for generalization; if empty, applies to all experiments.",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable
    gen_subset = {str(x) for x in (args.generalization_experiments or [])}

    for exp in args.experiments:
        exp_dir = os.path.join(root, exp)
        run_cmd([py, "train.py", "--exp_dir", exp_dir], cwd=exp_dir)
        if args.with_generalization and (len(gen_subset) == 0 or str(exp) in gen_subset):
            run_cmd([py, "generalization.py", "--exp_dir", exp_dir], cwd=exp_dir)


if __name__ == "__main__":
    main()
