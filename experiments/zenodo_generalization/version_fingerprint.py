import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--exp", type=str, default="", help="Optional experiment folder (e.g., 401)")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    files = [
        root / "core" / "dataset.py",
        root / "core" / "model.py",
        root / "core" / "train.py",
        root / "core" / "generalization.py",
        root / "run_experiments.py",
    ]
    if args.exp:
        files.append(root / args.exp / "config.json")

    record = {"project_root": str(root), "exp": args.exp or None, "files": {}, "combined_sha256": ""}
    acc = hashlib.sha256()
    for p in files:
        if not p.exists():
            continue
        s = sha256_file(p)
        record["files"][str(p.relative_to(root))] = s
        acc.update((str(p.relative_to(root)) + ":" + s + "\n").encode("utf-8"))
    record["combined_sha256"] = acc.hexdigest()

    text = json.dumps(record, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[OK] wrote fingerprint -> {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
