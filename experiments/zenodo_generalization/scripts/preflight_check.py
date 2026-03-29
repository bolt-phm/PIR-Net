import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZROOT = ROOT


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[OK] {msg}")


def check_exists(paths):
    for p in paths:
        if not p.exists():
            fail(f"Missing required path: {p}")
        ok(f"exists: {p}")


def check_compile(py_files):
    cmd = [sys.executable, "-m", "compileall", "-q"] + [str(p) for p in py_files]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        fail("Python compileall failed.")
    ok("compileall passed")


def check_scripts():
    sdir = ROOT / "scripts"
    scripts = [
        "server01_main_part1.sh",
        "server02_main_part2.sh",
        "server03_main_multiseed.sh",
        "server04_sensitivity_and_leakage.sh",
        "server05_robustness_pir.sh",
        "server06_baselines_and_robustness.sh",
        "progress_all.sh",
    ]
    check_exists([sdir / x for x in scripts])

    s1 = (sdir / "server01_main_part1.sh").read_text(encoding="utf-8")
    s2 = (sdir / "server02_main_part2.sh").read_text(encoding="utf-8")
    s3 = (sdir / "server03_main_multiseed.sh").read_text(encoding="utf-8")
    s4 = (sdir / "server04_sensitivity_and_leakage.sh").read_text(encoding="utf-8")
    s5 = (sdir / "server05_robustness_pir.sh").read_text(encoding="utf-8")
    s6 = (sdir / "server06_baselines_and_robustness.sh").read_text(encoding="utf-8")

    # Fold-balanced schedule:
    #   - S1/S2/S3: 401+403 (signal), 501+503 (image), generalization on 501 only
    #   - S4/S5/S6: 402+404 (signal), 502 (image), generalization on 404 only
    if "--folds fold_d15_cfg22" not in s1:
        fail("server01 must target fold_d15_cfg22.")
    if "--folds fold_d16_cfg21" not in s2:
        fail("server02 must target fold_d16_cfg21.")
    if "--folds fold_d17_cfg21" not in s3:
        fail("server03 must target fold_d17_cfg21.")
    if "--folds fold_d15_cfg22" not in s4:
        fail("server04 must target fold_d15_cfg22.")
    if "--folds fold_d16_cfg21" not in s5:
        fail("server05 must target fold_d16_cfg21.")
    if "--folds fold_d17_cfg21" not in s6:
        fail("server06 must target fold_d17_cfg21.")

    for name, txt in [("server01", s1), ("server02", s2), ("server03", s3)]:
        if "--experiments 401 403" not in txt:
            fail(f"{name} must include signal controls 401+403.")
        if "--experiments 501 503" not in txt:
            fail(f"{name} must include PIR(501)+decimate(503).")
        if "--generalization_experiments 501" not in txt:
            fail(f"{name} must run noisy protocol for 501.")

    for name, txt in [("server04", s4), ("server05", s5), ("server06", s6)]:
        if "--experiments 402 404" not in txt:
            fail(f"{name} must include signal controls 402+404.")
        if "--experiments 502" not in txt:
            fail(f"{name} must include average ablation 502.")
        if "--generalization_experiments 404" not in txt:
            fail(f"{name} must run noisy protocol for 404.")

    # User hard constraint: non-main models patience <= 10.
    for name, txt in [
        ("server01", s1),
        ("server02", s2),
        ("server03", s3),
        ("server04", s4),
        ("server05", s5),
        ("server06", s6),
    ]:
        if "--patience_other 10" not in txt:
            fail(f"{name} must keep non-main patience at 10.")

    # No legacy internal full-sweep tasks in supplement-only pack.
    banned_tokens = ["run_paper_pipeline.py", "--experiments 022", "--experiments 212", "baselines/307"]
    for token in banned_tokens:
        if token in s1 + s2 + s3 + s4 + s5 + s6:
            fail(f"Found legacy non-supplement token in scripts: {token}")

    ok("server scripts validated")


def check_layout():
    required = [
        ZROOT / "run_cv_experiments.py",
        ZROOT / "version_fingerprint.py",
        ZROOT / "core" / "dataset.py",
        ZROOT / "core" / "train.py",
        ZROOT / "core" / "generalization.py",
    ]
    for exp in ["401", "402", "403", "404", "501", "502", "503"]:
        required.append(ZROOT / exp / "config.json")
        required.append(ZROOT / exp / "train.py")
        required.append(ZROOT / exp / "generalization.py")
    check_exists(required)


def print_expected_workload():
    print("[INFO] Supplement workload (2nd-review only):")
    print("  - 7 experiments x (3 folds x 3 seeds) = 63 clean CV train runs")
    print("  - Noisy protocol only on 404 and 501: 2 x (3 folds x 3 seeds) = 18 generalization runs")
    print("  - Reviewer-3 mapping:")
    print("    #1 External PIR evaluation: 501 vs 401/402/403/404")
    print("    #3 Feature-engineering A/B/C:")
    print("       A = 502 (average downsampling, same model)")
    print("       B = 501 (physics-informed downsampling, same model)")
    print("       C = 503 + 401/402/403/404 (decimate + other feature pipelines/models)")


def main():
    print(f"[INFO] preflight root: {ROOT}")
    check_layout()
    check_scripts()
    check_compile(
        [
            ZROOT / "run_cv_experiments.py",
            ZROOT / "core" / "dataset.py",
            ZROOT / "core" / "train.py",
            ZROOT / "core" / "generalization.py",
        ]
    )
    print_expected_workload()
    print("[PASS] Preflight check completed.")


if __name__ == "__main__":
    main()
