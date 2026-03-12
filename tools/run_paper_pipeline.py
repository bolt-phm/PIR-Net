import argparse
import copy
import json
import os
import subprocess
from datetime import datetime
from statistics import mean, pstdev

import pandas as pd

DEFAULT_EXPERIMENTS = ['022', '122', '202', '212', '220', '221', '222']
DEFAULT_SNRS = [None, 20, 10, 5, 0, -5]


def parse_csv_list(values, cast_func, allow_none=False):
    out = []
    for v in values:
        if isinstance(v, str):
            parts = [x.strip() for x in v.split(',') if x.strip()]
        else:
            parts = [v]
        for p in parts:
            if allow_none and str(p).lower() == 'clean':
                out.append(None)
            else:
                out.append(cast_func(p))
    return out


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_exp_dir(root, exp):
    """
    Locate an experiment directory.

    To avoid polluting the project root, comparison/baseline experiments can
    live under `root/baselines/<exp>` while legacy experiments stay at
    `root/<exp>`. This function supports both locations.
    """
    candidates = [
        os.path.join(root, exp),
        os.path.join(root, 'baselines', exp),
    ]
    for p in candidates:
        if not os.path.isdir(p):
            continue
        cfg = os.path.join(p, 'config.json')
        if not os.path.exists(cfg):
            raise FileNotFoundError(f'config.json not found in: {p}')
        return p, cfg
    raise FileNotFoundError(f'Experiment folder not found. Tried: {candidates}')


def _fmt_float_for_id(x):
    # 0.7 -> 0p7 (stable, filesystem-friendly)
    s = f"{float(x):.6g}"
    return s.replace(".", "p").replace("-", "m")


def make_run_id(exp, split_mode, alpha, compression_ratio, seed):
    return f"{exp}_split-{split_mode}_a-{_fmt_float_for_id(alpha)}_c-{int(compression_ratio)}_seed-{int(seed)}"


def apply_overrides(cfg, exp, split_mode, alpha, compression_ratio, seed):
    c = copy.deepcopy(cfg)
    c.setdefault('data', {})
    c.setdefault('train', {})

    c['data']['split_mode'] = split_mode
    c['data']['resample_alpha'] = float(alpha)
    c['data']['base_resample_factor'] = int(compression_ratio)
    c['data']['file_split_seed'] = int(seed)
    c['train']['seed'] = int(seed)

    # speed-friendly defaults for RTX 5090 + large RAM
    if 'amp' not in c['train']:
        c['train']['amp'] = True

    # Avoid overwriting checkpoints/logs across different parameter combos.
    run_id = make_run_id(exp, split_mode, alpha, compression_ratio, seed)
    c['train']['run_id'] = run_id
    c['train']['model_dir'] = os.path.join('checkpoints', run_id)
    # Keep logs under the experiment folder, parallel to checkpoints/.
    c['train']['log_dir'] = os.path.join('logs', run_id)

    return c


def maybe_train(exp_dir, root):
    cmd = ['python', 'train.py', '--exp_dir', exp_dir]
    proc = subprocess.run(cmd, cwd=root)
    if proc.returncode != 0:
        raise RuntimeError(f'Training failed: {exp_dir}')


def run_eval(exp_dir, snr, save_artifacts):
    from generalization import evaluate
    return evaluate(exp_dir=exp_dir, force_snr=snr, save_artifacts=save_artifacts)


def aggregate_mean_std(df, group_cols, value_col='accuracy_pct'):
    rows = []
    grouped = df.groupby(group_cols)[value_col].apply(list).reset_index(name='vals')
    for _, r in grouped.iterrows():
        vals = [float(x) for x in r['vals']]
        row = {k: r[k] for k in group_cols}
        row['mean_pct'] = mean(vals)
        row['std_pct'] = pstdev(vals) if len(vals) > 1 else 0.0
        row['runs'] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(df):
    clean_df = df[df['snr_db'] == 'clean'].copy()
    noisy_df = df[df['snr_db'] != 'clean'].copy()

    main_table = aggregate_mean_std(
        clean_df,
        ['experiment', 'split_mode', 'alpha', 'compression_ratio']
    ).sort_values('mean_pct', ascending=False)

    robust_table = aggregate_mean_std(
        noisy_df,
        ['experiment', 'split_mode', 'alpha', 'compression_ratio', 'snr_db']
    ).sort_values(['experiment', 'snr_db'])

    split_compare = aggregate_mean_std(
        clean_df,
        ['split_mode', 'alpha', 'compression_ratio']
    ).sort_values('mean_pct', ascending=False)

    alpha_table = aggregate_mean_std(
        clean_df,
        ['alpha', 'split_mode', 'compression_ratio']
    ).sort_values('mean_pct', ascending=False)

    compression_table = aggregate_mean_std(
        clean_df,
        ['compression_ratio', 'split_mode', 'alpha']
    ).sort_values('mean_pct', ascending=False)

    return {
        'table_main_clean_mean_std.csv': main_table,
        'table_robustness_mean_std.csv': robust_table,
        'table_split_mode_compare.csv': split_compare,
        'table_alpha_sensitivity.csv': alpha_table,
        'table_compression_sensitivity.csv': compression_table,
    }


def sync_sources_to_experiments(root, experiments):
    source_files = ['dataset.py', 'generalization.py', 'model.py', 'train.py']
    for exp in experiments:
        exp_dir, _ = ensure_exp_dir(root, exp)
        for sf in source_files:
            src = os.path.join(root, sf)
            dst = os.path.join(exp_dir, sf)
            if os.path.exists(src):
                with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                    fdst.write(fsrc.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--experiments', nargs='+', default=DEFAULT_EXPERIMENTS)
    # Default to temporal to avoid accidentally doubling runs. Use --split_modes file explicitly for leakage-avoidance control.
    parser.add_argument('--split_modes', nargs='+', default=['temporal'])
    parser.add_argument('--alphas', nargs='+', default=['0.7'])
    parser.add_argument('--compression_ratios', nargs='+', default=['150'])
    # Seed=256 is intentionally removed (user constraint). Add it explicitly if you really need it.
    parser.add_argument('--seeds', nargs='+', default=['3407', '2026'])
    parser.add_argument('--snrs', nargs='+', default=['clean', '20', '10', '5', '0', '-5'])
    parser.add_argument('--out_dir', type=str, default='paper_outputs_round2')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--skip_train_if_exists', action='store_true',
                        help='When --train is set, skip training if best_model already exists for this run_id.')
    parser.add_argument('--no_save_artifacts', action='store_true')
    parser.add_argument('--sync_sources', action='store_true')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(args.out_dir, exist_ok=True)

    split_modes = [s.strip().lower() for s in args.split_modes]
    alphas = parse_csv_list(args.alphas, float)
    compressions = parse_csv_list(args.compression_ratios, int)
    seeds = parse_csv_list(args.seeds, int)
    snrs = parse_csv_list(args.snrs, float, allow_none=True)

    if args.sync_sources:
        sync_sources_to_experiments(root, args.experiments)

    config_backups = {}
    rows = []

    try:
        for exp in args.experiments:
            exp_dir, config_path = ensure_exp_dir(root, exp)
            base_config = load_json(config_path)
            config_backups[config_path] = copy.deepcopy(base_config)

            for split_mode in split_modes:
                for alpha in alphas:
                    for comp in compressions:
                        for seed in seeds:
                            cfg = apply_overrides(base_config, exp, split_mode, alpha, comp, seed)
                            save_json(config_path, cfg)

                            if args.train:
                                if args.skip_train_if_exists:
                                    best_name = cfg.get('train', {}).get('best_model_name', 'best_model_robust_noise.pth')
                                    best_path = os.path.join(exp_dir, cfg['train']['model_dir'], best_name)
                                    if os.path.exists(best_path):
                                        pass
                                    else:
                                        maybe_train(exp_dir=exp_dir, root=root)
                                else:
                                    maybe_train(exp_dir=exp_dir, root=root)

                            for snr in snrs:
                                result = run_eval(
                                    exp_dir=exp_dir,
                                    snr=snr,
                                    save_artifacts=(not args.no_save_artifacts),
                                )
                                rows.append({
                                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                                    'experiment': exp,
                                    'split_mode': split_mode,
                                    'alpha': alpha,
                                    'compression_ratio': comp,
                                    'seed': seed,
                                    'snr_db': 'clean' if snr is None else float(snr),
                                    'accuracy_pct': round(result['accuracy'] * 100.0, 4),
                                    'n_test_samples': result['n_test_samples'],
                                    'n_models': result['n_models'],
                                })
    finally:
        for p, backup in config_backups.items():
            save_json(p, backup)

    all_runs = pd.DataFrame(rows)
    all_runs_path = os.path.join(args.out_dir, 'all_runs.csv')
    all_runs.to_csv(all_runs_path, index=False)

    tables = build_tables(all_runs)
    for filename, table_df in tables.items():
        table_df.to_csv(os.path.join(args.out_dir, filename), index=False)

    print(f'[OK] saved: {all_runs_path}')
    for filename in tables:
        print(f'[OK] saved: {os.path.join(args.out_dir, filename)}')


if __name__ == '__main__':
    main()
