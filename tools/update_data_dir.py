#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def update_config(path: Path, data_dir: str, generalization_dir: str | None) -> bool:
    with open(path, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)

    changed = False
    cfg.setdefault('data', {})

    if cfg['data'].get('data_dir') != data_dir:
        cfg['data']['data_dir'] = data_dir
        changed = True

    if generalization_dir is not None and cfg['data'].get('generalization_dir') != generalization_dir:
        cfg['data']['generalization_dir'] = generalization_dir
        changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write('\n')

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch-update data_dir in all experiment config.json files.')
    parser.add_argument('--root', type=str, default='.', help='Package root path (default: current directory)')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory path to write into config.json files')
    parser.add_argument('--generalization_dir', type=str, default=None, help='Optional path for data.generalization_dir')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    search_paths = [
        root / 'experiments' / 'pirnet_ablation',
        root / 'experiments' / 'external_baselines',
    ]

    configs: list[Path] = []
    for sp in search_paths:
        if sp.exists():
            configs.extend(sorted(sp.rglob('config.json')))

    if not configs:
        print('No config.json files found. Please check --root path.')
        return

    updated = 0
    for cfg_path in configs:
        if update_config(cfg_path, args.data_dir, args.generalization_dir):
            updated += 1
            print(f'[UPDATED] {cfg_path}')
        else:
            print(f'[SKIPPED] {cfg_path}')

    print(f'\nDone. Updated {updated}/{len(configs)} config files.')


if __name__ == '__main__':
    main()
