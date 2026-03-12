import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from dataset import create_dataloaders
from model import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inject_awgn_tensor(inputs, snr_db):
    if snr_db is None:
        return inputs
    flat_inputs = inputs.view(inputs.size(0), -1)
    signal_power = torch.mean(flat_inputs ** 2, dim=1, keepdim=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(inputs)
    view_shape = [inputs.size(0)] + [1] * (inputs.ndim - 1)
    scale = torch.sqrt(noise_power).view(*view_shape)
    return inputs + noise * scale


class EnsembleModel(torch.nn.Module):
    def __init__(self, config, model_paths, device):
        super().__init__()
        self.models = torch.nn.ModuleList()
        for path in model_paths:
            if not os.path.exists(path):
                continue
            model = build_model(config).to(device)
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            self.models.append(model)
        if len(self.models) == 0:
            raise RuntimeError('No valid model weights loaded.')

    def forward(self, img, sig):
        outputs = [m(img, sig) for m in self.models]
        probs = [torch.softmax(out, dim=1) for out in outputs]
        avg_probs = torch.mean(torch.stack(probs), dim=0)
        return torch.log(avg_probs + 1e-8)


def load_config(exp_dir):
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_model_paths(config, exp_dir):
    paths = []
    for p in config.get('inference', {}).get('ensemble_models', []):
        if not p:
            continue
        paths.append(p if os.path.isabs(p) else os.path.join(exp_dir, p))

    if not paths:
        model_dir = config['train']['model_dir']
        candidate_dirs = []
        if os.path.isabs(model_dir):
            candidate_dirs.append(model_dir)
        else:
            candidate_dirs.append(os.path.join(exp_dir, model_dir))
            # Backward compatibility for older train.py behavior
            candidate_dirs.append(os.path.join(os.getcwd(), model_dir))

        for md in candidate_dirs:
            candidate = os.path.join(md, config['train']['best_model_name'])
            if os.path.exists(candidate):
                paths.append(candidate)
                break
        if not paths:
            # Keep first candidate for explicit error reporting downstream
            paths.append(os.path.join(candidate_dirs[0], config['train']['best_model_name']))
    return paths


def evaluate(exp_dir='.', force_snr=None, save_artifacts=True):
    config = load_config(exp_dir)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')

    original_data_dir = config['data']['data_dir']
    if not os.path.isabs(original_data_dir):
        config['data']['data_dir'] = os.path.join(exp_dir, original_data_dir)

    model_paths = resolve_model_paths(config, exp_dir)
    model = EnsembleModel(config, model_paths, device)

    _, _, test_loaders = create_dataloaders(config)
    if not test_loaders:
        raise RuntimeError('No test loaders created. Check split settings and dataset path.')

    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, loader in test_loaders.items():
            for batch in tqdm(loader, desc='Testing'):
                if len(batch) == 3:
                    img, sig, lbl = batch
                else:
                    img, sig, lbl, _ = batch
                if img is None:
                    continue

                img = img.to(device)
                sig = sig.to(device)
                lbl = lbl.to(device)

                if force_snr is not None:
                    sig = inject_awgn_tensor(sig, force_snr)
                    img = inject_awgn_tensor(img, force_snr)

                output = model(img, sig)
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    target_names = config['data'].get('case_ids', [str(i) for i in range(config['data']['num_classes'])])
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    result = {
        'exp_dir': os.path.abspath(exp_dir),
        'snr_db': force_snr,
        'accuracy': float(acc),
        'n_test_samples': int(len(all_labels)),
        'n_models': int(len(model.models)),
    }

    if save_artifacts:
        log_dir = config['train']['log_dir']
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(exp_dir, log_dir)
        os.makedirs(log_dir, exist_ok=True)

        suffix = f'_snr{int(force_snr)}' if force_snr is not None else '_clean'
        cm_path = os.path.join(log_dir, f'confusion_matrix{suffix}.png')
        report_path = os.path.join(log_dir, f'classification_report{suffix}.txt')

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix (Acc={acc:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        result['confusion_matrix_path'] = cm_path
        result['report_path'] = report_path

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='.')
    parser.add_argument('--snr_db', type=float, default=None)
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--json_out', type=str, default='')
    args = parser.parse_args()

    res = evaluate(exp_dir=args.exp_dir, force_snr=args.snr_db, save_artifacts=(not args.no_save))
    logging.info(f"FINAL RESULT ACCURACY: {res['accuracy']:.4f}")

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
