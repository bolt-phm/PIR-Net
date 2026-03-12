import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / 'lw2' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

items = [
    ('301', 'WDCNN', ROOT / 'baselines' / 'A' / 'baselines' / '301' / 'logs' / 'metrics_clean.json'),
    ('302', 'InceptionTime', ROOT / 'baselines' / 'A' / 'baselines' / '302' / 'logs' / 'metrics_clean.json'),
    ('303', 'ResNet1D', ROOT / 'baselines' / 'B' / 'baselines' / '303' / 'logs' / 'metrics_clean.json'),
    ('304', 'ResNet-18\nspectrogram', ROOT / 'baselines' / 'C' / 'baselines' / '304' / 'logs' / 'metrics_clean.json'),
    ('305', 'EfficientNet-B0\nspectrogram', ROOT / 'baselines' / 'D' / 'baselines' / '305' / 'logs' / 'metrics_clean.json'),
    ('306', 'Transformer\nencoder', ROOT / 'baselines' / 'B' / 'baselines' / '306' / 'logs' / 'metrics_clean.json'),
]

labels, values, colors = [], [], []
for exp, label, path in items:
    data = json.loads(path.read_text(encoding='utf-8'))
    labels.append(f'{exp}\n{label}')
    values.append(data['accuracy_pct'])
    colors.append('#2A6F97' if exp in {'301', '302', '303', '306'} else '#C97C5D')

plt.figure(figsize=(10.5, 5.5))
bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.title('External baselines under the non-PIR protocol (clean test split)')
plt.grid(axis='y', linestyle='--', alpha=0.25)

for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.text(0.01, 0.98, 'Waveform baselines: blue   Spectrogram baselines: brown',
         transform=plt.gca().transAxes, ha='left', va='top', fontsize=9)
plt.text(0.01, 0.91, 'Protocol note: all baselines avoid PIR preprocessing; test set n = 22,640',
         transform=plt.gca().transAxes, ha='left', va='top', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_rev2_external_baselines_clean.png', dpi=220)
