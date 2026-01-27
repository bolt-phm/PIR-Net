import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import logging
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

# 引入自定义模块
from dataset import create_dataloaders, run_offline_preprocessing
from model import build_model

# 禁用 OneDNN 优化以避免某些环境下的警告/报错
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------- #
#       工具函数
# ----------------------------- #

def set_seed(seed=42):
    """ 固定所有随机源 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # 开启可复现模式(可能会降低速度)

# ----------------------------- #
#       损失函数定义
# ----------------------------- #

class FocalLoss(nn.Module):
    """
    Focal Loss: 专注于难分类样本 (辅助 Loss)
    """
    def __init__(self, num_classes=7, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
            
    def forward(self, pred, target):
        device = pred.device
        
        # 展平处理
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, pred.size(2))
            target = target.view(-1, 1)
        
        target = target.view(-1, 1)
        
        logpt = F.log_softmax(pred, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.alpha is not None:
            if self.alpha.device != device:
                self.alpha = self.alpha.to(device)
            at = self.alpha.gather(0, target.view(-1))
            loss = loss * at

        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss


class CombinedLSLoss(nn.Module):
    """
    组合损失函数
    结构: Ratio * LabelSmoothing + (1 - Ratio) * FocalLoss
    目的: Label Smoothing 防止过拟合 (主导)，Focal Loss 挖掘难样本 (辅助)
    """
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1, ratio=0.8):
        super(CombinedLSLoss, self).__init__()
        self.ratio = ratio
        
        # 1. 主导部分: Label Smoothing CrossEntropy
        self.ce_smooth = nn.CrossEntropyLoss(weight=alpha, label_smoothing=smoothing)
        
        # 2. 辅助部分: Focal Loss
        self.focal = FocalLoss(num_classes=num_classes, alpha=alpha, gamma=gamma)

    def forward(self, pred, target):
        loss_ce = self.ce_smooth(pred, target)
        loss_focal = self.focal(pred, target)
        return self.ratio * loss_ce + (1 - self.ratio) * loss_focal


# ----------------------------- #
#       Mixup 逻辑
# ----------------------------- #

def mixup_data(x_img, x_sig, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x_img.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x_img = lam * x_img + (1 - lam) * x_img[index, :]
    mixed_x_sig = lam * x_sig + (1 - lam) * x_sig[index, :]
    
    y_a, y_b = y, y[index]
    return mixed_x_img, mixed_x_sig, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------- #
#       辅助类
# ----------------------------- #

class EarlyStopping:
    def __init__(self, patience=10, path='checkpoint.pth', verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logging.info(f'Validation loss decreased. Saving best model...')
        torch.save(model.state_dict(), self.path)

    def state_dict(self):
        return {'best_score': self.best_score, 'counter': self.counter}

    def load_state(self, state_dict):
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

# ----------------------------- #
#       训练流程
# ----------------------------- #

def train_epoch(model, loaders, criterion, optimizer, device, epoch, config, writer=None, scheduler=None, warmup_epochs=0):
    model.train()
    total_loss, total_acc = 0, 0
    
    accum_steps = max(1, int(config['train'].get('accumulation_steps', 1)))
    mixup_alpha = config['train'].get('mixup_alpha', 0.0) 
    use_mixup = mixup_alpha > 0
    base_lr = config['train']['learning_rate']
    
    total_batches = sum(len(l) for l in loaders.values())
    if total_batches == 0: return 0, 0
    
    iterators = {k: iter(v) for k, v in loaders.items()}
    keys = list(loaders.keys())
    
    optimizer.zero_grad()
    pbar = tqdm(range(total_batches), desc=f"[Train Ep {epoch+1}]", leave=False)
    
    for i in pbar:
        # Warmup
        if epoch < warmup_epochs:
            global_step = epoch * total_batches + i
            total_warmup_steps = warmup_epochs * total_batches
            warmup_lr = base_lr * (global_step + 1) / total_warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Data Loading
        key = keys[i % len(keys)]
        try:
            batch_data = next(iterators[key])
        except StopIteration:
            iterators[key] = iter(loaders[key])
            batch_data = next(iterators[key])
            
        img, sig, lbl = batch_data
        if img is None: continue 
        
        img, sig, lbl = img.to(device), sig.to(device), lbl.to(device)
        
        # Mixup Forward
        if use_mixup:
            img, sig, targets_a, targets_b, lam = mixup_data(img, sig, lbl, mixup_alpha, device)
            output = model(img, sig)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            output = model(img, sig)
            loss = criterion(output, lbl)
        
        loss = loss / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == total_batches:
            optimizer.step()
            optimizer.zero_grad()
            
            # Cosine Scheduler
            if scheduler is not None and epoch >= warmup_epochs:
                scheduler.step((epoch - warmup_epochs) + i / total_batches)
        
        current_loss = loss.item() * accum_steps
        total_loss += current_loss
        
        _, pred = torch.max(output, 1)
        acc = (pred == lbl).sum().item() / lbl.size(0)
        total_acc += acc
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=current_loss, lr=f"{current_lr:.2e}", mixup=use_mixup)
        
        if writer:
            step_idx = epoch * total_batches + i
            writer.add_scalar('Loss/train_step', current_loss, step_idx)
            writer.add_scalar('LR', current_lr, step_idx)

    return total_loss / total_batches, total_acc / total_batches


def validate(model, loaders, criterion, device, epoch, writer=None):
    model.eval()
    total_loss, total_acc = 0, 0
    total_batches = sum(len(l) for l in loaders.values())
    
    if total_batches == 0: return 0, 0
    
    with torch.no_grad():
        for scale_key, loader in loaders.items():
            for batch in loader:
                img, sig, lbl = batch
                if img is None: continue
                img, sig, lbl = img.to(device), sig.to(device), lbl.to(device)
                
                output = model(img, sig)
                loss = criterion(output, lbl)
                
                total_loss += loss.item()
                _, pred = torch.max(output, 1)
                total_acc += (pred == lbl).sum().item() / lbl.size(0)

    avg_loss = total_loss / total_batches
    avg_acc = total_acc / total_batches
    
    if writer:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', avg_acc, epoch)
        
    return avg_loss, avg_acc


# ----------------------------- #
#       主程序
# ----------------------------- #

def main():
    config_file = 'config.json'
    if not os.path.exists(config_file):
        logging.error("config.json not found.")
        return
        
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 1. 设置随机种子
    seed = config['train'].get('seed', 42)
    set_seed(seed)
    logging.info(f"Random Seed set to: {seed}")

    # 2. 模式与路径
    use_offline = config['data'].get('use_offline', False)
    if use_offline:
        run_offline_preprocessing(config)
    else:
        logging.info("Online Mode Enabled.")

    log_dir = config['train']['log_dir']
    model_dir = config['train']['model_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    writer = SummaryWriter(log_dir)

    # 3. 数据加载
    logging.info("Initializing Data Loaders...")
    train_loaders, val_loaders, test_loaders = create_dataloaders(config)
    
    # 4. 模型构建
    model = build_model(config).to(device)
    
    # 5. 优化器
    base_lr = config['train']['learning_rate']
    wd = config['train'].get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    
    # 6. Loss Setup (Combined)
    raw_weights = config['train'].get('class_weights', None)
    if raw_weights:
        logging.info(f"Applying Class Weights: {raw_weights}")
        target_weights = torch.tensor(raw_weights).float().to(device)
    else:
        target_weights = None
    
    ls_factor = config['train'].get('label_smoothing_factor', 0.1)
    ls_ratio = config['train'].get('combined_loss_ratio', 0.8)
    
    logging.info(f"Using CombinedLSLoss: Ratio={ls_ratio} (LabelSmoothing) / {1-ls_ratio:.1f} (Focal)")
    
    criterion = CombinedLSLoss(
        num_classes=config['data']['num_classes'],
        alpha=target_weights, 
        gamma=2.0, 
        smoothing=ls_factor,
        ratio=ls_ratio
    )

    # 7. 调度器
    scheduler_cfg = config['train'].get('scheduler', {})
    warmup_epochs = config['train'].get('warmup_epochs', 5)
    T_0 = scheduler_cfg.get('T_0', 10)
    T_mult = scheduler_cfg.get('T_mult', 2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6
    )

    # 8. 早停与断点恢复 (关键修复部分)
    best_model_path = os.path.join(model_dir, config['train']['best_model_name'])
    early_stopping = EarlyStopping(
        patience=config['train']['early_stopping_patience'], 
        path=best_model_path,
        verbose=True
    )

    ckpt_path = os.path.join(model_dir, "checkpoint_last.pth")
    start_epoch = 0
    
    # --- 修复后的断点检测逻辑 ---
    # 逻辑: 
    # 1. 如果 Config 显式指定 True/False，则遵从 Config
    # 2. 如果 Config 是 null (None)，则自动检测是否存在 checkpoint 文件
    
    resume_config = config['train'].get('resume', None)
    
    if resume_config is not None:
        resume_flag = resume_config
        logging.info(f"Resume flag set by config: {resume_flag}")
    else:
        # Config 未指定 (null)，启用自动检测
        resume_flag = os.path.exists(ckpt_path)
        if resume_flag:
            logging.info(f"[Auto-Resume] Config is null, but checkpoint found at {ckpt_path}. Will resume.")
        else:
            logging.info(f"[Auto-Resume] No checkpoint found at {ckpt_path}. Starting fresh.")

    if resume_flag:
        if os.path.exists(ckpt_path):
            logging.info(f"Restoring state from {ckpt_path}...")
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 尝试加载 scheduler
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except Exception as e:
                        logging.warning(f"Scheduler state mismatch ({e}), starting fresh scheduler.")
                
                # 尝试加载 early_stopping
                if 'early_stopping_state_dict' in checkpoint:
                    early_stopping.load_state(checkpoint['early_stopping_state_dict'])
                    
                start_epoch = checkpoint['epoch'] + 1
                logging.info(f"Resume successful. Restarting from Epoch {start_epoch + 1}")
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}. Starting fresh.")
                start_epoch = 0
        else:
            logging.warning(f"Resume requested but {ckpt_path} does not exist. Starting fresh.")
    else:
        logging.info("Starting a fresh training session (Resume=False or not found).")

    # 9. 训练主循环
    logging.info(f"Total Epochs: {config['train']['epochs']}")
    
    for epoch in range(start_epoch, config['train']['epochs']):
        start_t = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loaders, criterion, optimizer, device, epoch, config, 
            writer, scheduler, warmup_epochs=warmup_epochs
        )
        
        val_loss, val_acc = validate(model, val_loaders, criterion, device, epoch, writer)
        
        elapsed = time.time() - start_t
        curr_lr = optimizer.param_groups[0]['lr']
        phase_str = "WARMUP" if epoch < warmup_epochs else "COSINE"
        
        logging.info(f"Ep {epoch+1}/{config['train']['epochs']} [{phase_str}] ({elapsed:.1f}s) | "
                     f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                     f"LR: {curr_lr:.2e}")
        
        # 保存断点 (Save Checkpoint)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_state_dict': early_stopping.state_dict()
        }, ckpt_path)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Training stopped.")
            break

    if writer: writer.close()

    # 10. 最终评估
    if os.path.exists(best_model_path):
        logging.info("---------- Final Evaluation (Test Split) ----------")
        # 重新加载最佳模型
        try:
            model.load_state_dict(torch.load(best_model_path))
        except Exception as e:
            # 兼容有些保存是 dict 有些是 model 本身的情况
            ckpt = torch.load(best_model_path)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)

        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for _, loader in test_loaders.items():
                for batch in loader:
                    if len(batch) == 3:
                        img, sig, lbl = batch
                    else:
                        img, sig, lbl, _ = batch
                    
                    if img is None: continue
                    output = model(img.to(device), sig.to(device))
                    _, pred = torch.max(output, 1)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(lbl.cpu().numpy())
        
        final_acc = accuracy_score(all_labels, all_preds)
        logging.info(f"Final Test Accuracy: {final_acc:.4f}")
        
        target_names = config['data'].get('case_ids', [str(i) for i in range(config['data']['num_classes'])])
        print("\nDetailed Report:\n", classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        
        cm = confusion_matrix(all_labels, all_preds)
        cm_fig = plot_confusion_matrix(cm, labels=target_names, title=f'Final Matrix (CombLoss R={ls_ratio})')
        cm_fig.savefig(os.path.join(log_dir, 'final_confusion_matrix.png'))
        logging.info(f"Confusion matrix saved to {log_dir}")
    else:
        logging.warning("No best model found. Assessment skipped.")

if __name__ == '__main__':
    main()