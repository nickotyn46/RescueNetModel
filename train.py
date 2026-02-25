"""
Training script for Attention U-Net on RescueNet dataset.
Designed to run on Kaggle GPU (T4/P100).

Usage:
    python train.py --config configs/rescuenet_aunet.yaml
    python train.py --config configs/rescuenet_aunet.yaml --resume /kaggle/working/checkpoints/latest.pth
"""

import os
import argparse
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from data.dataset import RescueNetDataset, compute_class_weights
from models.unet import AttU_Net
from transforms import get_train_transform, get_val_transform
from utils.metrics import (
    AverageMeter, poly_learning_rate,
    intersectionAndUnionGPU, compute_iou_per_class, print_iou_table
)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Attention U-Net on RescueNet')
    parser.add_argument('--config', default='configs/rescuenet_aunet.yaml')
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume from')
    parser.add_argument('--no-val', action='store_true', help='Skip validation during training')
    return parser.parse_args()


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(state, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None):
    print(f'=> Loading checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt.get('epoch', 0)
    best_miou   = ckpt.get('best_miou', 0.0)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f'   Resumed from epoch {start_epoch}, best mIoU = {best_miou:.4f}')
    return start_epoch, best_miou


# ─── Training epoch ───────────────────────────────────────────────────────────

def train_epoch(loader, model, criterion, optimizer, epoch, cfg, writer=None, scaler=None):
    num_classes = cfg['DATA']['num_classes']
    epochs      = cfg['TRAIN']['epochs']
    base_lr     = cfg['TRAIN']['base_lr']
    power       = cfg['TRAIN']['power']
    print_freq  = cfg['TRAIN']['print_freq']
    use_amp     = cfg['TRAIN'].get('use_amp', False)
    max_iter    = epochs * len(loader)

    loss_meter   = AverageMeter()
    inter_meter  = AverageMeter()
    union_meter  = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, masks) in enumerate(loader):
        images = images.cuda(non_blocking=True).float()
        masks  = masks.cuda(non_blocking=True).long()

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update poly LR
        current_iter = epoch * len(loader) + i + 1
        lr = poly_learning_rate(base_lr, current_iter, max_iter, power)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Metrics
        preds = outputs.detach().max(1)[1]
        inter, union, target = intersectionAndUnionGPU(preds, masks, num_classes)
        inter  = inter.cpu().numpy()
        union  = union.cpu().numpy()
        target = target.cpu().numpy()

        inter_meter.update(inter)
        union_meter.update(union)
        target_meter.update(target)
        loss_meter.update(loss.item(), images.size(0))

        if (i + 1) % print_freq == 0:
            iou_cls  = inter_meter.sum / (union_meter.sum + 1e-10)
            miou     = np.mean(iou_cls)
            elapsed  = time.time() - end
            print(
                f'Train Epoch [{epoch+1}/{epochs}] '
                f'Iter [{i+1}/{len(loader)}] '
                f'Loss: {loss_meter.avg:.4f}  '
                f'mIoU: {miou:.4f}  '
                f'LR: {lr:.6f}  '
                f'Time: {elapsed:.1f}s'
            )
            end = time.time()

    iou_cls = inter_meter.sum / (union_meter.sum + 1e-10)
    miou    = float(np.mean(iou_cls))
    if writer:
        writer.add_scalar('Loss/train', loss_meter.avg, epoch)
        writer.add_scalar('mIoU/train', miou, epoch)
        writer.add_scalar('LR', lr, epoch)
    print(f'Train Epoch [{epoch+1}/{epochs}] Done — Loss: {loss_meter.avg:.4f}  mIoU: {miou:.4f}')
    return loss_meter.avg, miou


# ─── Validation epoch ─────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(loader, model, criterion, epoch, cfg, writer=None):
    num_classes = cfg['DATA']['num_classes']
    class_names = cfg['DATA']['class_names']
    epochs      = cfg['TRAIN']['epochs']

    loss_meter   = AverageMeter()
    inter_meter  = AverageMeter()
    union_meter  = AverageMeter()

    model.eval()
    for images, masks in loader:
        images = images.cuda(non_blocking=True).float()
        masks  = masks.cuda(non_blocking=True).long()

        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss_meter.update(loss.item(), images.size(0))

        preds = outputs.max(1)[1]
        inter, union, _ = intersectionAndUnionGPU(preds, masks, num_classes)
        inter_meter.update(inter.cpu().numpy())
        union_meter.update(union.cpu().numpy())

    iou_cls = inter_meter.sum / (union_meter.sum + 1e-10)
    miou    = float(np.mean(iou_cls))

    print_iou_table(iou_cls, class_names, title=f'Val Epoch {epoch+1}/{epochs}')

    if writer:
        writer.add_scalar('Loss/val', loss_meter.avg, epoch)
        writer.add_scalar('mIoU/val', miou, epoch)

    return loss_meter.avg, miou


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    train_cfg = cfg['TRAIN']
    data_cfg  = cfg['DATA']

    # Dataset
    train_ds = RescueNetDataset(
        root_dir=data_cfg['data_root'],
        mode='train',
        joint_transform=get_train_transform(train_cfg['train_h'], train_cfg['train_w']),
    )
    val_ds = RescueNetDataset(
        root_dir=data_cfg['data_root'],
        mode='val',
        joint_transform=get_val_transform(train_cfg['train_h'], train_cfg['train_w']),
    )

    print(f'Train: {len(train_ds)} samples | Val: {len(val_ds)} samples')

    # Class weights to handle imbalance
    print('Computing class weights (this may take a moment)...')
    raw_weights = compute_class_weights(train_ds, num_classes=data_cfg['num_classes'])
    class_weights = torch.from_numpy(raw_weights).float().cuda()
    print('Class weights:', np.round(raw_weights, 3))

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['batch_size_val'],
        shuffle=False,
        num_workers=train_cfg['workers'],
        pin_memory=True,
    )

    # Model
    model = AttU_Net(img_ch=3, output_ch=data_cfg['num_classes']).cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: Attention U-Net | Trainable params: {total_params/1e6:.1f}M')

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=train_cfg['ignore_label']
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg['base_lr'],
        momentum=train_cfg['momentum'],
        weight_decay=train_cfg['weight_decay'],
    )

    use_amp = train_cfg.get('use_amp', False)
    scaler  = GradScaler() if use_amp else None
    if use_amp:
        print('Mixed precision (AMP) enabled.')

    # Resume
    start_epoch = train_cfg['start_epoch']
    best_miou   = 0.0
    resume_path = args.resume or train_cfg.get('resume', '')
    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_miou = load_checkpoint(resume_path, model, optimizer)

    writer    = SummaryWriter(log_dir=os.path.join(train_cfg['save_path'], 'logs'))
    patience  = train_cfg['early_stopping_patience']
    no_improve = 0

    print(f'\nStarting training — epochs {start_epoch} → {train_cfg["epochs"]}')
    print(f'Save path: {train_cfg["save_path"]}\n')

    for epoch in range(start_epoch, train_cfg['epochs']):
        train_loss, train_miou = train_epoch(
            train_loader, model, criterion, optimizer, epoch, cfg, writer, scaler
        )

        val_loss, val_miou = val_epoch(
            val_loader, model, criterion, epoch, cfg, writer
        )

        # Save latest checkpoint every save_freq epochs
        if (epoch + 1) % train_cfg['save_freq'] == 0:
            save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_miou': best_miou},
                train_cfg['save_path'],
                'latest.pth'
            )

        # Save best checkpoint
        if val_miou > best_miou:
            best_miou  = val_miou
            no_improve = 0
            save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_miou': best_miou},
                train_cfg['save_path'],
                'best.pth'
            )
            print(f'★ New best mIoU: {best_miou:.4f} — checkpoint saved.')
        else:
            no_improve += 1
            print(f'No improvement for {no_improve}/{patience} epochs.')

        if no_improve >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}.')
            break

    writer.close()
    print(f'\nTraining complete. Best val mIoU: {best_miou:.4f}')
    print(f'Best model saved at: {os.path.join(train_cfg["save_path"], "best.pth")}')


if __name__ == '__main__':
    main()
