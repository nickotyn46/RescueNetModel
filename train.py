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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from data.dataset import RescueNetDataset, LABEL_MAP_11_TO_5
from models.unet import AttU_Net
from transforms import get_train_transform, get_val_transform
from utils.metrics import (
    AverageMeter, poly_learning_rate,
    intersectionAndUnionGPU, compute_iou_per_class, print_iou_table
)


def soft_dice_loss(logits, target, num_classes, ignore_index=255, important_class_ids=None):
    """Soft Dice (per class). important_class_ids: sadece bu sınıfların Dice'ı (bina+yol)."""
    probs = F.softmax(logits, dim=1)
    target_flat = target.long().clamp(0, num_classes - 1)
    one_hot = F.one_hot(target_flat, num_classes).permute(0, 3, 1, 2).float()
    valid = (target != ignore_index) & (target < num_classes)
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    card = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-8) / (card + 1e-8)
    if important_class_ids is not None:
        dice = dice[torch.tensor(important_class_ids, device=dice.device)]
    return 1.0 - dice.mean()


class CriterionWithDice(nn.Module):
    """CE + optional Dice. loss_only_on_important=True → gradient sadece bina+yol piksellerinde."""
    def __init__(self, ce_weight=None, ignore_index=255, dice_weight=0.0, num_classes=5,
                 important_class_ids=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.important_class_ids = important_class_ids  # [1,2,3,4] = sadece bina+yol

    def forward(self, logits, target):
        if self.important_class_ids is not None:
            # CE sadece bina+yol piksellerinde (Others'a gradient yok)
            ce_per_pixel = F.cross_entropy(
                logits, target, reduction='none',
                weight=self.ce.weight, ignore_index=self.ignore_index
            )
            important_mask = torch.zeros_like(target, dtype=torch.bool)
            for c in self.important_class_ids:
                important_mask = important_mask | (target == c)
            n = important_mask.sum().float().clamp(min=1.0)
            loss_ce = (ce_per_pixel * important_mask.float()).sum() / n
        else:
            loss_ce = self.ce(logits, target)

        if self.dice_weight <= 0:
            return loss_ce
        loss_dice = soft_dice_loss(
            logits, target, self.num_classes, self.ignore_index,
            important_class_ids=self.important_class_ids
        )
        return loss_ce + self.dice_weight * loss_dice


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
    # Handle both plain model and DataParallel-wrapped model
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.load_state_dict(ckpt['state_dict'])
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
        with autocast('cuda', enabled=use_amp):
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

    important_class_ids = cfg['DATA'].get('important_class_ids')
    if important_class_ids is not None:
        important_miou = float(np.mean(iou_cls[np.array(important_class_ids)]))
        print(f'  → Important mIoU (classes {important_class_ids}): {important_miou:.4f}')
        if writer:
            writer.add_scalar('mIoU_important/val', important_miou, epoch)

    if writer:
        writer.add_scalar('Loss/val', loss_meter.avg, epoch)
        writer.add_scalar('mIoU/val', miou, epoch)

    return loss_meter.avg, miou, iou_cls


# ─── Main ─────────────────────────────────────────────────────────────────────

class PSPNetSMPWrapper(nn.Module):
    """
    Wrap segmentation_models_pytorch.PSPNet to allow 713×713 inputs by
    padding up to the next multiple of 8 and then cropping back.
    """
    def __init__(self, encoder_name: str, num_classes: int):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch is required for PSPNet. "
                "Install it with `pip install segmentation-models-pytorch`."
            ) from e

        self.model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        new_h = (h + 7) // 8 * 8
        new_w = (w + 7) // 8 * 8

        pad_bottom = new_h - h
        pad_right = new_w - w

        if pad_bottom > 0 or pad_right > 0:
            x = F.pad(x, (0, pad_right, 0, pad_bottom))

        y = self.model(x)

        if pad_bottom > 0 or pad_right > 0:
            y = y[..., :h, :w]
        return y


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    train_cfg = cfg['TRAIN']
    data_cfg  = cfg['DATA']
    arch      = train_cfg.get('arch', 'aunet')

    # Dataset (optional 5-class: building light/heavy, road clear/blocked, others)
    label_mapping = LABEL_MAP_11_TO_5 if data_cfg.get('use_reduced_5class', False) else None
    train_ds = RescueNetDataset(
        root_dir=data_cfg['data_root'],
        mode='train',
        joint_transform=get_train_transform(train_cfg['train_h'], train_cfg['train_w']),
        label_mapping=label_mapping,
    )
    val_ds = RescueNetDataset(
        root_dir=data_cfg['data_root'],
        mode='val',
        joint_transform=get_val_transform(train_cfg['train_h'], train_cfg['train_w']),
        label_mapping=label_mapping,
    )

    print(f'Train: {len(train_ds)} samples | Val: {len(val_ds)} samples')

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
    if arch == 'aunet':
        model = AttU_Net(img_ch=3, output_ch=data_cfg['num_classes']).cuda()
        model_desc = 'Attention U-Net'
    elif arch in ('pspnet', 'pspnet_resnet101'):
        encoder_name = 'resnet101' if '101' in arch else 'resnet50'
        model = PSPNetSMPWrapper(
            encoder_name=encoder_name,
            num_classes=data_cfg['num_classes'],
        ).cuda()
        model_desc = f'PSPNet-{encoder_name}'
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: 'aunet', 'pspnet_resnet101'.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {model_desc} | Trainable params: {total_params/1e6:.1f}M')
    if torch.cuda.device_count() > 1:
        print(f'DataParallel: {torch.cuda.device_count()} GPU kullanılıyor')
        model = nn.DataParallel(model)

    # Loss: sadece bina+yol önemli → gradient da sadece orada (loss_only_on_important)
    weight = None
    if data_cfg.get('class_weights') is not None:
        weight = torch.tensor(data_cfg['class_weights'], dtype=torch.float32).cuda()
        assert weight.shape[0] == data_cfg['num_classes'], \
            f"class_weights length must be num_classes ({data_cfg['num_classes']})"
        print(f'Class weights: {data_cfg["class_weights"]}')
    use_dice = train_cfg.get('use_dice_ce', False)
    dice_weight = train_cfg.get('dice_weight', 0.5)
    loss_only_important = data_cfg.get('loss_only_on_important', False)
    important_ids = data_cfg.get('important_class_ids') if loss_only_important else None
    if use_dice or loss_only_important:
        criterion = CriterionWithDice(
            ce_weight=weight,
            ignore_index=train_cfg['ignore_label'],
            dice_weight=dice_weight if use_dice else 0.0,
            num_classes=data_cfg['num_classes'],
            important_class_ids=important_ids,
        )
        if loss_only_important:
            print('Loss: sadece bina+yol piksellerinde (Others yok)')
        if use_dice:
            print(f'Loss: CE + Dice (dice_weight={dice_weight})')
    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=train_cfg['ignore_label'],
            weight=weight,
        )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg['base_lr'],
        momentum=train_cfg['momentum'],
        weight_decay=train_cfg['weight_decay'],
    )

    use_amp = train_cfg.get('use_amp', False)
    scaler  = GradScaler('cuda') if use_amp else None
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
        _, _ = train_epoch(
            train_loader, model, criterion, optimizer, epoch, cfg, writer, scaler
        )

        _, val_miou, iou_cls = val_epoch(
            val_loader, model, criterion, epoch, cfg, writer
        )

        # Best model: by important mIoU (sadece bina+yol) veya full mIoU
        important_ids = data_cfg.get('important_class_ids')
        if important_ids is not None:
            best_metric = float(np.mean(iou_cls[np.array(important_ids)]))
            metric_name = 'important mIoU'
        else:
            best_metric = val_miou
            metric_name = 'mIoU'

        # Unwrap DataParallel for saving (ensures checkpoint is always loadable
        # regardless of whether DataParallel is used at resume time)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        # Save latest checkpoint every save_freq epochs
        if (epoch + 1) % train_cfg['save_freq'] == 0:
            save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': raw_model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_miou': best_miou},
                train_cfg['save_path'],
                'latest.pth'
            )

        # Save best checkpoint (by important mIoU when configured)
        if best_metric > best_miou:
            best_miou  = best_metric
            no_improve = 0
            save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': raw_model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_miou': best_miou},
                train_cfg['save_path'],
                'best.pth'
            )
            print(f'★ New best {metric_name}: {best_miou:.4f} — checkpoint saved.')
        else:
            no_improve += 1
            print(f'No improvement for {no_improve}/{patience} epochs.')

        if no_improve >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}.')
            break

    writer.close()
    metric_name = 'important mIoU (bina+yol)' if data_cfg.get('important_class_ids') else 'mIoU'
    print(f'\nTraining complete. Best val {metric_name}: {best_miou:.4f}')
    print(f'Best model saved at: {os.path.join(train_cfg["save_path"], "best.pth")}')


if __name__ == '__main__':
    main()
