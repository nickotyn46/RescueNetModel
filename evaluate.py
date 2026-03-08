"""
Evaluate a trained segmentation model (Attention U-Net or PSPNet) on
the RescueNet test set.

Usage:
    python evaluate.py --config configs/rescuenet_pspnet101.yaml \
                       --model-path /kaggle/working/checkpoints_pspnet/best.pth \
                       [--save-masks]
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import RescueNetDataset, LABEL_MAP_11_TO_5, LABEL_MAP_11_TO_3
from models.unet import AttU_Net
from transforms import get_val_transform, get_val_transform_crop
from utils.metrics import intersectionAndUnionGPU, compute_iou_per_class, print_iou_table


# Classes of interest (11-class setup)
INTEREST_CLASSES_11 = {
    2: 'Building-No-Damage',
    3: 'Building-Minor-Damage',
    4: 'Building-Major-Damage',
    5: 'Building-Total-Destruction',
    7: 'Road-Clear',
    8: 'Road-Blocked',
}
# 5-class: Building-Light, Building-Heavy, Road-Clear, Road-Blocked
INTEREST_CLASSES_5 = {1: 'Building-Light', 2: 'Building-Heavy', 3: 'Road-Clear', 4: 'Road-Blocked'}
# 3-class: sadece bina
INTEREST_CLASSES_3 = {1: 'Building-Light', 2: 'Building-Heavy'}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model on RescueNet test set')
    parser.add_argument('--config',     default='configs/rescuenet_pspnet101.yaml')
    parser.add_argument('--model-path', required=True, help='Path to best.pth checkpoint')
    parser.add_argument('--save-masks', action='store_true',
                        help='Save color-coded prediction masks to disk')
    return parser.parse_args()


def colorize_mask(mask_np, color_map):
    """Convert a 2D class-index mask to a 3-channel RGB image."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(color_map):
        rgb[mask_np == cls_id] = color
    return rgb


@torch.no_grad()
def evaluate(args, cfg):
    test_cfg  = cfg['TEST']
    train_cfg = cfg['TRAIN']
    data_cfg  = cfg['DATA']
    arch      = train_cfg.get('arch', 'aunet')

    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    color_map   = data_cfg['color_map']

    if data_cfg.get('use_reduced_3class', False):
        label_mapping = LABEL_MAP_11_TO_3
    elif data_cfg.get('use_reduced_5class', False):
        label_mapping = LABEL_MAP_11_TO_5
    else:
        label_mapping = None
    crop_size = test_cfg.get('test_crop_size') or train_cfg.get('train_crop_size')
    if crop_size is not None:
        joint_transform = get_val_transform_crop(crop_size)
        print(f'Test transform: center-crop/pad to {crop_size}x{crop_size}')
    else:
        joint_transform = get_val_transform(test_cfg['test_h'], test_cfg['test_w'])
    test_ds = RescueNetDataset(
        root_dir=data_cfg['data_root'],
        mode='test',
        joint_transform=joint_transform,
        label_mapping=label_mapping,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f'Test set: {len(test_ds)} samples')

    if arch == 'aunet':
        model = AttU_Net(img_ch=3, output_ch=num_classes).cuda()
        model_desc = 'Attention U-Net'
    elif arch in ('pspnet', 'pspnet_resnet101'):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch is required for PSPNet evaluation. "
                "Install it with `pip install segmentation-models-pytorch`."
            ) from e

        encoder_name = 'resnet101' if '101' in arch else 'resnet50'

        class PSPNetSMPWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = smp.PSPNet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=3,
                    classes=num_classes,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, h_in, w_in = x.shape
                new_h = (h_in + 7) // 8 * 8
                new_w = (w_in + 7) // 8 * 8
                pad_bottom = new_h - h_in
                pad_right = new_w - w_in
                if pad_bottom > 0 or pad_right > 0:
                    x = F.pad(x, (0, pad_right, 0, pad_bottom))
                y = self.model(x)
                if pad_bottom > 0 or pad_right > 0:
                    y = y[..., :h_in, :w_in]
                return y

        model = PSPNetSMPWrapper().cuda()
        model_desc = f'PSPNet-{encoder_name}'
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: 'aunet', 'pspnet_resnet101'.")
    ckpt  = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded {model_desc} checkpoint: {args.model_path}  (trained for {ckpt.get("epoch","?")} epochs)')

    save_dir = test_cfg.get('save_folder', '/kaggle/working/predictions')
    if args.save_masks:
        os.makedirs(save_dir, exist_ok=True)

    inter_sum  = np.zeros(num_classes, dtype=np.float64)
    union_sum  = np.zeros(num_classes, dtype=np.float64)

    for i, (images, masks) in enumerate(test_loader):
        images = images.cuda(non_blocking=True).float()
        masks  = masks.cuda(non_blocking=True).long()

        outputs = model(images)
        preds   = outputs.max(1)[1]

        inter, union, _ = intersectionAndUnionGPU(preds, masks, num_classes)
        inter_sum += inter.cpu().numpy()
        union_sum += union.cpu().numpy()

        if args.save_masks:
            pred_np  = preds[0].cpu().numpy()
            color_img = colorize_mask(pred_np, color_map)
            img_name  = os.path.splitext(
                os.path.basename(test_ds.get_image_path(i))
            )[0]
            Image.fromarray(color_img).save(
                os.path.join(save_dir, f'{img_name}_pred.png')
            )

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(test_loader)}] processed...')

    iou_per_class = compute_iou_per_class(inter_sum, union_sum)
    mean_iou      = print_iou_table(iou_per_class, class_names, title='Test Set Results')

    # Highlight the classes that matter for Teknofest
    if num_classes == 3:
        interest = INTEREST_CLASSES_3
    elif num_classes == 5:
        interest = INTEREST_CLASSES_5
    else:
        interest = INTEREST_CLASSES_11
    print('\n── Classes of Interest (Teknofest) ──')
    for cls_id, cls_name in interest.items():
        if cls_id < len(iou_per_class):
            iou = iou_per_class[cls_id]
            print(f'  {cls_name:<35} IoU: {iou:.4f}  ({iou*100:.2f}%)')

    return mean_iou, iou_per_class


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    mean_iou, _ = evaluate(args, cfg)
    print(f'\nFinal Mean IoU: {mean_iou*100:.2f}%')


if __name__ == '__main__':
    main()
