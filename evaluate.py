"""
Evaluate a trained Attention U-Net checkpoint on the RescueNet test set.

Usage:
    python evaluate.py --config configs/rescuenet_aunet.yaml \
                       --model-path /kaggle/working/checkpoints/best.pth \
                       [--save-masks]
"""

import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import RescueNetDataset
from models.unet import AttU_Net
from transforms import get_val_transform
from utils.metrics import intersectionAndUnionGPU, compute_iou_per_class, print_iou_table


# Classes of interest for the Teknofest project
INTEREST_CLASSES = {
    2: 'Building-No-Damage',
    3: 'Building-Minor-Damage',
    4: 'Building-Major-Damage',
    5: 'Building-Total-Destruction',
    7: 'Road-Clear',
    8: 'Road-Blocked',
}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Attention U-Net on RescueNet test set')
    parser.add_argument('--config',     default='configs/rescuenet_aunet.yaml')
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

    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    color_map   = data_cfg['color_map']

    test_ds = RescueNetDataset(
        root_dir=data_cfg['data_root'],
        mode='test',
        joint_transform=get_val_transform(test_cfg['test_h'], test_cfg['test_w']),
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f'Test set: {len(test_ds)} samples')

    model = AttU_Net(img_ch=3, output_ch=num_classes).cuda()
    ckpt  = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}  (trained for {ckpt.get("epoch","?")} epochs)')

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
    print('\n── Classes of Interest (Teknofest) ──')
    for cls_id, cls_name in INTEREST_CLASSES.items():
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
