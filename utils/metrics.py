import numpy as np
import torch


class AverageMeter:
    """Tracks a running average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def poly_learning_rate(base_lr, current_iter, max_iter, power=0.9):
    """Polynomial learning rate decay: lr = base_lr * (1 - iter/max_iter)^power."""
    return base_lr * ((1 - current_iter / max_iter) ** power)


def intersectionAndUnion(output, target, num_classes, ignore_index=255):
    """Compute per-class intersection and union on CPU (numpy arrays or tensors).

    Args:
        output: Predicted class indices, shape (H, W) or (N, H, W).
        target: Ground-truth class indices, same shape as output.
        num_classes: Total number of classes.
        ignore_index: Label to ignore (default 255).

    Returns:
        intersection: (num_classes,) array
        union:        (num_classes,) array
        target_area:  (num_classes,) array
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    output = output.flatten().astype(np.int64)
    target = target.flatten().astype(np.int64)

    valid = (target != ignore_index) & (target < num_classes)
    output = output[valid]
    target = target[valid]

    intersection = output[output == target]
    area_intersection = np.bincount(intersection, minlength=num_classes)
    area_output       = np.bincount(output,       minlength=num_classes)
    area_target       = np.bincount(target,       minlength=num_classes)
    area_union        = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, num_classes, ignore_index=255):
    """GPU version of intersectionAndUnion.

    Args:
        output: Predicted class indices tensor, shape (N, H, W) or (H, W).
        target: Ground-truth tensor, same shape.
        num_classes: Total number of classes.
        ignore_index: Label to ignore.

    Returns:
        intersection, union, target_area — all on the same device as input.
    """
    assert output.dim() in (2, 3)
    if output.dim() == 3:
        output = output.view(-1)
        target = target.view(-1)
    else:
        output = output.flatten()
        target = target.flatten()

    valid  = (target != ignore_index) & (target < num_classes)
    output = output[valid]
    target = target[valid]

    intersection = output[output == target]
    area_intersection = torch.bincount(intersection,       minlength=num_classes).float()
    area_output       = torch.bincount(output,             minlength=num_classes).float()
    area_target       = torch.bincount(target,             minlength=num_classes).float()
    area_union        = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def compute_iou_per_class(intersection_sum, union_sum, eps=1e-10):
    """Compute per-class IoU from accumulated intersection/union arrays."""
    iou = intersection_sum / (union_sum + eps)
    return iou


def print_iou_table(iou_per_class, class_names, title="IoU Results"):
    """Pretty-print per-class IoU results."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  {'Class':<30} {'IoU':>8}  {'%':>8}")
    print(f"  {'-'*46}")
    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        print(f"  {name:<30} {iou:>8.4f}  {iou*100:>7.2f}%")
    mean_iou = np.nanmean(iou_per_class)
    print(f"  {'-'*46}")
    print(f"  {'Mean IoU':<30} {mean_iou:>8.4f}  {mean_iou*100:>7.2f}%")
    print(f"{'='*55}\n")
    return mean_iou
