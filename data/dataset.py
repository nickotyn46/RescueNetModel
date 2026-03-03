"""
RescueNet Dataset
Classes: {Background:0, Water:1, Building_No_Damage:2, Building_Minor_Damage:3,
          Building_Major_Damage:4, Building_Total_Destruction:5, Vehicle:6,
          Road-Clear:7, Road-Blocked:8, Tree:9, Pool:10}

Optional 5-class reduced setup (label_mapping):
  Others:0, Building-Light:1, Building-Heavy:2, Road-Clear:3, Road-Blocked:4
  Mapping: 0,1,6,9,10→0; 2,3→1; 4,5→2; 7→3; 8→4; 255→255.

Optional 3-class building-only (label_mapping):
  Others:0, Building-Light:1, Building-Heavy:2
  Mapping: 0,1,6,7,8,9,10→0; 2,3→1; 4,5→2; 255→255.

Kaggle dataset folder structure (yaroslavchyrko/rescuenet):
  <root>/train/train-org-img/   -> .jpg images
  <root>/train/train-label-img/ -> .png masks (filename contains 'lab')
  <root>/val/val-org-img/
  <root>/val/val-label-img/
  <root>/test/test-org-img/
  <root>/test/test-label-img/
"""

# 11 → 5 class mapping: [new_id for old_id in 0..10]
# Others:0, Building-Light:1, Building-Heavy:2, Road-Clear:3, Road-Blocked:4
LABEL_MAP_11_TO_5 = [0, 0, 1, 1, 2, 2, 0, 3, 4, 0, 0]  # index = original class 0..10
REDUCED_NUM_CLASSES = 5
REDUCED_CLASS_NAMES = ['Others', 'Building-Light', 'Building-Heavy', 'Road-Clear', 'Road-Blocked']

# 11 → 3 class mapping (sadece bina: Others, Building-Light, Building-Heavy)
# Yol/su/araç/vb. hepsi Others
LABEL_MAP_11_TO_3 = [0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0]  # 0..10 → 0,1,2
REDUCED_3_CLASS_NAMES = ['Others', 'Building-Light', 'Building-Heavy']

import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data


def _get_files(folder, name_filter=None, extension_filter=None):
    if not os.path.isdir(folder):
        raise RuntimeError(f'"{folder}" is not a valid directory.')

    name_cond = (lambda f: name_filter in f) if name_filter else (lambda f: True)
    ext_cond = (lambda f: f.endswith(extension_filter)) if extension_filter else (lambda f: True)

    files = []
    for path, _, filenames in os.walk(folder):
        for fname in sorted(filenames):
            if name_cond(fname) and ext_cond(fname):
                files.append(os.path.join(path, fname))
    return files


class RescueNetDataset(data.Dataset):
    """RescueNet semantic segmentation dataset.

    Args:
        root_dir: Path to dataset root (contains train/, val/, test/).
        mode: One of 'train', 'val', 'test'.
        joint_transform: Applied to both image and mask (PIL Images).
        image_transform: Applied to image only after joint_transform.
        mask_transform: Applied to mask only after joint_transform.
    """

    FOLDER_MAP = {
        'train': ('train/train-org-img/', 'train/train-label-img/'),
        'val':   ('val/val-org-img/',     'val/val-label-img/'),
        'test':  ('test/test-org-img/',   'test/test-label-img/'),
    }

    NUM_CLASSES = 11

    def __init__(self, root_dir, mode='train',
                 joint_transform=None,
                 image_transform=None,
                 mask_transform=None,
                 label_mapping=None):
        """
        label_mapping: optional list of length 11; mapping[old_id] = new_id for original classes 0..10.
                       If provided, mask pixels are remapped (255 stays 255). Use LABEL_MAP_11_TO_5 for 5-class.
        """
        assert mode in self.FOLDER_MAP, f"mode must be one of {list(self.FOLDER_MAP)}"
        self.root_dir = root_dir
        self.mode = mode
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.label_mapping = label_mapping  # list of 11 elements, or None
        if label_mapping is not None:
            assert len(label_mapping) == 11, "label_mapping must have 11 elements (for original classes 0..10)"

        img_folder, lbl_folder = self.FOLDER_MAP[mode]
        self.images = _get_files(
            os.path.join(root_dir, img_folder),
            extension_filter='.jpg'
        )
        self.labels = _get_files(
            os.path.join(root_dir, lbl_folder),
            name_filter='lab',
            extension_filter='.png'
        )

        assert len(self.images) == len(self.labels), (
            f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels in {mode} set."
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.labels[index])

        if self.label_mapping is not None:
            arr = np.array(mask, dtype=np.int64)
            out = np.full_like(arr, 255)
            valid = (arr >= 0) & (arr <= 10)
            out[valid] = np.array(self.label_mapping, dtype=np.int64)[arr[valid]]
            mask = Image.fromarray(out.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.image_transform is not None:
            img = self.image_transform(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def get_image_path(self, index):
        return self.images[index]


def compute_class_weights(dataset, num_classes=11, c=1.02, max_samples=500):
    """Compute ENet-style class weights for CrossEntropyLoss.

    Uses a random subset of training images for speed.
    w_class = 1 / ln(c + p_class)  where p_class = freq_class / total_pixels
    """
    loader = data.DataLoader(dataset, batch_size=4, shuffle=True,
                             num_workers=2, drop_last=False)
    class_count = np.zeros(num_classes, dtype=np.int64)
    total = 0
    samples_seen = 0

    for _, masks in loader:
        if isinstance(masks, torch.Tensor):
            flat = masks.numpy().flatten()
        else:
            flat = np.array(masks).flatten()

        valid = flat[(flat != 255) & (flat < num_classes)]
        count = np.bincount(valid.astype(np.int64), minlength=num_classes)
        class_count += count
        total += valid.size
        samples_seen += masks.shape[0]
        if samples_seen >= max_samples:
            break

    propensity = class_count / (total + 1e-10)
    weights = 1.0 / (np.log(c + propensity) + 1e-10)
    return weights.astype(np.float32)
