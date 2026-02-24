"""
Joint transforms for image + mask pairs.
Random operations are applied identically to both so spatial alignment is preserved.
Color/intensity transforms are applied to the image only.
"""

import random
import numpy as np
import torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class Compose:
    """Apply a list of joint transforms sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Resize:
    """Resize both image and mask to (height, width)."""
    def __init__(self, height, width):
        self.size = (height, width)

    def __call__(self, img, mask):
        img  = TF.resize(img,  self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask


class RandomRotation:
    """Random rotation in [-max_angle, max_angle] degrees.
    Mask uses nearest-neighbor; fill with 255 (ignore label).
    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, img, mask):
        angle = random.uniform(-self.max_angle, self.max_angle)
        img  = TF.rotate(img,  angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST,  fill=255)
        return img, mask


class RandomScale:
    """Random scale crop: zoom in/out then resize back to original size."""
    def __init__(self, scale_min=0.5, scale_max=2.0, base_size=512):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.base_size = base_size

    def __call__(self, img, mask):
        scale = random.uniform(self.scale_min, self.scale_max)
        new_size = int(self.base_size * scale)
        img  = TF.resize(img,  (new_size, new_size), interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (new_size, new_size), interpolation=T.InterpolationMode.NEAREST)

        if new_size >= self.base_size:
            # Crop down to base_size
            i = random.randint(0, new_size - self.base_size)
            j = random.randint(0, new_size - self.base_size)
            img  = TF.crop(img,  i, j, self.base_size, self.base_size)
            mask = TF.crop(mask, i, j, self.base_size, self.base_size)
        else:
            # Pad up to base_size with reflect padding
            pad_h = self.base_size - new_size
            pad_w = self.base_size - new_size
            img  = TF.pad(img,  (0, 0, pad_w, pad_h), padding_mode='reflect')
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)
            img  = TF.resize(img,  (self.base_size, self.base_size), interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.base_size, self.base_size), interpolation=T.InterpolationMode.NEAREST)

        return img, mask


class ColorJitter:
    """Apply color jitter to image only (mask unchanged)."""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img, mask):
        img = self.jitter(img)
        return img, mask


class ToTensorAndNormalize:
    """Convert image to float tensor and normalize with ImageNet stats.
    Convert mask to LongTensor.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.MEAN, self.STD)

        mask_np = np.array(mask, dtype=np.int64)
        mask_t  = torch.from_numpy(mask_np)
        return img, mask_t


def get_train_transform(height=512, width=512):
    return Compose([
        RandomScale(scale_min=0.5, scale_max=2.0, base_size=height),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(max_angle=10),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ToTensorAndNormalize(),
    ])


def get_val_transform(height=512, width=512):
    return Compose([
        Resize(height, width),
        ToTensorAndNormalize(),
    ])
