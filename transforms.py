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


class RandomRotation90:
    """Random rotation by 0, 90, 180, or 270 degrees. Applied to both image and mask."""
    def __call__(self, img, mask):
        angle = random.choice([0, 90, 180, 270])
        img  = TF.rotate(img,  angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST,  fill=255)
        return img, mask


class SmartRandomCrop:
    """Random crop of crop_size x crop_size. Prefer centers on building-heavy (class 2), then building-light (class 1).
    Expects mask already in 3-class (0=bg, 1=building-light, 2=building-heavy). Pads if image smaller than crop_size.
    """
    def __init__(self, crop_size, prob_heavy=0.5, prob_light=0.3):
        self.crop_size = crop_size
        self.prob_heavy = prob_heavy
        self.prob_light = prob_light

    def __call__(self, img, mask):
        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.int64)
        H, W = mask_np.shape[:2]

        if H < self.crop_size or W < self.crop_size:
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            img_np = np.pad(
                img_np, ((0, pad_h), (0, pad_w), (0, 0)),
                mode='reflect'
            )
            mask_np = np.pad(
                mask_np, ((0, pad_h), (0, pad_w)),
                constant_values=255
            )
            H, W = mask_np.shape[0], mask_np.shape[1]

        heavy_ys, heavy_xs = np.where(mask_np == 2)
        light_ys, light_xs = np.where(mask_np == 1)
        r = random.random()
        if r < self.prob_heavy and len(heavy_ys) > 0:
            idx = random.randint(0, len(heavy_ys) - 1)
            center_y, center_x = int(heavy_ys[idx]), int(heavy_xs[idx])
        elif r < self.prob_heavy + self.prob_light and len(light_ys) > 0:
            idx = random.randint(0, len(light_ys) - 1)
            center_y, center_x = int(light_ys[idx]), int(light_xs[idx])
        else:
            center_y = random.randint(0, H - 1) if H > 0 else 0
            center_x = random.randint(0, W - 1) if W > 0 else 0

        top = center_y - self.crop_size // 2
        left = center_x - self.crop_size // 2
        top = max(0, min(top, H - self.crop_size))
        left = max(0, min(left, W - self.crop_size))
        img_crop = img_np[top:top + self.crop_size, left:left + self.crop_size]
        mask_crop = mask_np[top:top + self.crop_size, left:left + self.crop_size]
        return Image.fromarray(img_crop), Image.fromarray(mask_crop.astype(np.uint8))


class CenterCropOrResize:
    """For validation: center crop of crop_size, or pad then center crop if image smaller than crop_size."""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.int64)
        H, W = mask_np.shape[0], mask_np.shape[1]

        if H < self.crop_size or W < self.crop_size:
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            img_np = np.pad(
                img_np, ((0, pad_h), (0, pad_w), (0, 0)),
                mode='reflect'
            )
            mask_np = np.pad(
                mask_np, ((0, pad_h), (0, pad_w)),
                constant_values=255
            )
            H, W = mask_np.shape[0], mask_np.shape[1]

        top = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        top = max(0, min(top, H - self.crop_size))
        left = max(0, min(left, W - self.crop_size))
        img_crop = img_np[top:top + self.crop_size, left:left + self.crop_size]
        mask_crop = mask_np[top:top + self.crop_size, left:left + self.crop_size]
        return Image.fromarray(img_crop), Image.fromarray(mask_crop.astype(np.uint8))


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
        # Replace any value outside valid class range (0-10) that isn't
        # already the ignore label (255) with 255 so CrossEntropyLoss skips it.
        mask_np[(mask_np > 10) & (mask_np != 255)] = 255
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


def get_train_transform_crop(crop_size, prob_heavy=0.5, prob_light=0.3):
    """Crop-based train transform: smart crop then light aug only (H flip, V flip, 90° rot, mild color)."""
    return Compose([
        SmartRandomCrop(crop_size, prob_heavy=prob_heavy, prob_light=prob_light),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        ToTensorAndNormalize(),
    ])


def get_val_transform_crop(crop_size):
    """Crop-based val transform: center crop (pad if smaller) then ToTensor."""
    return Compose([
        CenterCropOrResize(crop_size),
        ToTensorAndNormalize(),
    ])
