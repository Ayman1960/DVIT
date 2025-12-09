import cv2
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import yaml
import tifffile as tiff
import numpy as np
import os
import matplotlib.pyplot as plt


def randomHueSaturationValue(
    image,
    hue_shift_limit=(-180, 180),
    sat_shift_limit=(-255, 255),
    val_shift_limit=(-255, 255),
    u=0.5,
):
    if np.random.random() < u:
        hsv_image = np.stack(
            [
                np.mod(image[:, :, 0] / 256 * 360, 360),  # Hue
                image[:, :, 1] / 255 * 100,  # Saturation
                image[:, :, 2] / 255 * 100,  # Value
            ],
            axis=-1,
        )

        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        sat_shift = (
            np.random.uniform(sat_shift_limit[0], sat_shift_limit[1]) / 255 * 100
        )
        val_shift = (
            np.random.uniform(val_shift_limit[0], val_shift_limit[1]) / 255 * 100
        )

        hsv_image[:, :, 0] = np.mod(hsv_image[:, :, 0] + hue_shift, 360)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + sat_shift, 0, 100)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + val_shift, 0, 100)

        image = np.stack(
            [
                (hsv_image[:, :, 0] / 360 * 256).astype(np.uint8),
                (hsv_image[:, :, 1] / 100 * 255).astype(np.uint8),
                (hsv_image[:, :, 2] / 100 * 255).astype(np.uint8),
            ],
            axis=-1,
        )

    return image


def randomShiftScaleRotate(
    image,
    mask,
    shift_limit=(-0.0, 0.0),
    scale_limit=(-0.0, 0.0),
    rotate_limit=(-0.0, 0.0),
    aspect_limit=(-0.0, 0.0),
    borderMode=cv2.BORDER_CONSTANT,
    u=0.5,
):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect**0.5)
        sy = scale / (aspect**0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ]
        )
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy]
        )

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(
                0,
                0,
                0,
            ),
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(
                0,
                0,
                0,
            ),
        )

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.flipud(image)
        mask = np.flipud(mask)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(id, root, shape, isEva=False):
    _img = tiff.imread(os.path.join(root, id))
    _mask = tiff.imread(os.path.join(root, id.replace("train", "label")))
    
    # 确保 _mask 是 3D 数组
    if _mask.ndim == 2:
        _mask = np.expand_dims(_mask, axis=2)
    elif _mask.ndim == 3 and _mask.shape[2] > 1:
        _mask = _mask[:,:,0:1]  # 只保留第一个通道
    
    # Resize images
    img = cv2.resize(_img, (shape, shape))
    mask = cv2.resize(_mask, (shape, shape))

    # Data augmentation (only if not in evaluation mode)
    if not isEva:
        img = randomHueSaturationValue(
            img,
            hue_shift_limit=(-30, 30),
            sat_shift_limit=(-5, 5),
            val_shift_limit=(-15, 15),
        )

        img, mask = randomShiftScaleRotate(
            img,
            mask,
            shift_limit=(-0.1, 0.1),
            scale_limit=(-0.1, 0.1),
            aspect_limit=(-0.1, 0.1),
            rotate_limit=(-0, 0),
        )
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    # Normalize and transpose the images
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

    # Ensure binary mask (0 or 1)
    mask[mask > 0] = 1

    return img, mask


class TiffFolder(data.Dataset):
    def __init__(self, root, size=256, isEva=False):
        self.ids = list(filter(lambda x: x.find("train") != -1, os.listdir(root)))
        self.loader = default_loader
        self.root = root
        self.isEva = isEva
        self.shape = size

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(
            id,
            self.root,
            self.shape,
            self.isEva,
        )
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)


def main():
    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(
            f,
        )

    root = config["data"]["root"]

    # Create dataset and dataloader
    dataset = TiffFolder(root)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Iterate through the dataloader and print some info
    for i, (img, mask) in enumerate(dataloader):
        print(f"Batch {i + 1}")
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        if i == 1:  # Only test two batches
            break

