# tools/dataset.py
"""
多标签分割数据集加载器
支持.npy格式的多通道掩码
"""
import os
from typing import List, Tuple, Optional
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

__all__ = ['MultiLabelDataset', 'ResizeAndNormalize', 'IMAGENET_MEAN', 'IMAGENET_STD']

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeAndNormalize:
    """图像和掩码的预处理变换"""
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        # 处理图像
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)
        img_t = TF.normalize(img_t, self.mean, self.std)
        
        # 处理多通道掩码 - 注意：不使用 to_tensor，因为它会除以255
        H, W = self.size
        
        # 兼容单通道(二维)和多通道(三维)掩码
        if mask_hwc.ndim == 2:
            mask_hwc = np.expand_dims(mask_hwc, axis=-1)
            
        C = mask_hwc.shape[2]
        mask_resized = np.zeros((H, W, C), dtype=np.float32)
        
        for c in range(C):
            # 使用cv2.resize进行最近邻插值
            ch = mask_hwc[:, :, c].astype(np.float32)
            mask_resized[:, :, c] = cv2.resize(ch, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # 转换为 (C, H, W) 格式的 tensor
        mask_t = torch.from_numpy(mask_resized).permute(2, 0, 1)  # (C, H, W)
        # 二值化（确保是0或1）
        mask_t = (mask_t > 0.5).float()
        
        return img_t, mask_t


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    """列出目录下的所有图片文件"""
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out


class MultiLabelDataset(data.Dataset):
    """
    多标签分割数据集
    
    目录结构:
        root/
            train/
                image/
                mask/  (.npy格式，形状为 H,W,C)
            test/
                image/
                mask/
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "image",
        mask_dir_name: str = "mask",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: str = ".npy",
        transform: Optional[ResizeAndNormalize] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split, img_dir_name)
        self.mask_dir = os.path.join(root, split, mask_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # 列出所有图片
        self.img_paths = _list_files(self.img_dir, img_exts)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # 匹配图片和掩码
        self.pairs: List[Tuple[str, str]] = []
        for img_path in self.img_paths:
            img_name = os.path.basename(img_path)
            mask_name = os.path.splitext(img_name)[0] + mask_ext
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                print(f"[Warning] Mask not found for: {img_name}")

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, mask) pairs in {split} set!")
        
        print(f"[Dataset] {split}: {len(self.pairs)} samples loaded")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 读取多通道掩码
        mask_hwc = np.load(mask_path)  # (H, W, C) 或 (H, W)
        if mask_hwc is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
            
        if mask_hwc.ndim == 2:
            mask_hwc = np.expand_dims(mask_hwc, axis=-1)

        # 应用变换
        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_hwc).permute(2, 0, 1).float()
            mask_t = (mask_t > 0.5).float()

        img_name = os.path.basename(img_path)
        return img_t, mask_t, img_name
