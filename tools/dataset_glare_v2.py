# tools/dataset_glare_v2.py
"""
带反光标签的多标签分割数据集 V2

支持两种数据集格式：
1. 多通道 mask 格式: train/mask/*.npy (shape: [H, W, C])
2. 独立 class 目录格式: train/label/class_0/*.npy

反光标签从单独的数据集目录加载
"""
import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['MultiLabelGlareDatasetV2', 'ResizeAndNormalize']


class ResizeAndNormalize:
    """调整大小并归一化"""
    def __init__(self, size=(512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)
    
    def __call__(self, image, masks, glare_mask=None):
        h, w = self.size
        
        # Resize image
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        image = self.normalize(image)
        
        # Resize masks
        resized_masks = []
        for i in range(masks.shape[0]):
            m = cv2.resize(masks[i].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(m)
        masks = np.stack(resized_masks, axis=0)
        masks = torch.from_numpy(masks).float()
        
        # Resize glare mask
        if glare_mask is not None:
            glare_mask = cv2.resize(glare_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            glare_mask = torch.from_numpy(glare_mask).float().unsqueeze(0)
        
        return image, masks, glare_mask


class MultiLabelGlareDatasetV2(Dataset):
    """
    带反光标签的多标签分割数据集 V2
    
    支持两种主数据集格式：
    1. 多通道 mask 格式:
       main_data_dir/train/image/*.jpg
       main_data_dir/train/mask/*.npy (shape: [H, W, num_classes])
    
    2. 独立 class 目录格式:
       main_data_dir/train/image/*.jpg
       main_data_dir/train/label/class_0/*.npy
       main_data_dir/train/label/class_1/*.npy
       ...
    
    反光数据集格式:
       glare_data_dir/train/mask/*.npy (shape: [H, W, 1] 或 [H, W])
    """
    def __init__(self, main_data_dir, glare_data_dir, split='train', transform=None, num_classes=3):
        self.main_data_dir = main_data_dir
        self.glare_data_dir = glare_data_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        
        # 主数据集路径
        self.img_dir = os.path.join(main_data_dir, split, 'image')
        self.mask_dir = os.path.join(main_data_dir, split, 'mask')  # 多通道格式
        self.label_base_dir = os.path.join(main_data_dir, split, 'label')  # 分离格式
        
        # 反光数据集路径
        self.glare_mask_dir = os.path.join(glare_data_dir, split, 'mask')
        
        # 检查目录
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        # 确定主标签格式
        self.use_multichannel_mask = os.path.exists(self.mask_dir)
        self.use_separate_labels = os.path.exists(self.label_base_dir)
        
        if self.use_multichannel_mask:
            print(f"[Dataset] Using multi-channel mask format: {self.mask_dir}")
        elif self.use_separate_labels:
            print(f"[Dataset] Using separate label directories: {self.label_base_dir}")
        else:
            raise FileNotFoundError(f"No mask or label directory found in {os.path.join(main_data_dir, split)}")
        
        # 获取图像列表
        self.img_names = sorted([f for f in os.listdir(self.img_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 如果使用多通道 mask，构建文件映射
        if self.use_multichannel_mask:
            self.mask_files = {os.path.splitext(f)[0]: f 
                              for f in os.listdir(self.mask_dir) 
                              if f.endswith('.npy')}
        
        # 检查反光数据集
        self.has_glare = os.path.exists(self.glare_mask_dir)
        if self.has_glare:
            self.glare_files = {os.path.splitext(f)[0]: f 
                               for f in os.listdir(self.glare_mask_dir) 
                               if f.endswith('.npy')}
            print(f"[Dataset] Found {len(self.glare_files)} glare masks")
        else:
            self.glare_files = {}
            print(f"[Dataset] Warning: No glare masks found at {self.glare_mask_dir}")
        
        print(f"[Dataset] Loaded {len(self.img_names)} images from {split} split")
    
    def __len__(self):
        return len(self.img_names)
    
    def _load_multichannel_mask(self, basename):
        """加载多通道 mask [H, W, C]"""
        if basename not in self.mask_files:
            return None
        
        path = os.path.join(self.mask_dir, self.mask_files[basename])
        mask = np.load(path)  # [H, W, C]
        
        # 确保格式正确
        if len(mask.shape) == 2:
            # 单通道，复制成多通道
            mask = np.stack([mask] * self.num_classes, axis=-1)
        elif mask.shape[-1] != self.num_classes:
            print(f"Warning: mask has {mask.shape[-1]} channels, expected {self.num_classes}")
        
        # 转换为 [C, H, W]
        mask = mask.transpose(2, 0, 1)  # [C, H, W]
        
        # 归一化到 [0, 1]
        if mask.max() > 1:
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        return mask
    
    def _load_separate_masks(self, basename, H, W):
        """加载分离的 class 目录格式标签"""
        masks = []
        for c in range(self.num_classes):
            mask = None
            for ext in ['.npy', '.png', '.jpg']:
                label_path = os.path.join(self.label_base_dir, f'class_{c}', basename + ext)
                if os.path.exists(label_path):
                    if ext == '.npy':
                        mask = np.load(label_path)
                    else:
                        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    break
            
            if mask is None:
                mask = np.zeros((H, W), dtype=np.float32)
            else:
                # 归一化到 [0, 1]
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.float32)
                else:
                    mask = mask.astype(np.float32)
            
            masks.append(mask)
        
        return np.stack(masks, axis=0)  # [C, H, W]
    
    def _load_glare_mask(self, basename, H, W):
        """加载反光标签"""
        if basename not in self.glare_files:
            return np.zeros((H, W), dtype=np.float32)
        
        path = os.path.join(self.glare_mask_dir, self.glare_files[basename])
        glare = np.load(path)
        
        # 处理多通道
        if len(glare.shape) > 2:
            glare = glare[..., 0] if glare.shape[-1] <= 3 else glare[:, :, 0]
        
        # 归一化
        if glare.max() > 1:
            glare = (glare > 0.5).astype(np.float32)
        else:
            glare = glare.astype(np.float32)
        
        return glare
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        basename = os.path.splitext(img_name)[0]
        
        # 加载图像
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        
        # 加载主标签
        if self.use_multichannel_mask:
            masks = self._load_multichannel_mask(basename)
            if masks is None:
                masks = np.zeros((self.num_classes, H, W), dtype=np.float32)
        else:
            masks = self._load_separate_masks(basename, H, W)
        
        # 加载反光标签
        glare_mask = self._load_glare_mask(basename, H, W)
        
        # 数据增强
        if self.transform is not None:
            image, masks, glare_mask = self.transform(image, masks, glare_mask)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            masks = torch.from_numpy(masks).float()
            glare_mask = torch.from_numpy(glare_mask).float().unsqueeze(0)
        
        return image, masks, glare_mask, basename
