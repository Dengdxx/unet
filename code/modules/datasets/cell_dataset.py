"""细胞分割数据集模块

本模块定义了用于加载和处理细胞分割数据集的类和函数。
支持 Cell Tracking Challenge 格式的数据，提供数据加载、预处理和增强功能。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class CellSample:
    """单个样本的元数据。

    Attributes:
        image_path (Path): 图像文件的路径。
        mask_path (Path): 掩码文件的路径。
        sequence (str): 序列编号（例如 "01"）。
        frame_id (str): 帧编号。
    """

    image_path: Path
    mask_path: Path
    sequence: str
    frame_id: str


class CellSegDataset(Dataset):
    """Cell Tracking Challenge 分割任务的数据集封装。

    负责加载图像和掩码，并进行统一的预处理和可选的数据增强。

    Attributes:
        root (Path): 数据集根目录。
        mask_source (str): 掩码来源类型 ("ST" 或 "GT")。
        resize_to (Tuple[int, int]): 图像调整的目标尺寸 (height, width)。
        augment (bool): 是否启用数据增强。
        mean (torch.Tensor): 归一化均值。
        std (torch.Tensor): 归一化标准差。
        threshold (int): 掩码二值化阈值。
        samples (List[CellSample]): 样本列表。
    """

    def __init__(
        self,
        root: Path | str,
        *,
        samples: Optional[Sequence[CellSample]] = None,
        sequences: Sequence[str] = ("01", "02"),
        mask_source: str = "ST",
        resize_to: Tuple[int, int] = (512, 512),
        augment: bool = False,
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.5,),
        threshold: int = 0,
    ) -> None:
        """初始化 CellSegDataset。

        Args:
            root (Path | str): 数据集根目录路径。
            samples (Optional[Sequence[CellSample]]): 预先扫描的样本列表。如果为 None，则自动扫描。
            sequences (Sequence[str]): 要包含的序列编号列表，默认为 ("01", "02")。
            mask_source (str): 优先使用的掩码来源，"ST" (Silver Truth) 或 "GT" (Ground Truth)。
            resize_to (Tuple[int, int]): 调整后的大小，默认为 (512, 512)。
            augment (bool): 是否启用训练时的数据增强，默认为 False。
            mean (Sequence[float]): 图像归一化的均值，默认为 (0.5,)。
            std (Sequence[float]): 图像归一化的标准差，默认为 (0.5,)。
            threshold (int): 读取掩码时的阈值，默认为 0。

        Raises:
            RuntimeError: 如果未找到任何有效样本。
        """
        self.root = Path(root)
        self.mask_source = mask_source
        self.resize_to = resize_to
        self.augment = augment
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.threshold = threshold

        if samples is None:
            self.samples = self._scan_samples(self.root, sequences, mask_source)
        else:
            self.samples = list(samples)

        if not self.samples:
            raise RuntimeError(
                f"未在 {self.root} 下找到有效的图像-掩码对，请检查数据目录或mask_source设置。"
            )

    @staticmethod
    def _scan_samples(root: Path, sequences: Sequence[str], mask_source: str) -> List[CellSample]:
        """扫描目录以发现有效的数据样本。

        Args:
            root (Path): 数据集根目录。
            sequences (Sequence[str]): 要扫描的序列。
            mask_source (str): 掩码来源 ("ST" 或 "GT")。

        Returns:
            List[CellSample]: 发现的样本列表。

        Raises:
            FileNotFoundError: 如果指定的掩码目录不存在且无法回退。
        """
        samples: List[CellSample] = []
        for seq in sequences:
            image_dir = root / seq
            if not image_dir.exists():
                continue

            mask_dir = root / f"{seq}_{mask_source}" / "SEG"
            if not mask_dir.exists():
                # 回退到GT（人工标注）
                fallback = root / f"{seq}_GT" / "SEG"
                if fallback.exists():
                    mask_dir = fallback
                else:
                    raise FileNotFoundError(
                        f"未找到 {seq} 序列的 {mask_source} 或 GT 掩码目录: {mask_dir}"
                    )

            for image_path in sorted(image_dir.glob("t*.tif")):
                frame_id = image_path.stem[1:]
                mask_name = f"man_seg{frame_id}.tif"
                mask_path = mask_dir / mask_name
                if not mask_path.exists():
                    # 对于GT只有部分帧有掩码，跳过缺失项
                    continue
                samples.append(
                    CellSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        sequence=seq,
                        frame_id=frame_id,
                    )
                )
        return samples

    def __len__(self) -> int:
        """返回数据集大小。

        Returns:
            int: 样本总数。
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | Dict[str, str]]:
        """获取指定索引的样本。

        加载图像和掩码，应用调整大小、增强和归一化。

        Args:
            index (int): 样本索引。

        Returns:
            Dict[str, Any]: 包含处理后数据的字典。
                - 'image': (C, H, W) 图像张量。
                - 'mask': (1, H, W) 掩码张量。
                - 'meta': 元数据字典（序列、帧ID、路径）。
        """
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("L")
        mask = Image.open(sample.mask_path).convert("L")

        image, mask = self._resize(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = (image_tensor - self.mean) / self.std

        mask_np = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_np > self.threshold).astype(np.float32)).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": {
                "sequence": sample.sequence,
                "frame_id": sample.frame_id,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            },
        }

    def _resize(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """调整图像和掩码的大小。

        Args:
            image (Image.Image): 输入图像。
            mask (Image.Image): 输入掩码。

        Returns:
            Tuple[Image.Image, Image.Image]: 调整大小后的图像和掩码。
        """
        image_resized = TF.resize(image, self.resize_to, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_resized = TF.resize(mask, self.resize_to, interpolation=InterpolationMode.NEAREST)
        return image_resized, mask_resized

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """应用数据增强。

        包括几何变换（翻转、旋转、仿射）和光度变换（Gamma、对比度、亮度）。
        几何变换同步应用于图像和掩码，光度变换仅应用于图像。

        Args:
            image (Image.Image): 输入图像。
            mask (Image.Image): 输入掩码。

        Returns:
            Tuple[Image.Image, Image.Image]: 增强后的图像和掩码。
        """
        
        # 几何变换: 翻转
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 几何变换: 旋转
        angle = random.uniform(-25.0, 25.0)
        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        # 几何变换: 仿射(平移+缩放+错切)
        if random.random() < 0.5:
            translate = (
                random.uniform(-0.05, 0.05) * image.size[0],
                random.uniform(-0.05, 0.05) * image.size[1],
            )
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-5.0, 5.0)
            image = TF.affine(
                image,
                angle=0.0,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.affine(
                mask,
                angle=0.0,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        # 光度变换: 只作用于图像
        if random.random() < 0.4:
            image = TF.adjust_gamma(image, gamma=random.uniform(0.8, 1.3))
        if random.random() < 0.4:
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))

        return image, mask


def _split_samples(
    samples: Sequence[CellSample],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[CellSample], List[CellSample], List[CellSample]]:
    """将样本划分为训练集、验证集和测试集。

    Args:
        samples (Sequence[CellSample]): 所有样本列表。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        seed (int): 随机种子，保证划分的可复现性。

    Returns:
        Tuple[List[CellSample], List[CellSample], List[CellSample]]:
            (训练集列表, 验证集列表, 测试集列表)。

    Raises:
        ValueError: 如果比例之和 >= 1.0 或训练集为空。
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio 必须小于1。")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    total = len(samples)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)
    n_train = total - n_val - n_test

    if n_train <= 0:
        raise ValueError("拆分比例导致训练样本为0，请调整val/test比例或增加数据量。")

    val_set = [samples[i] for i in indices[:n_val]]
    test_set = [samples[i] for i in indices[n_val : n_val + n_test]]
    train_set = [samples[i] for i in indices[n_val + n_test :]]

    # 确保至少有一个样本
    if not train_set:
        train_set = val_set[:1]
    if not val_set:
        val_set = train_set[:1]
    if not test_set:
        test_set = val_set[:1]

    return train_set, val_set, test_set


def build_cell_dataloaders(
    data_root: str | Path,
    dataset_name: str = "DIC-C2DH-HeLa",
    *,
    sequences: Sequence[str] = ("01", "02"),
    mask_source: str = "ST",
    resize_to: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    mean: Sequence[float] = (0.5,),
    std: Sequence[float] = (0.5,),
    pin_memory: bool = True,
    train_augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练、验证和测试的数据加载器。

    这是使用此模块的主要入口函数。它处理样本发现、划分和加载器创建。

    Args:
        data_root (str | Path): 数据集根目录。
        dataset_name (str): 数据集名称（子目录名）。
        sequences (Sequence[str]): 使用的序列列表。
        mask_source (str): 掩码来源 ("ST" 或 "GT")。
        resize_to (Tuple[int, int]): 图像调整尺寸。
        batch_size (int): 批大小。
        num_workers (int): DataLoader 的工作进程数。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        seed (int): 随机种子。
        mean (Sequence[float]): 归一化均值。
        std (Sequence[float]): 归一化标准差。
        pin_memory (bool): 是否使用 pinned memory（加速 CPU 到 GPU 传输）。
        train_augment (bool): 是否在训练集启用数据增强。

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (训练加载器, 验证加载器, 测试加载器)。
    """
    base_path = Path(data_root) / dataset_name
    samples = CellSegDataset._scan_samples(base_path, sequences, mask_source)
    train_samples, val_samples, test_samples = _split_samples(
        samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    def _build_dataset(sample_subset: Sequence[CellSample], augment: bool) -> CellSegDataset:
        return CellSegDataset(
            base_path,
            samples=sample_subset,
            sequences=sequences,
            mask_source=mask_source,
            resize_to=resize_to,
            augment=augment,
            mean=mean,
            std=std,
        )

    train_ds = _build_dataset(train_samples, augment=train_augment)
    val_ds = _build_dataset(val_samples, augment=False)
    test_ds = _build_dataset(test_samples, augment=False)

    def _make_loader(dataset: CellSegDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # 持久化worker进程,减少启动开销
            prefetch_factor=2 if num_workers > 0 else None,  # 预取batch数
        )

    return _make_loader(train_ds, shuffle=True), _make_loader(val_ds, shuffle=False), _make_loader(test_ds, shuffle=False)
