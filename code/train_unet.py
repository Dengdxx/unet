"""U-Net 训练脚本

本脚本负责训练 U-Net 模型进行细胞分割。
包括参数解析、环境设置、模型初始化、训练循环、验证、测试以及模型保存。
支持 AMP 混合精度训练和 Tensor Cores 加速。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.amp as torch_amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from modules.datasets import build_cell_dataloaders
from modules.models import UNet
from modules.utils import DiceLoss, dice_coefficient, plot_training_history, visualize_predictions


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    定义并解析训练脚本所需的各种参数，包括数据路径、模型超参数、训练配置等。

    Returns:
        argparse.Namespace: 包含解析后参数的对象。
    """
    parser = argparse.ArgumentParser(description="细胞分割 U-Net 训练脚本")
    parser.add_argument("--data-root", type=str, default="../dataset", help="数据根目录")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="DIC-C2DH-HeLa",
        choices=["DIC-C2DH-HeLa", "PhC-C2DH-U373"],
        help="选择使用的数据集",
    )
    parser.add_argument(
        "--mask-source",
        type=str,
        default="ST",
        choices=["ST", "GT"],
        help="优先使用的掩码来源 (ST: 银标准, GT: 人工标注)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["01", "02"],
        help="希望参与训练的序列编号，例如 --sequences 01 02",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resize", type=int, default=512, help="输入与输出统一 resize 尺寸")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dice-weight", type=float, default=1.0, help="DiceLoss 的loss权重")
    parser.add_argument("--output-dir", type=str, default="../model")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dice-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-train-augment",
        action="store_true",
        help="若指定则关闭训练阶段的数据增强",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子确保可复现性。

    同时设置 Python、NumPy 和 PyTorch 的随机种子。

    Args:
        seed (int): 随机种子值，默认为 42。

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 注意: cudnn.deterministic=True会降低性能,因此不启用


def make_amp_context(device: torch.device):
    """创建 AMP (Automatic Mixed Precision) 上下文。

    如果设备支持 CUDA，则启用 FP16 混合精度训练，可加速训练并节省显存。
    否则返回一个空的上下文管理器。

    Args:
        device (torch.device): 运行设备。

    Returns:
        Callable: 返回上下文管理器的函数。
    """
    if device.type == "cuda":
        return lambda: torch_amp.autocast("cuda", dtype=torch.float16)
    return lambda: nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch_amp.GradScaler,
    criterion: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    dice_weight: float,
    threshold: float,
    amp_context,
) -> Tuple[float, float]:
    """执行单个 Epoch 的训练流程。

    关键优化点:
    - non_blocking=True: CPU->GPU异步传输,与计算并行
    - set_to_none=True: 梯度置None比置0更快且省显存
    - AMP自动混合精度: FP16前向+FP32梯度累积

    Args:
        model (nn.Module): 待训练的模型。
        loader (DataLoader): 训练数据加载器。
        optimizer (Optimizer): 优化器。
        scaler (GradScaler): 混合精度训练的梯度缩放器。
        criterion (nn.Module): 主损失函数（通常是 BCE）。
        dice_loss (DiceLoss): Dice 损失函数。
        device (torch.device): 运行设备。
        dice_weight (float): Dice 损失的权重。
        threshold (float): Dice 系数计算的阈值。
        amp_context (Callable): AMP 上下文管理器工厂函数。

    Returns:
        Tuple[float, float]: (平均损失, 平均 Dice 系数)。
    """
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    total = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            logits = model(images)
            bce_loss = criterion(logits, masks)
            d_loss = dice_loss(logits, masks)
            loss = bce_loss + dice_weight * d_loss  # 复合损失 = BCE + λ*Dice

        # GradScaler处理FP16下的梯度缩放,防止下溢
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_dice = dice_coefficient(logits, masks, threshold=threshold).item()

        epoch_loss += loss.item() * batch_size
        epoch_dice += batch_dice * batch_size
        total += batch_size
        progress.set_postfix({"loss": loss.item(), "dice": batch_dice})

    return epoch_loss / total, epoch_dice / total


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    dice_weight: float,
    threshold: float,
    amp_context,
) -> Tuple[float, float]:
    """在给定数据加载器上评估模型。

    Args:
        model (nn.Module): 待评估的模型。
        loader (DataLoader): 验证或测试数据加载器。
        criterion (nn.Module): 主损失函数。
        dice_loss (DiceLoss): Dice 损失函数。
        device (torch.device): 运行设备。
        dice_weight (float): Dice 损失的权重。
        threshold (float): Dice 系数计算的阈值。
        amp_context (Callable): AMP 上下文管理器工厂函数。

    Returns:
        Tuple[float, float]: (平均损失, 平均 Dice 系数)。
    """
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    total = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="Eval", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            batch_size = images.size(0)

            with amp_context():
                logits = model(images)
            bce_loss = criterion(logits, masks)
            d_loss = dice_loss(logits, masks)
            loss = bce_loss + dice_weight * d_loss
            batch_dice = dice_coefficient(logits, masks, threshold=threshold).item()

            epoch_loss += loss.item() * batch_size
            epoch_dice += batch_dice * batch_size
            total += batch_size
            progress.set_postfix({"loss": loss.item(), "dice": batch_dice})

    return epoch_loss / total, epoch_dice / total


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch_amp.GradScaler,
    history: Dict[str, list],
    epoch: int,
    path: Path,
) -> None:
    """保存完整训练状态以支持断点续训。

    保存模型权重、优化器状态、调度器状态、Scaler 状态、训练历史和当前 Epoch。

    Args:
        model (nn.Module): 模型。
        optimizer (Optimizer): 优化器。
        scheduler (_LRScheduler): 学习率调度器。
        scaler (GradScaler): 梯度缩放器。
        history (Dict[str, list]): 训练历史记录。
        epoch (int): 当前 Epoch。
        path (Path): 保存路径。

    Returns:
        None
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "epoch": epoch,
    }
    torch.save(state, path)


def main() -> None:
    """主函数。

    组织整个训练流程：
    1. 解析参数和设置环境。
    2. 准备数据加载器。
    3. 初始化模型、损失函数、优化器。
    4. 执行训练循环。
    5. 保存模型和日志。
    6. 执行测试和可视化。

    Returns:
        None
    """
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 性能压榨配置 - 充分利用RTX4070的Tensor Cores (PyTorch 2.9+ API)
    torch.backends.cudnn.benchmark = True  # 自动寻优卷积算法
    if device.type == "cuda":
        # TF32加速: Ampere架构(RTX 30/40系)的Tensor Cores专用
        # TF32提供接近FP16的速度,同时保持FP32的数值范围
        torch.backends.cudnn.conv.fp32_precision = 'tf32'      # 卷积层使用TF32
        torch.backends.cuda.matmul.fp32_precision = 'tf32'     # 矩阵乘法使用TF32
    
    amp_context = make_amp_context(device)

    resize = (args.resize, args.resize)
    train_loader, val_loader, test_loader = build_cell_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        resize_to=resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        sequences=tuple(args.sequences),
        mask_source=args.mask_source,
        train_augment=not args.disable_train_augment,
    )

    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch_amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    run_name = args.run_name or f"{args.dataset_name}_{int(time.time())}"
    output_dir = Path(args.output_dir) / run_name
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 详细的训练历史记录 (参考U-Net原论文的评估指标)
    history = {
        "train_loss": [], "val_loss": [], "test_loss": None,
        "train_dice": [], "val_dice": [], "test_dice": None,
        "learning_rates": [],  # 学习率变化
        "epoch_times": [],     # 每个epoch耗时
        "best_epoch": None,    # 最佳epoch
        "best_val_loss": float("inf"),
        "hyperparameters": {   # 超参数记录
            "model": "UNet",
            "base_channels": 32,
            "depth": 4,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": f"BCEWithLogitsLoss + {args.dice_weight}*DiceLoss",
            "resize": args.resize,
            "epochs": args.epochs,
            "dataset": args.dataset_name,
            "sequences": args.sequences,
            "mask_source": args.mask_source,
            "augmentation": not args.disable_train_augment,
            "device": str(device),
            "mixed_precision": device.type == "cuda",
            "seed": args.seed,
        }
    }
    best_val = float("inf")
    best_path = output_dir / "best.ckpt"

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            dice_loss,
            device,
            args.dice_weight,
            args.dice_threshold,
            amp_context,
        )
        val_loss, val_dice = evaluate(
            model,
            val_loader,
            criterion,
            dice_loss,
            device,
            args.dice_weight,
            args.dice_threshold,
            amp_context,
        )
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step()

        # 记录详细指标
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Train Dice {train_dice:.4f} | Val Dice {val_dice:.4f} | LR {current_lr:.6f} | Time {epoch_time:.2f}s"
        )

        # 保存最佳模型
        if val_loss < best_val:
            best_val = val_loss
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val
            save_checkpoint(model, optimizer, scheduler, scaler, history, epoch, best_path)
            print(f"  → 最佳模型已更新 (Epoch {epoch}, Val Loss: {best_val:.4f})")
        
        # 只在最后一个epoch保存
        if epoch == args.epochs:
            last_checkpoint = output_dir / "last.ckpt"
            save_checkpoint(model, optimizer, scheduler, scaler, history, epoch, last_checkpoint)
            print(f"  → 最后epoch模型已保存")

    # 加载最佳模型进行测试集评估
    print(f"\n{'='*60}")
    print(f"加载最佳模型 (Epoch {history['best_epoch']}) 进行测试集评估...")
    print(f"{'='*60}\n")
    
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test_loss, test_dice = evaluate(
        model,
        test_loader,
        criterion,
        dice_loss,
        device,
        args.dice_weight,
        args.dice_threshold,
        amp_context,
    )
    
    # 记录测试集指标
    history["test_loss"] = test_loss
    history["test_dice"] = test_dice
    
    print(f"\n{'='*60}")
    print(f"测试集评估 | Loss: {test_loss:.4f} | Dice: {test_dice:.4f}")
    print(f"训练总耗时: {sum(history['epoch_times']):.1f}秒")
    print(f"平均epoch耗时: {sum(history['epoch_times'])/len(history['epoch_times']):.2f}秒")
    print(f"{'='*60}\n")
    
    # 保存完整训练历史
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # 绘制训练曲线(包含过拟合监控)
    plot_training_history(history, figures_dir)

    # 批量可视化推理结果
    print("生成推理可视化...")
    visualize_predictions(
        model,
        val_loader.dataset,
        device,
        figures_dir,
        threshold=args.dice_threshold,
        num_samples=8,
    )
    print(f"\n所有结果已保存至: {output_dir}")
    print(f"✓ 训练完成!")
    
    # 清理资源,确保进程能正常退出
    del model
    del train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
