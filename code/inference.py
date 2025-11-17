"""独立推理脚本 - 加载训练好的模型进行批量预测"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from modules.datasets import build_cell_dataloaders
from modules.models import UNet
from modules.utils import visualize_predictions, dice_coefficient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='U-Net推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径(.ckpt)')
    parser.add_argument('--data-root', type=str, default='../dataset')
    parser.add_argument('--dataset-name', type=str, default='DIC-C2DH-HeLa')
    parser.add_argument('--sequences', nargs='+', default=['01', '02'])
    parser.add_argument('--mask-source', type=str, default='ST')
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--output-dir', type=str, default='../inference_results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num-vis', type=int, default=8, help='可视化样本数')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建数据集
    resize = (args.resize, args.resize)
    train_loader, val_loader, test_loader = build_cell_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        resize_to=resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequences=tuple(args.sequences),
        mask_source=args.mask_source,
        train_augment=False,  # 推理时关闭增强
    )
    
    loader_map = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    loader = loader_map[args.split]
    
    # 加载模型
    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f'已加载模型: {args.checkpoint}')
    print(f'推理数据集: {args.dataset_name} - {args.split} split')
    
    # 批量推理计算指标
    total_dice = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='推理中'):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            batch_size = images.size(0)
            
            logits = model(images)
            batch_dice = dice_coefficient(logits, masks, threshold=args.threshold).item()
            
            total_dice += batch_dice * batch_size
            total_samples += batch_size
    
    avg_dice = total_dice / total_samples
    print(f'\n平均Dice系数: {avg_dice:.4f}')
    
    # 可视化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\n生成可视化结果...')
    visualize_predictions(
        model,
        loader.dataset,
        device,
        output_dir,
        threshold=args.threshold,
        num_samples=args.num_vis,
    )
    print(f'结果已保存至: {output_dir}/inference_samples.png')


if __name__ == '__main__':
    main()
