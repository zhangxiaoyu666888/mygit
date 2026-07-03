"""
扣件缺陷检测 - 训练脚本
使用 YOLOv8s 作为起点，在 RTX 3050 上训练
"""
from ultralytics import YOLO
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # 加载预训练模型
    model = YOLO('yolo26s.pt')
    results = model.train(
        data=r'D:\pythoncharm\pythonProject14\datasets\钢轨表面\data.yaml',
        epochs=50,  # 增加到150
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project='runs/train',
        name='rail_surface_exp_50',
        exist_ok=True,
        mosaic=1.0,
        close_mosaic=5,  # 最后15个epoch关闭Mosaic
        mixup=0.1,  # 加入MixUp增强
        copy_paste=0.5,  # 对少数类复制粘贴
        cls_pw=0.5,  # 类别权重提高，特别是少数类（需在data.yaml中指定类别权重）
        cos_lr=True,  # 余弦退火学习率
        lr0=0.01,  # 初始学习率稍高
        warmup_epochs=5,  # 前5个epoch预热
        patience=15,  # 如果50个epoch无提升则早停
    )
    print(f"训练完成，最佳权重保存在: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == '__main__':
    main()