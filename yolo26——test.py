"""
扣件缺陷检测 - 训练脚本
使用 YOLOv8s 作为起点，在 RTX 3050 上训练
"""
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # 加载预训练模型
    model = YOLO('yolo26s.pt')

    # 训练
    results = model.train(
        data=r'D:\pythoncharm\pythonProject14\datasets\扣件\data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,        # 多进程加载可保留
        project='runs/train',
        name='fastener_exp1_26s',#改
        #optimizer='SGD',
        #close_mosaic=15,
        exist_ok=True
    )
    print(f"训练完成，最佳权重保存在: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == '__main__':
    main()