from ultralytics import YOLO
import torch

# 全局导入和函数定义可以放在这里（可选）
def some_helper():
    pass

if __name__ == '__main__':
    # 1. 加载模型
    model = YOLO('yolo26s.pt')  # 或 'yolov8s.pt'
    print(666)
    print()

    # 2. 强制搬到 GPU（更推荐用 model.to('cuda')）
    model.model = model.model.cuda()
    # 或者直接用 model.to('cuda')，但 ultralytics 通常自动管理设备

    # 3. 验证设备
    print(f"Using device: {next(model.model.parameters()).device}")

    # 4. 训练（workers=2 会用到多进程，现在被保护了）
   #model.train(
   #    data=r'D:\pythoncharm\pythonProject14\datasets\扣件\data.yaml',
   #    epochs=1,
   #    imgsz=640,
   #    batch=4,
   #    device=0,
   #    workers=2,
   #    project='runs/train',
   #    name='rail_fastener_exp_gi'
   #)

    # 5. 推理
    results = model(
        r'D:\pythoncharm\pythonProject14\datasets\扣件\valid\images\2_20220726125323_jpg.rf.049882188aea12583fe69838d7817d9b.jpg',
        save=True, conf=0.25, device=0
    )
    results[0].show()