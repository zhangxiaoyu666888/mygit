from ultralytics import YOLO
import torch

# 1. 加载模型
model = YOLO('yolo26s.pt')  # 先用 yolov8s.pt 验证

# 2. 强制把所有模型参数搬到 GPU
model.model = model.model.cuda()

# 3. 验证设备
print(f"Using device: {next(model.model.parameters()).device}")

# 4. 推理时明确指定用 GPU
results = model.train(
    data=r'D:\pythoncharm\pythonProject14\铁路数据集\扣件\data.yaml',
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    project='runs/train',
    name='rail_fastener_exp1'
)
results = model(
    r'D:\pythoncharm\pythonProject14\铁路数据集\扣件\valid\images\Namnlo-st-105_jpeg_jpg.rf.4dd27ffd28a1a48b974791320a32f0f8.jpg',
    save=True, conf=0.25, device=0
)
results[0].show()