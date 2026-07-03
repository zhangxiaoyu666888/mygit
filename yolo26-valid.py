"""
扣件缺陷检测 - 推理脚本
使用训练好的 best.pt 对单张图片进行推理并显示结果
"""
from ultralytics import YOLO

# 1. 加载训练好的最佳权重
# 路径根据你的实际训练结果调整（通常为 runs/train/fastener_exp1/weights/best.pt）
model_path = r'D:\pythoncharm\pythonProject14\runs\detect\runs\train\fastener_exp1\weights\best.pt'
model = YOLO(model_path)

# 2. 测试图片路径（这里用你之前那张图片，可替换成任意图片）
test_image = r'D:\pythoncharm\pythonProject14\铁路数据集\扣件\valid\images\Namnlo-st-105_jpeg_jpg.rf.4dd27ffd28a1a48b974791320a32f0f8.jpg'

# 3. 推理
results = model(test_image, save=True, conf=0.25, device=0)

# 4. 在屏幕上显示结果（会弹出图片窗口）
results[0].show()

# 5. 输出检测到的信息
boxes = results[0].boxes
if boxes is not None and len(boxes) > 0:
    print(f"检测到 {len(boxes)} 个目标:")
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        class_name = model.names[cls]
        print(f"  - {class_name}: 置信度 {conf:.2f}")
else:
    print("未检测到任何目标")