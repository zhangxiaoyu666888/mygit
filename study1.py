from ultralytics import YOLO
model = YOLO('yolo26s.pt')
print(model.task)
print(model.names)
print(sum(p.numel() for p in model.parameters()))
print(888)
#print(666)
model.predict(
    source=0,#摄像头
    save=False,
    show=True,
)