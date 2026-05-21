from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # n是最小最快的版本

# 开始训练
results = model.train(
    data='yolo_dataset/data.yaml',  # 你的数据集配置
    epochs=20,                       # 训练50轮（可以先跑50试试）
    imgsz=320,                       # 图片尺寸（和你的数据集一致）
    batch=4,                         # 批次大小（内存不够就改小）
    device='cpu',                    # 用CPU训练（没有GPU就用这个）
    workers=0,                       # Windows必须设为0
    name='helmet_detection'          # 实验名称
)

print("训练完成！")
