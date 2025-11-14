# train_model.py
import os
from ultralytics import YOLO

# 1. 模型設定與加載
# 如果是第一次訓練，建議從預訓練的骨架模型開始 (例如 yolo11n-pose.pt)
# 如果是繼續訓練，請加載您上次訓練中途生成的 best.pt 權重檔案
model = YOLO(r"C:\Users\CiRou\Dev\ultralytics-main\runs\pose\train\weights\best.pt") 

# 2. 訓練配置
# data: 官方資料集配置文件，例如 coco8-pose.yaml (Ultralytics內建)
# epochs: 訓練迭代次數 (請根據您的需求調整)
# imgsz: 圖片尺寸 (建議保持在 640 或 1280)
# project/name: 設置訓練結果的儲存路徑，例如 runs/pose/my_custom_train
results = model.train(
    data="coco8-pose.yaml", 
    epochs=100, 
    imgsz=640, 
    device='cpu', 
    project="runs/pose",
    name="my_custom_train"
)

# 3. 模型權重保存位置說明
print("\n--- 訓練完成 ---")
print("最佳訓練權重 (best.pt) 將保存在：")
print("runs/pose/my_custom_train/weights/best.pt")
print("請記住這個路徑，它將用於測試程式。")