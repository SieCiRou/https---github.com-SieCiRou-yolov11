# yolo_face_pose_system.py - 完全 YOLO 版：臉部辨識 + 姿勢 + Webcam
# 基於 Medium 文章升級：用 Ultralytics YOLOv11 取代 Darkflow
# 使用：先跑 train_custom() 註冊/訓練，然後 run_webcam()

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==================== 設定 ====================
DATASET_DIR = Path("dataset")  # 你的資料夾
YAML_PATH = DATASET_DIR / "dataset.yaml"
MODEL_PATH = "runs/detect/train/weights/best.pt"  # 訓練後模型
POSE_MODEL = YOLO("yolo11n-pose.pt")  # 預訓練姿勢模型

# 類別映射 (從 YAML)
CLASS_NAMES = {0: 'robert', 1: 'candy', 2: 'person'}  # 依你的 labels.txt 調整

def train_custom():
    """階段1: 訓練自訂臉部模型 (註冊 = 標註後訓練)"""
    if not os.path.exists(DATASET_DIR):
        print("請先準備 dataset/ 資料夾與 labelImg 標註！")
        return
    
    # 載入預訓練偵測模型並訓練
    model = YOLO("yolo11n.pt")  # 從偵測預訓練開始
    results = model.train(
        data=str(YAML_PATH),      # YAML 設定
        epochs=100,               # 文章建議 1000，但 YOLOv11 快，100 足夠
        imgsz=640,
        batch=16,
        project="runs/detect",    # 輸出資料夾
        name="train",
        device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'  # GPU if available
    )
    print(f"訓練完成！模型存於 {MODEL_PATH}")
    # 驗證
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")

def run_webcam():
    """階段2: Webcam 即時辨識 - 臉部類別 + 姿勢關鍵點"""
    if not os.path.exists(MODEL_PATH):
        print("請先訓練模型！跑 train_custom()")
        return
    
    # 載入自訂臉部模型
    face_model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(0)  # Webcam
    if not cap.isOpened():
        print("無法開啟 Webcam！")
        return
    
    print("即時辨識開始，按 'q' 結束。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 臉部/人體偵測 (自訂模型)
        results = face_model.predict(source=frame, conf=0.5, device='cpu')  # 調整 conf 閾值
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 提取類別與置信度
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = CLASS_NAMES.get(cls, 'Unknown')
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 畫臉部/人體框 (你的示例風格：藍框 + 名稱 + conf)
                    color = (255, 0, 0) if name != 'person' else (0, 255, 0)  # 臉部藍，人體綠
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 姿勢估計 (用 pose 模型，文章未提但你的圖有姿勢線)
        pose_results = POSE_MODEL.predict(source=frame, conf=0.5, device='cpu')
        for result in pose_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()  # (17, 2)
                for i, (x, y) in enumerate(keypoints):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # 紅點關鍵點
                # 畫骨架線 (COCO 17 點連接，簡化版)
                skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,0],[0,15],[0,16]]
                for kp1, kp2 in skeleton:
                    p1 = tuple(keypoints[kp1].astype(int))
                    p2 = tuple(keypoints[kp2].astype(int))
                    if all(p1) and all(p2):
                        cv2.line(frame, p1, p2, (255, 255, 0), 2)  # 黃線骨架
        
        cv2.imshow("YOLOv11: 臉部辨識 + 姿勢 (q 結束)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 執行範例
if __name__ == "__main__":
    # 先訓練 (註冊階段)
    # train_custom()
    
    # 然後即時 Webcam
    run_webcam()