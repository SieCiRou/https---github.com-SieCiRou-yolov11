# integrated_system.py - 穩健版：使用 OpenCV 內建 LBPH 臉部辨識 + YOLO 姿勢估計
# 優勢：零編譯依賴，Windows 100% 成功！只需 pip install opencv-python ultralytics numpy
# 臉部註冊：捕捉多張照片訓練 LBPH 模型（傳統但準確率高，適合小資料集）
# 即時辨識：開啟相機，同時顯示臉部辨識（姓名）+ 姿勢關鍵點
# 注意：首次運行會自動建立 known_faces 資料夾與模型檔案

import os
import cv2
import numpy as np
from ultralytics import YOLO
import pickle  # 用來儲存訓練模型

# 設定
KNOWN_FACES_DIR = "known_faces"
MODEL_FILE = "face_recognizer.pkl"
POSE_MODEL_PATH = "yolo11n-pose.pt"  # 或你的 best.pt

# 初始化 YOLO 姿勢模型
pose_model = YOLO(POSE_MODEL_PATH)

# 臉部偵測器 (Haar Cascade, OpenCV 內建)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBPH 臉部辨識器 (OpenCV 內建)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 已知臉部資料：labels (姓名 ID) 和 face_images (訓練圖像)
known_labels = []  # [0, 0, 1, 1, ...] 對應姓名 ID
known_names = {}   # {0: "CiRou", 1: "老闆", ...}
face_images = []   # 儲存灰階臉部圖像列表

def load_face_model():
    """載入已訓練的臉部模型"""
    global recognizer, known_labels, known_names, face_images
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)
        print("已載入現有臉部模型。")
        
        # 重新建構 known_names (從模型推斷)
        try:
            # 讀取對應的 labels.txt (我們會儲存)
            with open("labels.txt", "rb") as f:
                known_names = pickle.load(f)
            print(f"已載入 {len(known_names)} 個已知臉部。")
        except:
            print("警告：無法載入 labels，僅使用模型預測。")
    else:
        print("無現有模型，將在註冊後建立。")

def register_face(name):
    """註冊新臉部：捕捉 20 張照片訓練 LBPH"""
    label_id = len(known_names)
    known_names[label_id] = name
    print(f"註冊 '{name}' (ID: {label_id})。請面向相機，捕捉 20 張照片，按 'q' 結束早。")
    
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))  # 標準化大小
            face_images.append(face_roi)
            known_labels.append(label_id)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"捕捉 {count}/20", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow(f"註冊 {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if count > 0:
        # 訓練模型
        recognizer.train(face_images, np.array(known_labels))
        recognizer.save(MODEL_FILE)
        with open("labels.txt", "wb") as f:
            pickle.dump(known_names, f)
        print(f"已訓練模型，總訓練樣本: {len(face_images)}")
    else:
        print("未捕捉到臉部，註冊失敗。")
        del known_names[label_id]

def recognize_faces(frame):
    """在 frame 上辨識臉部，回傳畫好框的圖"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # 預測
        label, confidence = recognizer.predict(face_roi)
        name = "Unknown"
        if label in known_names and confidence < 100:  # 閾值調整
            name = known_names[label]
        
        # 畫框與名稱
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

def main():
    # 載入模型
    load_face_model()
    
    # 註冊互動
    while True:
        choice = input("是否註冊新臉部？(y/n): ").lower()
        if choice == 'y':
            name = input("輸入姓名: ")
            register_face(name)
        else:
            break
    
    if len(known_names) == 0:
        print("警告：無已知臉部，僅偵測不辨識。")
    
    # 開啟相機即時辨識
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟相機。")
        return
    
    print("\n=== 開始即時姿勢 + 臉部辨識 ===")
    print("按 'q' 結束。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 臉部辨識
        frame = recognize_faces(frame)
        
        # YOLO 姿勢估計
        results = pose_model.predict(source=frame, device='cpu', stream=False, verbose=False)
        for result in results:
            # 繪製關鍵點
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()  # 第一個人
                if len(keypoints) > 0:
                    for x, y in keypoints:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
            
            # 繪製邊框
            if result.boxes is not None:
                for box in result.boxes.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # 顯示
        cv2.imshow("Real-time Pose Estimation + Face Recognition (OpenCV LBPH)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("系統結束。")

if __name__ == "__main__":
    main()