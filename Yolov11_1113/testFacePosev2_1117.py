import os

import cv2
import numpy as np

from ultralytics import YOLO

# ==================== 設定 ====================
MODEL_PATH = "runs/detect/train/weights/best.pt"  # 你的自訓臉部模型
POSE_MODEL = YOLO("yolo11n-pose.pt")  # 姿勢模型

# 請依照你的 dataset.yaml 調整順序
CLASS_NAMES = {0: "robert", 1: "candy", 2: "person"}

# 正確的 COCO 17 點骨架連接（0-16）
SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],  # arms
    [11, 12],
    [5, 11],
    [6, 12],  # shoulders
    [5, 6],
    [5, 7],
    [6, 8],  # hips
    [7, 9],
    [8, 10],  # legs
    [1, 2],
    [0, 1],
    [0, 2],  # face
    [0, 5],
    [0, 6],  # nose to shoulders
]


def run_webcam():
    if not os.path.exists(MODEL_PATH):
        print(f"找不到臉部模型：{MODEL_PATH}")
        print("請先用 labelImg 標註 + 訓練出 best.pt")
        return

    face_model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("即時辨識啟動！按 q 離開")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === 1. 自訓模型：臉部辨識 + person ===
        results = face_model(frame, conf=0.4, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = CLASS_NAMES.get(cls, "unknown")

            # 藍框=特定人，綠框=person
            color = (255, 0, 0) if name in ["robert", "candy"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # === 2. 姿勢估計（已兼容最新版 has_visible 為 bool 的情況）===
        pose_results = POSE_MODEL(frame, conf=0.5, verbose=False)[0]

        if pose_results.keypoints is not None:
            kpts = pose_results.keypoints.xy.cpu().numpy()  # shape: (n_person, 17, 2)
            # 處理 visibility（新版可能是 bool，舊版是 tensor）
            if pose_results.keypoints.has_visible is not None:
                if isinstance(pose_results.keypoints.has_visible, bool):
                    # 新版：True 表示有 visibility 欄位，但實際資料在 .data
                    vis = pose_results.keypoints.data[:, :, 2].cpu().numpy() > 0.5
                else:
                    # 舊版
                    vis = pose_results.keypoints.has_visible.cpu().numpy()
            else:
                vis = np.ones_like(kpts[..., 0], dtype=bool)  # 全可見

            # 支援多個人
            for person_kpts, person_vis in zip(kpts, vis):
                # 畫關鍵點（紅點）
                for (x, y), visible in zip(person_kpts, person_vis):
                    if x > 0 and y > 0 and visible:
                        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

                # 畫骨架線（亮黃色）
                for i, j in SKELETON:
                    if person_vis[i] and person_vis[j] and person_kpts[i][0] > 0 and person_kpts[j][0] > 0:
                        pt1 = (int(person_kpts[i][0]), int(person_kpts[i][1]))
                        pt2 = (int(person_kpts[j][0]), int(person_kpts[j][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

        cv2.imshow("YOLOv11 臉部辨識 + 姿勢估計 (2025 最終穩定版)", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
