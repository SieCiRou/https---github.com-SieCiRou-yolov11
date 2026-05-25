# final_gui_system.py
# 2025 終極穩定版：Tkinter 美觀介面 + OpenCV LBPH 臉部辨識 + YOLOv11 姿勢估計
# 安裝指令：pip install opencv-python ultralytics numpy pillow

import os
import pickle
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from ultralytics import YOLO

# ==================== 全域設定 ====================
KNOWN_FACES_DIR = "known_faces"
MODEL_FILE = "face_recognizer.yml"  # OpenCV 標準格式
LABELS_FILE = "labels.pkl"
POSE_MODEL_PATH = "yolo11n-pose.pt"  # 或你的 best.pt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
pose_model = YOLO(POSE_MODEL_PATH)

known_names = {}  # {0: "CiRou", 1: "Mom", ...}

# === UI Colors ===
COLORS = {
    "text": "#1E90FF",
    "Unknown_face": (114, 128, 250),  # Salmon
    "member_face": (237, 149, 100),  # Blue
    "idle_face": (122, 150, 233),  # Gray
}


# ==================== 臉部模型載入 ====================
def load_face_model():
    global known_names
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        recognizer.read(MODEL_FILE)
        with open(LABELS_FILE, "rb") as f:
            known_names = pickle.load(f)
        print(f"已載入臉部模型，共 {len(known_names)} 人")
    else:
        print("無臉部模型，將從零開始")


# ==================== 註冊新臉部 ====================
def register_new_face(name):
    if name in known_names.values():
        messagebox.showwarning("警告", f"{name} 已存在！")
        return

    cap = cv2.VideoCapture(0)
    faces = []
    count = 0
    max_samples = 30

    messagebox.showinfo("開始註冊", f"請正對鏡頭，系統將捕捉 {max_samples} 張臉部照片\n按任意鍵繼續")

    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in detected:
            if count >= max_samples:
                break
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (150, 150))
            faces.append(face_roi)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"{count}/{max_samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("註冊中 - 按 q 提早結束", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) < 10:
        messagebox.showerror("失敗", "捕捉臉部太少，註冊失敗")
        return

    # 訓練或更新模型
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)
        recognizer.update(faces, np.array([len(known_names)] * len(faces)))
    else:
        recognizer.train(faces, np.array([0] * len(faces)))

    known_names[len(known_names)] = name
    recognizer.write(MODEL_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(known_names, f)

    messagebox.showinfo("成功", f"{name} 註冊成功！共收集 {len(faces)} 張照片")


# ==================== 主 GUI 類別 ====================
class FacePoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 姿勢 + 臉部辨識系統 v2025")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        self.cap = None
        self.running = False

        self.setup_ui()
        load_face_model()

    def setup_ui(self):
        # 標題
        title = tk.Label(
            self.root, text="即時姿勢與臉部辨識系統", font=("微軟正黑體", 20, "bold"), bg="#2c3e50", fg="#ecf0f1"
        )
        title.pack(pady=10)

        # 影片區
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)

        # 控制按鈕
        btn_frame = tk.Frame(self.root, bg="#2c3e50")
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="開始辨識",
            command=self.start_camera,
            font=("微軟正黑體", 14),
            bg="#27ae60",
            fg="white",
            width=12,
        ).pack(side=tk.LEFT, padx=10)
        tk.Button(
            btn_frame,
            text="停止",
            command=self.stop_camera,
            font=("微軟正黑體", 14),
            bg="#c0392b",
            fg="white",
            width=12,
        ).pack(side=tk.LEFT, padx=10)
        tk.Button(
            btn_frame,
            text="註冊新臉部",
            command=self.register_gui,
            font=("微軟正黑體", 14),
            bg="#2980b9",
            fg="white",
            width=15,
        ).pack(side=tk.LEFT, padx=10)

        # 狀態列
        self.status = tk.StringVar(value="狀態：已就緒")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status,
            font=("微軟正黑體", 12),
            bg="#34495e",
            fg="#ecf0f1",
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 已註冊人員列表
        list_frame = tk.LabelFrame(self.root, text="已註冊人員", font=("微軟正黑體", 12), bg="#2c3e50", fg="white")
        list_frame.pack(pady=10, fill=tk.X, padx=20)
        self.name_listbox = tk.Listbox(list_frame, height=6, font=("Consolas", 11))
        self.name_listbox.pack(fill=tk.X, padx=10, pady=5)
        self.update_name_list()

    def update_name_list(self):
        self.name_listbox.delete(0, tk.END)
        for name in known_names.values():
            self.name_listbox.insert(tk.END, f" {name}")

    def register_gui(self):
        name = simpledialog.askstring("註冊", "請輸入姓名：")
        if name and name.strip():
            threading.Thread(target=register_new_face, args=(name.strip(),), daemon=True).start()
            self.root.after(1000, self.update_name_list)

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟相機！")
            return
        self.running = True
        self.status.set("狀態：辨識中...")
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.status.set("狀態：已停止")

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 臉部辨識
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                roi = cv2.resize(gray[y : y + h, x : x + w], (150, 150))
                if os.path.exists(MODEL_FILE):
                    label, confidence = recognizer.predict(roi)
                    name = known_names.get(label, "Unknown")
                    if confidence < 80:
                        text = name
                        color = COLORS["member_face"]
                    else:
                        text = "Unknown"
                        color = COLORS["Unknown_face"]
                else:
                    text = "未訓練"
                    color = COLORS["idle_face"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{text} ({confidence:.0f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # YOLO 姿勢估計（已修好 numpy.float32 錯誤）
            results = pose_model(frame, verbose=False, device="cpu")
            for result in results:
                if result.keypoints is not None and result.keypoints.xy is not None:
                    kpts = result.keypoints.xy.cpu().numpy()
                    # 修復：kpts 可能是 (n, 17, 2) 或 (17, 2) 或單一 float
                    if kpts.ndim == 3:  # 多個人
                        for person in kpts:
                            for x, y in person:
                                if x > 0 and y > 0:
                                    cv2.circle(frame, (int(x), int(y)), 6, (255, 215, 0), -1)
                    elif kpts.ndim == 2:  # 一個人
                        for x, y in kpts:
                            if x > 0 and y > 0:
                                cv2.circle(frame, (int(x), int(y)), 6, (255, 215, 0), -1)

            # 轉成 Tkinter 可顯示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((860, 540), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.video_label.configure(image=photo)
            self.video_label.image = photo  # 保持參考

            time.sleep(0.03)

        self.video_label.configure(image="")

    def on_closing(self):
        if messagebox.askokcancel("退出", "確定要關閉程式嗎？"):
            self.stop_camera()
            self.root.destroy()


# ==================== 程式進入點 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = FacePoseApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
