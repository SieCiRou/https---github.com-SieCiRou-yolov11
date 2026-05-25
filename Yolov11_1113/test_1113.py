import datetime
import os
import sys

import clearml

from ultralytics import YOLO


def main():
    clearml.browser_login()

    # 建立日誌
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")

    export_path = r"C:\Users\CiRou\Dev\ultralytics-main\Yolov11_1113"
    my_image_folder = r"C:\Users\CiRou\Dev\ultralytics-main\Yolov11_1113\yolo_train_img"

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print(f"=== YOLOv11 訓練啟動於 {timestamp} ===\n")

    # 載入模型
    model = YOLO("yolo11n-pose.pt").load("yolo11n-pose.pt")

    # 訓練模型
    results = model.train(
        data="coco8-pose.yaml",
        epochs=100,
        imgsz=640,
        project="runs/pose",
        name="my_custom_train",
        device=0,  # 改用 GPU
    )

    # 驗證模型
    metrics = model.val()
    print("\n=== 驗證結果 ===")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Pose mAP50-95: {metrics.pose.map:.4f}")

    # 預測
    results = model.predict(source=my_image_folder, save=True, device=0)
    for i, result in enumerate(results):
        print(f"--- 處理圖片 {i + 1} ---")
        result.save()

    # 匯出模型
    model.export(format="onnx")
    print(f"\n模型已匯出：{export_path}")
    print(f"\n日誌已保存於：{log_file}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # ✅ 關鍵修正（for Windows）
    main()
