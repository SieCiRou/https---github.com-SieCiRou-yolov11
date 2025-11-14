# test_inference.py
import os
from ultralytics import YOLO

# 1. 加載您訓練好的模型
# 請將此處路徑替換為 train_model.py 輸出的 best.pt 檔案路徑
trained_model_path = r"runs/pose/my_custom_train2/weights/best.pt" 
model = YOLO(trained_model_path) 

# 2. 設置您的照片集路徑
# 請將此路徑替換為您本地存放照片的資料夾
my_image_folder = r"Yolov11_1113\yolo_train_img"

print(f"\n=== 使用訓練好的模型對 {my_image_folder} 進行推理 ===")

# 3. 執行推理
# stream=True 適用於處理大量圖片/影片，記憶體效率更高
results = model.predict(source=my_image_folder, device='cpu', stream=True)

# 4. 處理並保存結果
for i, result in enumerate(results):
    print(f"--- 處理圖片 {i+1} ---")
    # result.save()：將帶有骨架的圖片保存到 'runs/detect/predict*/' 資料夾中
    # 注意：推理結果的路徑通常是 runs/detect/predict 或 runs/pose/predict，
    # 這裡會根據模型自動創建一個新的 'predict' 資料夾。
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    keypoints_xy = result.keypoints.xy
    result.save() 
    
    # 可以在控制台即時顯示
    # result.show() 

print("\n--- 測試完成 ---")
# 輸出結果會保存在: [執行腳本的目錄]/runs/pose/predict[N]/
print(f"帶有骨架的測試結果已保存至 runs/pose/predict*/ 資料夾中。")

# 可選：如果您需要提取關鍵點數據，請參考上一個回答中的 .txt 標註生成代碼。