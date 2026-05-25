import torch

if torch.cuda.is_available():
    print("CUDA 可用。")
    print(f"可用的 GPU 數量：{torch.cuda.device_count()}")
    print(f"當前 GPU 名稱：{torch.cuda.get_device_name(0)}")  # 獲取第一個 GPU 的名稱
else:
    print("CUDA 不可用，請使用 device='cpu'。")
