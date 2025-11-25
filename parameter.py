import os
import torch

# 设备选择逻辑：默认期望使用 CUDA；若不可用且未显式要求 CPU，则给出警告
_env_device = os.environ.get("DEVICE", "cuda").lower()  # 可选: cuda / cpu
if _env_device.startswith("cuda"):
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		print("[警告] 请求使用 CUDA 但当前不可用，自动回退到 CPU。若需静默回退，设置 DEVICE=cpu")
		device = torch.device("cpu")
elif _env_device == "cpu":
	device = torch.device("cpu")
else:
	# 未识别的值，回退逻辑
	print(f"[警告] 未识别的 DEVICE 环境变量值: {_env_device}，使用自动选择。")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[Info] 使用设备: {device}")

# preprocess parameters
max_length = 191  # 若真实为 192，可在此改为 192
n_mfcc = 13

# network parameters
input_size = n_mfcc
hidden_size = 128
num_layer = 2
num_class = 8
batch_size = 32
num_epochs = 100
learning_rate = 0.01  # 改进脚本中使用 1e-3，可在调用处覆盖
