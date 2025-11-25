import torch
import torch.nn as nn
from wav_loader import MyDataset
from torch.utils.data import DataLoader
import parameter as p

model = torch.load("model\cnn_model_lts\cnn_20.pth", weights_only=False)
model.eval()

data = MyDataset("./dataset/", p.n_mfcc)
data_loader = DataLoader(data, batch_size=1, shuffle=True)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

infer_times = []  # ← 新增：用于记录每次推理时间

for i, (mfcc, label, mfcc_size) in enumerate(data_loader):
    mfcc = mfcc.squeeze(1).transpose(1, 2).to(p.device)
    label = label.to(p.device)
    
    starter.record()
    with torch.no_grad():  # ← 建议加上，避免计算梯度（提速 + 节省内存）
        output = model(mfcc)
    ender.record()
    
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)  # 单位：毫秒
    infer_times.append(curr_time)
    print(f"Sample {i+1} infer time: {curr_time:.2f} ms")

# ← 新增：计算并打印平均推理时间
avg_time = sum(infer_times) / len(infer_times)
print(f"\nAverage inference time over {len(infer_times)} samples: {avg_time:.2f} ms")