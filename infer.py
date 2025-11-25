import torch
import torch.nn as nn
from wav_loader import MyDataset
from torch.utils.data import DataLoader
import parameter as p

model = torch.load("model\model_98.pth", weights_only=False)
model.eval()

data = MyDataset("./dataset/", p.n_mfcc)
data_loader = DataLoader(data, batch_size=1, shuffle=True)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)

for i, (mfcc, label, mfcc_size) in enumerate(data_loader):
    mfcc = mfcc.squeeze(1).transpose(1, 2).to(p.device)
    label = label.to(p.device)
    starter.record()
    output = model(mfcc)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)  # 计算时间
    print(f"infer time: {curr_time} ms")
