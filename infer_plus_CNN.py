import os
import time

import torch
from torch.utils.data import DataLoader

from wav_loader import MyDataset
import parameter as p


def main():
    model_path = os.path.join("model", "cnn_model_lts", "cnn_20.pth")
    model = torch.load(model_path, map_location=p.device, weights_only=False)
    model = model.to(p.device)
    model.eval()

    data = MyDataset("./dataset/", p.n_mfcc)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)

    use_cuda = p.device.type == "cuda"
    if use_cuda:
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
    else:
        starter = ender = None

    infer_times = []

    for i, (mfcc, label, mfcc_size) in enumerate(data_loader):
        mfcc = mfcc.to(p.device)
        if use_cuda:
            starter.record()
        else:
            start_time = time.perf_counter()

        with torch.no_grad():
            output = model(mfcc)

        if use_cuda:
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
        else:
            curr_time = (time.perf_counter() - start_time) * 1000.0

        infer_times.append(curr_time)
        print(f"Sample {i + 1} infer time: {curr_time:.2f} ms")

    avg_time = sum(infer_times) / len(infer_times)
    print(f"\nAverage inference time over {len(infer_times)} samples: {avg_time:.2f} ms")


if __name__ == "__main__":
    main()