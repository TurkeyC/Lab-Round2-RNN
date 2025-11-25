import torchaudio
import torch
import numpy as np
from torch.utils.data import DataLoader
import glob
import parameter as p
import matplotlib.pyplot as plt


def my_collate(batch):
    mfcc = [item[0] for item in batch]
    target = [item[1] for item in batch]
    mfcc_size = [item[2] for item in batch]
    return [mfcc, target, mfcc_size]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path, n_mfcc):
        super(MyDataset, self).__init__()
        wav_files = glob.glob(wav_path + "*.wav")
        label_dict = {
            "start": 0,
            "stop": 1,
            "left": 2,
            "right": 3,
            "speedup": 4,
            "speeddown": 5,
            "straight": 6,
            "back": 7,
        }
        wav_list = []
        for wav_file in wav_files:
            for name, label in label_dict.items():
                if name in wav_file:
                    wav_list.append((wav_file, label))

        self.wav_list = wav_list
        self.n_mfcc = n_mfcc
        self.count = 0

    def __getitem__(self, index):
        wav, label = self.wav_list[index]
        # Try torchaudio first. If torchcodec/FFmpeg is missing or incompatible,
        # fall back to soundfile (if installed) or the stdlib wave module.
        try:
            waveform, sample_rate = torchaudio.load(wav, normalize=True)
        except Exception:
            # Try soundfile (pysoundfile) if available
            try:
                import soundfile as sf

                data, sample_rate = sf.read(wav, dtype="float32")
                # soundfile returns shape (n_samples,) or (n_samples, channels)
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(data.T)
            except Exception:
                # Final fallback: use the built-in wave module (supports PCM WAV)
                import wave
                import struct

                with wave.open(wav, "rb") as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(n_frames)
                    # little-endian signed integers
                    fmt = {1: "b", 2: "h", 4: "i"}.get(sampwidth)
                    if fmt is None:
                        raise RuntimeError(f"Unsupported sample width: {sampwidth}")
                    total_samples = struct.unpack("<" + fmt * (n_frames * n_channels), frames)
                    arr = np.array(total_samples, dtype=np.float32)
                    if n_channels > 1:
                        arr = arr.reshape(-1, n_channels).T
                    else:
                        arr = arr.reshape(1, -1)
                    # Normalize to [-1, 1]
                    max_val = float(2 ** (8 * sampwidth - 1))
                    arr = arr / max_val
                    waveform = torch.from_numpy(arr)
                    sample_rate = sr

        '''
         plt.cla()
        plt.plot(
            range(len(waveform.numpy().flatten()))[:15000],
            waveform.numpy().flatten()[:15000],
        )
        plt.axis("off")  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        # plt.show()
        plt.savefig(f"./wave_fig/{self.count}.png", dpi=300, bbox_inches="tight")
        '''
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 23,
                "center": False,
            },
        )
        mfcc = transform(waveform)
        mfcc_size = mfcc.size()[2]  # max lenght 191 #TODO zero padding
        if mfcc_size < p.max_length:  # max_length is a pre-kown hyperparameter
            padding = torch.zeros([1, self.n_mfcc, p.max_length - mfcc_size])
            mfcc = torch.cat((mfcc, padding), 2)
        self.count += 1
        return mfcc, label, mfcc_size

    def __len__(self):
        return len(self.wav_list)


if __name__ == "__main__":
    path = "./dataset/"
    data = MyDataset(path, p.n_mfcc)
    data_loader = DataLoader(data, batch_size=16, shuffle=True)
    for mfcc, label, mfcc_size in data_loader:
        print(mfcc.size())
