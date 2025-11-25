import os
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wav_loader import MyDataset  # assumes existing file provides mfcc_size
import parameter as p
from network_gru import GRU
from network import LSTM
from network_cnn import CNN, CNNBaseline, CNNUltraFast
try:
    # TritonMissing 在缺失 triton 时用于更细粒度捕捉
    from torch._inductor.exc import TritonMissing  # type: ignore
except Exception:  # pragma: no cover
    class TritonMissing(Exception):  # fallback 定义
        pass


@dataclass
class TrainConfig:
    arch: Literal["gru", "lstm", "cnn", "cnn_fast", "cnn_ultrafast"] = "gru"
    batch_size: int = p.batch_size
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    clip_grad: float = 5.0
    dropout: float = 0.3
    use_attention: bool = True
    use_pack: bool = True
    save_dir: str = "./model/seq_improved"
    log_dir: str = "./record"
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(batch_size: int):
    train_data = MyDataset("./train/", p.n_mfcc)
    test_data = MyDataset("./test/", p.n_mfcc)
    train_len = len(train_data)
    test_len = len(test_data)
    if train_len == 0 or test_len == 0:
        msg = (
            f"数据集为空: train={train_len}, test={test_len}. 请确认:\n"
            "1) 已运行数据划分脚本 (例如 train_test_spilt.py) 并在当前工作目录生成 ./train 与 ./test 中的 .wav 文件。\n"
            "2) .wav 文件命名包含标签关键字(start/stop/left/right/speedup/speeddown/straight/back)。\n"
            "3) 当前运行目录与文件夹层级正确 (pwd 应包含 train/ 与 test/ 目录)。\n"
            "4) 若数据放在其他路径，可修改 build_loaders 中的路径或传入环境变量 DATA_ROOT。"
        )
        raise RuntimeError(msg)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_model(cfg: TrainConfig):
    if cfg.arch == "cnn":
        return CNN(num_class=p.num_class)
    if cfg.arch == "cnn_fast":
        return CNNBaseline(num_class=p.num_class)
    if cfg.arch == "cnn_ultrafast":
        return CNNUltraFast(num_class=p.num_class)
    common_kwargs = dict(
        input_size=p.input_size,
        hidden_size=p.hidden_size,
        num_layer=p.num_layer,
        num_class=p.num_class,
        device=p.device,
        dropout=cfg.dropout,
        use_attention=cfg.use_attention,
        use_pack=cfg.use_pack,
    )
    if cfg.arch == "gru":
        return GRU(**common_kwargs)
    if cfg.arch == "lstm":
        return LSTM(**common_kwargs)
    raise ValueError(f"Unsupported arch: {cfg.arch}")


def train_and_eval(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"seq_improved_{cfg.arch}.csv")

    fast_mode = False
    ultra_mode = False
    fast_allow_early_stop = False
    ultra_target_acc = 1.01
    ultra_step_size = 0
    ultra_gamma = 1.0

    if cfg.arch == "cnn_fast":
        cfg.lr = float(os.environ.get("FAST_LR", p.learning_rate))
        cfg.dropout = 0.0
        cfg.clip_grad = 0.0
        fast_mode = True
        cfg.epochs = int(os.environ.get("FAST_EPOCHS", 20))
        fast_allow_early_stop = os.environ.get("FAST_EARLYSTOP", "0") == "1"
    elif cfg.arch == "cnn_ultrafast":
        cfg.lr = float(os.environ.get("ULTRA_LR", 0.04))
        cfg.dropout = 0.0
        cfg.clip_grad = float(os.environ.get("ULTRA_CLIP", 0.0))
        cfg.weight_decay = float(os.environ.get("ULTRA_WD", 0.0))
        cfg.epochs = int(os.environ.get("ULTRA_EPOCHS", 5))
        cfg.batch_size = int(os.environ.get("ULTRA_BATCH", cfg.batch_size))
        ultra_step_size = int(os.environ.get("ULTRA_STEP", 2))
        ultra_gamma = float(os.environ.get("ULTRA_GAMMA", 0.3))
        ultra_target_acc = float(os.environ.get("ULTRA_TARGET_ACC", 0.98))
        ultra_mode = True
    elif cfg.arch == "cnn":
        cfg.lr = p.learning_rate

    # 通用环境变量覆盖，便于复现特定实验
    env_batch = os.environ.get("TRAIN_BATCH")
    if env_batch:
        cfg.batch_size = max(1, int(env_batch))
    env_lr = os.environ.get("TRAIN_LR")
    if env_lr:
        cfg.lr = float(env_lr)
    env_wd = os.environ.get("TRAIN_WEIGHT_DECAY")
    if env_wd:
        cfg.weight_decay = float(env_wd)
    env_clip = os.environ.get("TRAIN_CLIP")
    if env_clip:
        cfg.clip_grad = float(env_clip)
    env_epochs = os.environ.get("TRAIN_EPOCHS")
    if env_epochs:
        cfg.epochs = max(1, int(env_epochs))
    env_seed = os.environ.get("TRAIN_SEED")
    if env_seed:
        cfg.seed = int(env_seed)
        set_seed(cfg.seed)

    if ultra_mode:
        cfg.batch_size = max(4, cfg.batch_size)

    train_loader, test_loader = build_loaders(cfg.batch_size)

    model = build_model(cfg).to(p.device)
    print(f"[Info] Training on device: {p.device} | arch={cfg.arch} | lr={cfg.lr}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if fast_mode or ultra_mode:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if fast_mode:
        scheduler = None
    elif ultra_mode:
        scheduler = (
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, ultra_step_size), gamma=ultra_gamma)
            if ultra_step_size > 0
            else None
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )

    # 尝试编译优化（PyTorch 2.x）
    # fast_mode 不再使用 torch.compile，确保兼容性与启动速度
    if fast_mode:
        print("[Info] cnn_fast 跳过 torch.compile，使用最简策略")
        if fast_allow_early_stop:
            print("[Info] FAST_EARLYSTOP=1，达到阈值会提前终止")
        else:
            print("[Info] FAST_EARLYSTOP 未开启，将跑满设定的 epoch")
    if ultra_mode:
        print(
            f"[Info] cnn_ultrafast 基线结构 + StepLR(step={max(1, ultra_step_size)}, gamma={ultra_gamma}), 总轮 {cfg.epochs}，目标 {ultra_target_acc:.2%}"
        )

    best_acc = 0.0
    records = ["epoch,train_loss,test_loss,accuracy,lr"]

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for step, (mfcc, label, mfcc_size) in enumerate(train_loader):
            # CNN / CNN_FAST 直接使用 4D 输入；循环模型转换为 [B,T,F]
            if cfg.arch in ("cnn", "cnn_fast", "cnn_ultrafast"):
                x = mfcc.to(p.device)
                logits = model(x)
                label = label.to(p.device) if torch.is_tensor(label) else torch.tensor(label, device=p.device)
            else:
                x = mfcc.squeeze(1).transpose(1, 2).to(p.device)
                lengths = mfcc_size.to(p.device) if torch.is_tensor(mfcc_size) else torch.tensor(mfcc_size, device=p.device)
                label = label.to(p.device) if torch.is_tensor(label) else torch.tensor(label, device=p.device)
                logits = model(x, lengths=lengths)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            if cfg.clip_grad and cfg.arch != "cnn_fast":
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            # OneCycleLR 需按 batch step
            # fast_mode 无调度器步进
            train_losses.append(loss.item())

        train_loss_mean = float(np.mean(train_losses)) if train_losses else 0.0

        # evaluation
        model.eval()
        test_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for mfcc, label, mfcc_size in test_loader:
                if cfg.arch in ("cnn", "cnn_fast", "cnn_ultrafast"):
                    x = mfcc.to(p.device)
                    label = label.to(p.device) if torch.is_tensor(label) else torch.tensor(label, device=p.device)
                    logits = model(x)
                else:
                    x = mfcc.squeeze(1).transpose(1, 2).to(p.device)
                    lengths = mfcc_size.to(p.device) if torch.is_tensor(mfcc_size) else torch.tensor(mfcc_size, device=p.device)
                    label = label.to(p.device) if torch.is_tensor(label) else torch.tensor(label, device=p.device)
                    logits = model(x, lengths=lengths)
                loss = criterion(logits, label)
                test_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (preds == label).sum().item()

        test_loss_mean = float(np.mean(test_losses)) if test_losses else 0.0
        acc = correct / total if total else 0.0
        if scheduler is not None:
            if ultra_mode:
                scheduler.step()
            elif not fast_mode:
                scheduler.step(test_loss_mean)
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{cfg.epochs} | Train {train_loss_mean:.4f} | Test {test_loss_mean:.4f} | Acc {acc*100:.2f}% | LR {lr_current:.5f}"
        )
        records.append(
            f"{epoch},{train_loss_mean:.6f},{test_loss_mean:.6f},{acc:.6f},{lr_current:.6f}"
        )

        # save best
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "acc": acc,
                    "config": cfg.__dict__,
                },
                os.path.join(cfg.save_dir, f"best_{cfg.arch}.pt"),
            )

        # 早停：fast 模式达到高阈值提前结束
        if fast_mode and fast_allow_early_stop and acc >= 0.995 and epoch >= 2:
            print(f"[EarlyStop] cnn_fast 在 epoch {epoch} 达到 {acc*100:.2f}% 提前结束")
            break
        if ultra_mode and acc >= ultra_target_acc:
            print(f"[EarlyStop] cnn_ultrafast 在 epoch {epoch} 达到 {acc*100:.2f}% 提前结束")
            break

    # final log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(records))
    print(f"Best accuracy: {best_acc*100:.2f}%")
    return best_acc


if __name__ == "__main__":
    # basic CLI via env vars (optional quick override)
    arch = os.environ.get("ARCH", "gru")
    cfg = TrainConfig(arch=arch)
    train_and_eval(cfg)
