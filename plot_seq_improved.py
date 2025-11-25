import os
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_log(path: str):
    epochs, train_loss, test_loss, acc, lr = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            test_loss.append(float(row["test_loss"]))
            acc.append(float(row["accuracy"]))
            lr.append(float(row["lr"]))
    return epochs, train_loss, test_loss, acc, lr


def plot_curves(log_path: str, out_path: str):
    epochs, train_loss, test_loss, acc, lr = load_log(log_path)
    if not epochs:
        print("日志为空，确认是否已训练。")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="Train Loss", color="#d62728")
    ax1.plot(epochs, test_loss, label="Test Loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [a * 100 for a in acc], label="Accuracy %", color="#2ca02c")
    ax2.set_ylabel("Accuracy (%)")
    acc_line = ax2.lines[0]
    lines = ax1.lines + [acc_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"保存曲线图: {out_path}")

    # 另外输出学习率曲线
    fig_lr, ax_lr = plt.subplots(figsize=(8, 3))
    ax_lr.plot(epochs, lr, color="#9467bd")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.grid(alpha=0.3, linestyle="--")
    fig_lr.tight_layout()
    lr_out = out_path.replace(".png", "_lr.png")
    fig_lr.savefig(lr_out, dpi=300)
    print(f"保存学习率图: {lr_out}")


def main():
    arch = os.environ.get("ARCH", "gru")
    log_path = f"./record/seq_improved_{arch}.csv"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if not Path(log_path).exists():
        print(f"未找到日志文件: {log_path}")
        sys.exit(1)
    out_path = f"seq_improved_{arch}.png"
    plot_curves(log_path, out_path)


if __name__ == "__main__":
    main()
