import os

from wav_loader import MyDataset
from network_cnn import CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import parameter as p
import numpy as np
import pandas as pd


def build_dataloaders():
    train_data = MyDataset("./train/", p.n_mfcc)
    test_data = MyDataset("./test/", p.n_mfcc)

    if p.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin_memory = p.device.type == "cuda"
    num_workers = min(4, os.cpu_count() or 1)

    train_loader_kwargs = {
        "batch_size": p.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
    }
    test_loader_kwargs = {
        "batch_size": p.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
    }

    if pin_memory:
        train_loader_kwargs["pin_memory"] = True
        test_loader_kwargs["pin_memory"] = True

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        test_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_data, **train_loader_kwargs)
    test_loader = DataLoader(test_data, **test_loader_kwargs)
    return train_loader, test_loader, pin_memory


def train():
    train_data_loader, test_data_loader, pin_memory = build_dataloaders()
    os.makedirs("./model/cnn_model", exist_ok=True)
    os.makedirs("./record", exist_ok=True)
    Net = CNN(p.num_class).to(p.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=p.learning_rate)
    logger = {"epoch": [], "train_loss": [], "test_loss": [], "accuracy": []}

    total_step = len(train_data_loader)
    for epoch in range(p.num_epochs):
        Net.train()
        train_loss = []
        for i, (mfcc, label, mfcc_size) in enumerate(train_data_loader):
            mfcc = mfcc.to(p.device, non_blocking=pin_memory)
            label = label.to(p.device, non_blocking=pin_memory)
            output = Net(mfcc)
            loss = criterion(output, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{p.num_epochs}], Step [{i}/{total_step}]], Loss {loss.item()}"
                )
        train_loss_mean = np.mean(train_loss)

        test_loss = []
        Net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (mfcc, label, mfcc_size) in enumerate(test_data_loader):
                mfcc = mfcc.to(p.device, non_blocking=pin_memory)
                label = label.to(p.device, non_blocking=pin_memory)
                output = Net(mfcc)
                loss = criterion(output, label)
                test_loss.append(loss.item())
                _, predict = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predict == label).sum().item()

            print(f"Test Accuracy of the model: {100 * correct / total} %")
        test_loss_mean = np.mean(test_loss)

        logger["epoch"].append(epoch)
        logger["train_loss"].append(train_loss_mean)
        logger["test_loss"].append(test_loss_mean)
        logger["accuracy"].append(correct / total)
        torch.save(Net, f"./model/cnn_model/cnn_{epoch}.pth")

    df = pd.DataFrame(logger)
    df.to_csv("./record/log_cnn.csv", index=False)


if __name__ == "__main__":
    train()
