from wav_loader import MyDataset
from network_cnn import CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import parameter as p
import numpy as np
import pandas as pd


train_data = MyDataset("./train/", p.n_mfcc)
test_data = MyDataset("./test/", p.n_mfcc)

train_data_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=True)

Net = CNN(p.num_class).to(p.device) # 简化了模型初始化，假设CNN构造函数只接收num_class

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = torch.optim.Adam(Net.parameters(), lr=p.learning_rate)
logger = {"epoch": [], "train_loss": [], "test_loss": [], "accuracy": []}

total_step = len(train_data_loader)
for epoch in range(p.num_epochs):
    train_loss = []
    for i, (mfcc, label, mfcc_size) in enumerate(train_data_loader):
        mfcc = mfcc.to(p.device)
        label = label.to(p.device)
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
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (mfcc, label, mfcc_size) in enumerate(test_data_loader):
            mfcc = mfcc.to(p.device)
            label = label.to(p.device)
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
    torch.save(Net, f"./model/cnn_model_lts/cnn_{epoch}.pth")

df = pd.DataFrame(logger)
df.to_csv("./record/log_cnn_lts.csv", index=False)
