import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pylab import mpl
from matplotlib.legend import Legend

config = {
    "font.family": "Times New Roman",
    "font.size": 20,
    "font.serif": ["SimSun"],
}
rcParams.update(config)
mpl.rcParams["font.sans-serif"] = ["SimSun"]  # 显示中文
mpl.rcParams["axes.unicode_minus"] = False  # 显示负号

log_csv = pd.read_csv("./record/log_cnn_lts.csv")

epoch = np.array(log_csv["epoch"])
train_loss = np.array(log_csv["train_loss"])
test_loss = np.array(log_csv["test_loss"])
accuracy = np.array(log_csv["accuracy"]) * 100

plot_step = 100

fig, ax1 = plt.subplots()

color = "tab:red"
color1 = "tab:blue"
font_size = 20
ax1.set_xlabel("迭代回合数(CNN_3xConv)", fontsize=font_size, font="SimSun")
ax1.set_ylabel("损失", fontsize=font_size, font="SimSun")
line1 = ax1.plot(
    epoch[:plot_step], train_loss[:plot_step], color=color, linewidth=2.5, label="训练损失"
)
line2 = ax1.plot(
    epoch[:plot_step], test_loss[:plot_step], color=color1, linewidth=2.5, label="测试损失"
)
ax1.legend(prop="SimSun", frameon=False, bbox_to_anchor=(0.45, 0.8))
ax1.tick_params(axis="y")
ax1.spines["top"].set_visible(False)
# ax1.tick_params(axis="x", labelsize=20)

# for label in ax1.get_xticks():
#     label.set_fontsize(50)
#     label.set_font("Times New Roman")
# ax1.set_xticks([0,100,200,300,400], fontsize=20, font="Times New Roman")
# ax1.set_yticks([0,200,400,600,800,1000], fontsize=20, font="Times New Roman")
# plt.xticks(font="Times New Roman")
# plt.yticks(font="Times New Roman")
# plt.show()

ax2 = ax1.twinx()

color2 = "tab:green"
ax2.set_ylabel("正确率/%", fontsize=font_size, font="SimSun")
line3 = ax2.plot(
    epoch[:plot_step], accuracy[:plot_step], color=color2, linewidth=2.5, label="正确率"
)
ax2.tick_params(axis="y")
leg = Legend(
    ax1,
    line3[:2],
    ["正确率"],
    frameon=False,
    bbox_to_anchor=(0.45, 0.58),
    prop="SimSun",
)
ax1.add_artist(leg)
# ax2.spines["left"].set_color(color)
# ax2.spines["right"].set_color(color1)
ax2.spines["top"].set_visible(False)
bwith = 3  # 边框宽度设置为2
TK = plt.gca()  # 获取边框
TK.spines["bottom"].set_linewidth(bwith)  # 图框下边
TK.spines["left"].set_linewidth(bwith)  # 图框左边
TK.spines["top"].set_visible(False)
TK.spines["right"].set_linewidth(bwith)  # 图框右边
fig.tight_layout()
fig.savefig("CNN_LTS.png", dpi=300)
plt.show()
