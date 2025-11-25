# 循环神经网络

到目前为止，我们遇到过两种类型的数据：表格数据和图像数据。对于图像数据，我们设计了专门的卷积神经网络架构来为这类特殊的数据结构建模。换句话说，如果我们拥有一张图像，我们需要有效地利用其像素位置，假若我们对图像中的像素位置进行重排，就会对图像中内容的推断造成极大的困难。

最重要的是，到目前为止我们默认数据都来自于某种分布，并且所有样本都是独立同分布的（independently and identically distributed, i.i.d.）。然而，大多数的数据并非如此。例如，文章中的单词是按顺序写的，如果顺序被随机地重排，就很难理解文章原始的意思。同样，视频中的图像帧、对话中的音频信号以及网站上的浏览行为都是有顺序的。因此，针对此类数据而设计特定模型，可能效果会更好。

另一个问题来自这样一个事实：我们不仅仅可以接收一个序列作为输入，而是还可能期望继续猜测这个序列的后续。例如，一个任务可以是继续预测2,4,6,8,10,...。这在时间序列分析中是相当常见的，可以用来预测股市的波动、患者的体温曲线或者赛车所需的加速度。同理，我们需要能够处理这些数据的特定模型。

简言之,如果说卷积神经网络可以有效地处理空间信息,那么本章的循环神经网络（recurrent neural network, RNN）则可以更好地处理序列信息。循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。

许多使用循环网络的例子都是基于文本数据的，因此我们将在本章中重点介绍语言模型。在对序列数据进行更详细的回顾之后，我们将介绍文本预处理的实用技术。然后，我们将讨论语言模型的基本概念，并将此讨论作为循环神经网络设计的灵感。最后，我们描述了循环神经网络的梯度计算方法，以探讨训练此类网络时可能遇到的问题。

# 8.1 序列模型

想象一下有人正在看网飞（Netflix，一个国外的视频网站）上的电影。一名忠实的用户会对每一部电影都给出评价，毕竟一部好电影需要更多的支持和认可。然而事实证明，事情并不那么简单。随着时间的推移，人们对电影的看法会发生很大的变化。事实上，心理学家甚至对这些现象起了名字：

- 锚定（anchoring）效应：基于其他人的意见做出评价。例如，奥斯卡颁奖后，受到关注的电影的评分会上升，尽管它还是原来那部电影。这种影响将持续几个月，直到人们忘记了这部电影曾经获得的奖项。结果表明((Wu et al., 2017))，这种效应会使评分提高半个百分点以上。  
- 享乐适应 (hedonic adaption): 人们迅速接受并且适应一种更好或者更坏的情况作为新的常态。例如,在看了很多好电影之后, 人们会强烈期望下部电影会更好。因此, 在许多精彩的电影被看过之后, 即使是一部普通的也可能被认为是糟糕的。  
- 季节性(seasonality)：少有观众喜欢在八月看圣诞老人的电影。  
- 有时，电影会由于导演或演员在制作中的不当行为变得不受欢迎。  
- 有些电影因为其极度糟糕只能成为小众电影。Plan9from Outer Space和Troll2就因为这个原因而臭名昭著的。

简而言之，电影评分决不是固定不变的。因此，使用时间动力学可以得到更准确的电影推荐 (Koren, 2009)。当然，序列数据不仅仅是关于电影评分的。下面给出了更多的场景。

- 在使用程序时，许多用户都有很强的特定习惯。例如，在学生放学后社交媒体应用更受欢迎。在市场开放时股市交易软件更常用。  
- 预测明天的股价要比过去的股价更困难，尽管两者都只是估计一个数字。毕竟，先见之明比事后诸葛亮难得多。在统计学中，前者（对超出已知观测范围进行预测）称为外推法（extrapolation），而后者（在现有观测值之间进行估计）称为内插法（interpolation）。  
- 在本质上，音乐、语音、文本和视频都是连续的。如果它们的序列被我们重排，那么就会失去原有的意义。比如，一个文本标题“狗咬人”远没有“人咬狗”那么令人惊讶，尽管组成两句话的字完全相同。  
- 地震具有很强的相关性，即大地震发生后，很可能会有几次小余震，这些余震的强度比非大地震后的余震要大得多。事实上，地震是时空相关的，即余震通常发生在很短的时间跨度和很近的距离内。  
- 人类之间的互动也是连续的，这可以从微博上的争吵和辩论中看出。

# 8.1.1 统计工具

处理序列数据需要统计工具和新的深度神经网络架构。为了简单起见，我们以图8.1.1所示的股票价格（富时100指数）为例。

![](images/1048bc7d35b6f0d6b223029b182aaa3df71a329340767fc67d970554c2cb5f4c.jpg)  
图8.1.1: 近30年的富时100指数

其中，用  $x_{t}$  表示价格，即在时间步（time step） $t \in \mathbb{Z}^{+}$ 时，观察到的价格  $x_{t}$ 。请注意， $t$  对于本文中的序列通常是离散的，并在整数或其子集上变化。假设一个交易员想在  $t$  日的股市中表现良好，于是通过以下途径预测  $x_{t}$ ：

$$
x _ {t} \sim P \left(x _ {t} \mid x _ {t - 1}, \dots , x _ {1}\right). \tag {8.1.1}
$$

# 自回归模型

为了实现这个预测，交易员可以使用回归模型，例如在3.3节中训练的模型。仅有一个主要问题：输入数据的数量，输入  $x_{t-1}, \ldots, x_1$  本身因  $t$  而异。也就是说，输入数据的数量这个数字将会随着我们遇到的数据量的增加而增加，因此需要一个近似方法来使这个计算变得容易处理。本章后面的大部分内容将围绕着如何有效估计  $P(x_t \mid x_{t-1}, \ldots, x_1)$  展开。简单地说，它归结为以下两种策略。

第一种策略，假设在现实情况下相当长的序列  $x_{t-1}, \ldots, x_1$  可能是不必要的，因此我们只需要满足某个长度为  $\tau$  的时间跨度，即使用观测序列  $x_{t-1}, \ldots, x_{t-\tau}$  。当下获得的最直接的好处就是参数的数量总是不变的，至少在  $t > \tau$  时如此，这就使我们能够训练一个上面提及的深度网络。这种模型被称为自回归模型（autoregressive models），因为它们是对自己执行回归。

第二种策略，如图8.1.2所示，是保留一些对过去观测的总结  $h_t$ ，并且同时更新预测  $\hat{x}_t$  和总结  $h_t$  。这就产生了基于  $\hat{x}_t = P(x_t \mid h_t)$  估计  $x_t$ ，以及公式  $h_t = g(h_{t-1}, x_{t-1})$  更新的模型。由于  $h_t$  从未被观测到，这类模型也被称为隐变量自回归模型（latent autoregressive models）。

![](images/da066718fe9adba6f2b7de8c8f3fe758dd83aae0c7436c07cb1d86390954803b.jpg)  
图8.1.2: 隐变量自回归模型

# 8.1. 序列模型

这两种情况都有一个显而易见的问题：如何生成训练数据？一个经典方法是使用历史观测来预测下一个未来观测。显然，我们并不指望时间会停滞不前。然而，一个常见的假设是虽然特定值  $x_{t}$  可能会改变，但是序列本身的动力学不会改变。这样的假设是合理的，因为新的动力学一定受新的数据影响，而我们不可能用目前所掌握的数据来预测新的动力学。统计学家称不变的动力学为静止的（stationary）。因此，整个序列的估计值都将通过以下的方式获得：

$$
P \left(x _ {1}, \dots , x _ {T}\right) = \prod_ {t = 1} ^ {T} P \left(x _ {t} \mid x _ {t - 1}, \dots , x _ {1}\right). \tag {8.1.2}
$$

注意, 如果我们处理的是离散的对象 (如单词), 而不是连续的数字, 则上述的考虑仍然有效。唯一的差别是,对于离散的对象, 我们需要使用分类器而不是回归模型来估计  $P\left(x_{t} \mid x_{t-1}, \ldots, x_{1}\right)$  。

# 马尔可夫模型

回想一下，在自回归模型的近似法中，我们使用  $x_{t-1}, \ldots, x_{t-\tau}$  而不是  $x_{t-1}, \ldots, x_1$  来估计  $x_t$ 。只要这种是近似精确的，我们就说序列满足马尔可夫条件（Markov condition）。特别是，如果  $\tau = 1$ ，得到一个一阶马尔可夫模型（first-order Markov model）， $P(x)$  由下式给出：

$$
P (x _ {1}, \ldots , x _ {T}) = \prod_ {t = 1} ^ {T} P (x _ {t} \mid x _ {t - 1}) \text {当} P (x _ {1} \mid x _ {0}) = P (x _ {1}). \tag {8.1.3}
$$

当假设  $x_{t}$  仅是离散值时，这样的模型特别棒，因为在这种情况下，使用动态规划可以沿着马尔可夫链精确地计算结果。例如，我们可以高效地计算  $P(x_{t + 1} \mid x_{t - 1})$ ：

$$
\begin{array}{l} P (x _ {t + 1} \mid x _ {t - 1}) = \frac {\sum_ {x _ {t}} P (x _ {t + 1} , x _ {t} , x _ {t - 1})}{P (x _ {t - 1})} \\ = \frac {\sum_ {x _ {t}} P \left(x _ {t + 1} \mid x _ {t} , x _ {t - 1}\right) P \left(x _ {t} , x _ {t - 1}\right)}{P \left(x _ {t - 1}\right)} \tag {8.1.4} \\ = \sum_ {x _ {t}} P (x _ {t + 1} \mid x _ {t}) P (x _ {t} \mid x _ {t - 1}) \\ \end{array}
$$

利用这一事实，我们只需要考虑过去观察中的一个非常短的历史： $P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$ 。隐马尔可夫模型中的动态规划超出了本节的范围（我们将在 9.4 节再次遇到），而动态规划这些计算工具已经在控制算法和强化学习算法广泛使用。

# 因果关系

原则上，将  $P(x_{1},\ldots ,x_{T})$  倒序展开也没什么问题。毕竟，基于条件概率公式，我们总是可以写出：

$$
P \left(x _ {1}, \dots , x _ {T}\right) = \prod_ {t = T} ^ {1} P \left(x _ {t} \mid x _ {t + 1}, \dots , x _ {T}\right). \tag {8.1.5}
$$

事实上，如果基于一个马尔可夫模型，我们还可以得到一个反向的条件概率分布。然而，在许多情况下，数据存在一个自然的方向，即在时间上是前进的。很明显，未来的事件不能影响过去。因此，如果我们改变  $x_{t}$ ，可能会影响未来发生的事情  $x_{t + 1}$ ，但不能反过来。也就是说，如果我们改变  $x_{t}$ ，基于过去事件得到的分布不会改变。因此，解释  $P(x_{t + 1} \mid x_t)$  应该比解释  $P(x_{t} \mid x_{t + 1})$  更容易。例如，在某些情况下，对于某些可加性噪

声  $\epsilon$  ，显然我们可以找到  $x_{t + 1} = f(x_t) + \epsilon$  ，而反之则不行(Hoyer et al.,2009)。而这个向前推进的方向恰好也是我们通常感兴趣的方向。彼得斯等人(Peters et al.,2017)对该主题的更多内容做了详尽的解释，而我们的上述讨论只是其中的冰山一角。

# 8.1.2 训练

在了解了上述统计工具后，让我们在实践中尝试一下！首先，我们生成一些数据：使用正弦函数和一些可加性噪声来生成序列数据，时间步为1,2,...,1000。

```python
%matplotlib inline  
import torch  
from torch import nn  
from d2l import torch as d2l
```

```txt
T = 1000 # 总共产生1000个点  
time = torch.arange(1, T + 1, dtype=torch.float32)  
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  
d21.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

![](images/b9ff71a89e47213e592cc56fde1784d87389d39fb012182efc4e064388efe8fb.jpg)

接下来，我们将这个序列转换为模型的特征一标签（feature-label）对。基于嵌入维度  $\tau$  ，我们将数据映射为数据对  $y_{t} = x_{t}$  和  $\mathbf{x}_t = [x_{t - \tau},\dots ,x_{t - 1}]$  。这比我们提供的数据样本少了  $\tau$  个，因为我们没有足够的历史记录来描述前  $\tau$  个数据样本。一个简单的解决办法是：如果拥有足够长的序列就丢弃这几项；另一个方法是用零填充序列。在这里，我们仅使用前600个“特征一标签”对进行训练。

```python
tau = 4  
features = torch.zeros((T - tau, tau))  
for i in range(tau):  
    features[:, i] = x[i: T - tau + i]  
labels = x[tau].reshape((-1, 1))
```

# 8.1. 序列模型

```python
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
```

在这里，我们使用一个相当简单的架构训练模型：一个拥有两个全连接层的多层感知机，ReLU激活函数和平方损失。

初始化网络权重的函数  
```python
def initweights(m):
    if type(m) == nn.Linear:
        nn.init.xavier.uniform_(m.weight)
```

一个简单的多层感知机  
平方损失。注意：MSELoss计算平方误差时不带系数1/2 loss = nn.MSELoss(reduction='none')  
```python
def get_net(): net  $=$  nnSequential(nn.Linear(4,10), nn.ReLU(), nn.Linear(10,1)) net.apply(initweights) return net
```

现在，准备训练模型了。实现下面的训练代码的方式与前面几节（如3.3节）中的循环训练基本相同。因此，我们不会深入探讨太多细节。

epoch 1, loss: 0.076846  
```python
def train(net,train_iter,loss,epochs，lr): trainer  $=$  torch.optim.Adam(net.params(),lr) for epoch in range(epoos): for X,y in train_iter: trainer.zero_grad() 1  $=$  loss(net(X)，y) 1-sum().backward() trainer step() print(f'epoch{epoch+1},f' f'loss:{d2l.evaluate_loss(net,train_iter，loss):f}') net  $=$  get_net()   
train(net,train_iter，loss，5，0.01)
```

(continues on next page)

epoch 2, loss: 0.056340

epoch 3, loss: 0.053779

epoch 4, loss: 0.056320

epoch 5, loss: 0.051650

# 8.1.3 预测

由于训练损失很小，因此我们期望模型能有很好的工作效果。让我们看看这在实践中意味着什么。首先是检查模型预测下一个时间步的能力，也就是单步预测（one-step-ahead prediction）。

```python
onestep_preds = net features
d21.plot([time, time[tau]])
[xdetach().numpy(), onestep_predsdetach().numpy(), 'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
```

![](images/11361c578c5e7843cad6025f53aff3b23356efa7e500db3a2e4736411d14c0d1.jpg)

正如我们所料，单步预测效果不错。即使这些预测的时间步超过了  $600 + 4$ （n_train + tau），其结果看起来仍然是可信的。然而有一个小问题：如果数据观察序列的时间步只到604，我们需要一步一步地向前迈进：

$$
\hat {x} _ {6 0 5} = f (x _ {6 0 1}, x _ {6 0 2}, x _ {6 0 3}, x _ {6 0 4}),
$$

$$
\hat {x} _ {6 0 6} = f \left(x _ {6 0 2}, x _ {6 0 3}, x _ {6 0 4}, \hat {x} _ {6 0 5}\right),
$$

$$
\hat {x} _ {6 0 7} = f \left(x _ {6 0 3}, x _ {6 0 4}, \hat {x} _ {6 0 5}, \hat {x} _ {6 0 6}\right), \tag {8.1.6}
$$

$$
\hat {x} _ {6 0 8} = f \left(x _ {6 0 4}, \hat {x} _ {6 0 5}, \hat {x} _ {6 0 6}, \hat {x} _ {6 0 7}\right),
$$

$$
\hat {x} _ {6 0 9} = f \left(\hat {x} _ {6 0 5}, \hat {x} _ {6 0 6}, \hat {x} _ {6 0 7}, \hat {x} _ {6 0 8}\right),
$$

.

通常，对于直到  $x_{t}$  的观测序列，其在时间步  $t + k$  处的预测输出  $\hat{x}_{t + k}$  称为  $k$  步预测（ $k$ -step-ahead-prediction）。由于我们的观察已经到了  $x_{604}$ ，它的  $k$  步预测是  $\hat{x}_{604 + k}$ 。换句话说，我们必须使用我们自己的预测（而不是原始数据）来进行多步预测。让我们看看效果如何。

# 8.1. 序列模型

```python
multistep_preds = torch.zeros(T)  
multistep_preds[: n_train + tau] = x[: n_train + tau]  
for i in range(n_train + tau, T):  
    multistep_preds[i] = net(  
        multistep_preds[i - tau:i].reshape((1, -1)))
```

```python
d21.plot([time, time[tau], time[n_train + tau:]], [xdetach().numpy(), onestep_predsdetach().numpy(), multistep_preds[n_train + tau].detach().numpy}], 'time', 'x', legend=['data', '1-step predicts', 'multistep predicts'], xlim=[1, 1000], figsize=(6, 3))
```

![](images/3de38aa6f0ab1efae32990bcd2abcfadd925c42a1ec10a2f03c36bd8c3a51f2a.jpg)

如上面的例子所示，绿线的预测显然并不理想。经过几个预测步骤之后，预测的结果很快就会衰减到一个常数。为什么这个算法效果这么差呢？事实是由于错误的累积：假设在步骤1之后，我们积累了一些错误  $\epsilon_{1} = \bar{\epsilon}_{0}$  。于是，步骤2的输入被扰动了  $\epsilon_{1}$ ，结果积累的误差是依照次序的  $\epsilon_{2} = \bar{\epsilon} + c\epsilon_{1}$ ，其中  $c$  为某个常数，后面的预测误差依此类推。因此误差可能会相当快地偏离真实的观测结果。例如，未来24小时的天气预报往往相当准确，但超过这一点，精度就会迅速下降。我们将在本章及后续章节中讨论如何改进这一点。

基于  $k = 1,4,16,64$  ，通过对整个序列预测的计算，让我们更仔细地看一下  $k$  步预测的困难。

```python
max_steps = 64
```

```python
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps)) # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1) for i in range(tau): features[:，i] = x[i: i + T - tau - max_steps + 1]
```

```txt
列i（  $\mathrm{i} >= \mathrm{tau}$  ）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）for i in range(tau, tau + max_steps):
```

(continues on next page)

```python
features[:, i] = net(features[:, i - tau:i]).reshape(-1)
```

```txt
steps  $=$  (1,4,16,64)   
d2l.plot([time[tau  $^+$  i-1:T-max_steps  $^+$  i]fori in steps], [features:,（tau+i-1)].detach().numpy()fori in steps],‘time'，'x', legend  $\coloneqq$  [f'\{i]-steppreds'fori in steps],xlim=[5,1000], figsize=(6,3))
```

![](images/a7466fa843de0d7aa20de003f42212fbc8dbd8aa114717c35af171ee8926f807.jpg)

以上例子清楚地说明了当我们试图预测更远的未来时，预测的质量是如何变化的。虽然“4步预测”看起来仍然不错，但超过这个跨度的任何预测几乎都是无用的。

# 小结

- 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进行训练。  
- 序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型。  
- 对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。  
- 对于直到时间步  $t$  的观测序列，其在时间步  $t + k$  的预测输出是“ $k$  步预测”。随着我们对预测时间  $k$  值的增加，会造成误差的快速累积和预测质量的极速下降。

# 练习

1. 改进本节实验中的模型

1. 是否包含了过去4个以上的观测结果？真实值需要是多少个？  
2. 如果没有噪音，需要多少个过去的观测结果？提示：把sin和cos写成微分方程。  
3. 可以在保持特征总数不变的情况下合并旧的观察结果吗？这能提高正确度吗？为什么？  
4. 改变神经网络架构并评估其性能。

2. 一位投资者想要找到一种好的证券来购买。他查看过去的回报，以决定哪一种可能是表现良好的。这一策略可能会出什么问题呢？  
3. 时间是向前推进的因果模型在多大程度上适用于文本呢？  
4. 举例说明什么时候可能需要隐变量自回归模型来捕捉数据的动力学模型。

Discussions<sup>98</sup>

# 8.2 文本预处理

对于序列数据处理问题，我们在8.1节中评估了所需的统计工具和预测时面临的挑战。这样的数据存在许多种形式，文本是最常见例子之一。例如，一篇文章可以被简单地看作一串单词序列，甚至是一串字符序列。本节中，我们将解析文本的常见预处理步骤。这些步骤通常包括：

1. 将文本作为字符串加载到内存中。  
2. 将字符串拆分为词元（如单词和字符）。  
3. 建立一个词表，将拆分的词元映射到数字索引。  
4. 将文本转换为数字索引序列，方便模型操作。

```python
import collections  
import re  
from d21 import torch as d21
```

# 8.2.1 读取数据集

首先，我们从H.G.Well的时光机器99中加载文本。这是一个相当小的语料库，只有30000多个单词，但足够我们小试牛刀，而现实中的文档集合可能会包含数十亿个单词。下面的函数将数据集读取到由多条文本行组成的列表中，其中每条文本行都是一个字符串。为简单起见，我们在这里忽略了标点符号和字母大写。

```python
@save
d21.DATA_HUB['time_MACHINE'] = (d21.DATA_URL + 'timemachine.txt',
'090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_MACHINE(): # save
    '''将时间机器数据集加载到文本行的列表中''''
    with open(d21.download('time_MACHINE'), 'r') as f:
        lines = f.readlines()
        return [re.sub(['^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    lines = read_time_MACHINE()
print(f'# 文本总行数: {len lines}')'
printlines[0])
printlines[10])
```

```txt
Downloading ../data/timemachine.txt from http://d21-data.s3-accelerate.amazon.com/timemachine.txt...  
# 文本总行数：3221  
the time machine by h g wells  
twinkled and his usually pale face was flushed and animated the
```

# 8.2.2 词元化

下面的tokenizer函数将文本行列表（lines）作为输入，列表中的每个元素是一个文本序列（如一条文本行）。每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。

```python
def tokenize lines, token='word'): #save
    '''将文本行拆分为单词或字符词元''
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
```

(continues on next page)

```txt
tokens  $=$  tokenizerlines)   
for i in range(11): print(tokens[i])
```

```txt
['the', 'time', 'machine', 'by', 'h', 'g', 'wells']  
[]  
[]  
[]  
[]  
['i']  
[]  
[]  
['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']  
['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']  
['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
```

# 8.2.3 词表

词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。现在，让我们构建一个字典，通常也叫做词表（vocabulary），用来将字符串类型的词元映射到从0开始的数字索引中。我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计，得到的统计结果称之为语料（corpus）。然后根据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元通常被移除，这可以降低复杂性。另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。我们可以选择增加一个列表，用于保存那些被保留的词元，例如：填充词元（“<pad>”）；序列开始词元（“<bos>”）；序列结束词元（“<eos>”）。

```python
class Vocab: @save
    """文本词表"
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count Corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>] + reserved_tokens
```

(continues on next page)

(continued from previous page)

```python
self_token_toidx = {token: idx
    for idx, token in enumerate(self.idx_to_token)}
    for token, freq in self._token_freqs:
        if freq < min_freq:
            break
            if token not in self_token_toidx:
                self.idx_to_token.append(token)
                self_token_toidx(token] = len(self.idx_to_token) - 1
def __len__(self):
    return len(self.idx_to_token)
def __getitem__(self, tokens):
    if not isinstance(tokens, (list, tuple)):
        return self_token_toidx.get(tokens, self.unk)
        return [self._getitem__(token) for token in tokens]
def to_tokens(self, indices):
    if not isinstance(indices, (list, tuple)):
        return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
@property
def unk(self): # 未知词元的索引为0
    return 0
@property
def token_freqs(self):
    return self._token_freqs
def count Corpus(tokens): # @save
    ""统计词元的频率''
# 这里的tokens是1D列表或2D列表
if len(tokens) == 0 or isinstance(tokens[0], list):
    # 将词元列表展平成一个列表
    tokens = [token for line in tokens for token in line]
return collections.Counter(tokens)
```

我们首先使用时光机器数据集作为语料库来构建词表，然后打印前几个高频词元及其索引。

```lua
vocab = VOCab(tokens)
print(list(vocab_token_toidx.items()[10])
```

```latex
[ \left[ \left( ^{\prime} < \text{unk} >, 0 \right), \left( ^{\prime} \text{the} ^{\prime}, 1 \right), \left( ^{\prime} i ^{\prime}, 2 \right), \left( ^{\prime} \text{and} ^{\prime}, 3 \right), \left( ^{\prime} o f ^{\prime}, 4 \right), \left( ^{\prime} a ^{\prime}, 5 \right), \left( ^{\prime} t o ^{\prime}, 6 \right), \left( ^{\prime} w a s ^{\prime}, 7 \right), \left( ^{\prime} i n ^{\prime}, 8 \right), \right. ]
```

现在，我们可以将每一条文本行转换成一个数字索引列表。

```python
for i in [0, 10]:  
    print('文本：', tokens[i])  
    print('索引：', vocab(tokens[i]))
```

```txt
文本：['the'，'time'，'machine'，'by'，'h'，'g'，'wells']  
索引：[1，19，50，40，2183，2184，400]  
文本：['twinkled'，'and'，'his'，'usually'，'pale'，'face'，'was'，'flushed'，'and'，'animated'，'the']  
索引：[2186，3，25，1044，362，113，7，1421，3，1045，1]
```

# 8.2.4 整合所有功能

在使用上述函数时，我们将所有功能打包到load(corpus_time MACHINE函数中，该函数返回corpus（词元素引列表）和vocab（时光机器语料库的词表)。我们在这里所做的改变是：

1. 为了简化后面章节中的训练，我们使用字符（而不是单词）实现文本词元化；  
2. 时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多词元列表构成的一个列表。

```python
def load Corpus_time-machine(max_tokens=-1): #@save
    return lines = read_time-machine()
    tokens = tokenizelines,'char')
    vocab = Vocabulary(tokens)
    #因为时光机器数据集中的每个文本行不一定是一个句子或一个段落,
        #所以将所有文本行展平到一个列表中
        corpus = [vocab(token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab
    corpus, vocab = load Corpus_time-machine()
    len(corpus), len(vocab)
```

```txt
(170580, 28)
```

# 小结

- 文本是序列数据的一种最常见的形式之一。  
- 为了对文本进行预处理，我们通常将文本拆分为词元，构建词表将词元字符串映射为数字索引，并将文本数据转换为词元素引以供模型操作。

# 练习

1. 词元化是一个关键的预处理步骤，它因语言而异。尝试找到另外三种常用的词元化文本的方法。  
2. 在本节的实验中，将文本词元为单词和更改Vocab实例的min_freq参数。这对词表大小有何影响？  
Discussions<sup>100</sup>

# 8.3 语言模型和数据集

在 8.2 节中, 我们了解了如何将文本数据映射为词元, 以及将这些词元可以视为一系列离散的观测, 例如单词或字符。假设长度为  $T$  的文本序列中的词元依次为  $x_{1}, x_{2}, \ldots, x_{T}$  。于是,  $x_{t} (1 \leq t \leq T)$  可以被认为是文本序列在时间步  $t$  处的观测或标签。在给定这样的文本序列时, 语言模型 (language model) 的目标是估计序列的联合概率

$$
P \left(x _ {1}, x _ {2}, \dots , x _ {T}\right). \tag {8.3.1}
$$

例如，只需要一次抽取一个词元  $x_{t} \sim P(x_{t} \mid x_{t-1}, \ldots, x_{1})$ ，一个理想的语言模型就能够基于模型本身生成自然文本。与猴子使用打字机完全不同的是，从这样的模型中提取的文本都将作为自然语言（例如，英语文本）来传递。只需要基于前面的对话片断中的文本，就足以生成一个有意义的对话。显然，我们离设计出这样的系统还很遥远，因为它需要“理解”文本，而不仅仅是生成语法合理的内容。

尽管如此，语言模型依然是非常有用的。例如，短语“to recognize speech”和“to wreck a nice beach”读音上听起来非常相似。这种相似性会导致语音识别中的歧义，但是这很容易通过语言模型来解决，因为第二句的语义很奇怪。同样，在文档摘要生成算法中，“狗咬人”比“人咬狗”出现的频率要高得多，或者“我想吃奶奶”是一个相当匪夷所思的语句，而“我想吃，奶奶”则要正常得多。

# 8.3.1 学习语言模型

显而易见，我们面对的问题是如何对一个文档，甚至是一个词元序列进行建模。假设在单词级别对文本数据进行词元化，我们可以依靠在8.1节中对序列模型的分析。让我们从基本概率规则开始：

$$
P \left(x _ {1}, x _ {2}, \dots , x _ {T}\right) = \prod_ {t = 1} ^ {T} P \left(x _ {t} \mid x _ {1}, \dots , x _ {t - 1}\right). \tag {8.3.2}
$$

例如，包含了四个单词的一个文本序列的概率是：

$$
P (\text {d e e p}, \text {l e a n i n g}, \text {i s}, \text {f u n}) = P (\text {d e e p}) P (\text {l e a n i n g} \mid \text {d e e p}) P (\text {i s} \mid \text {d e e p}, \text {l e a n i n g}) P (\text {f u n} \mid \text {d e e p}, \text {l e a n i n g}, \text {i s}).
$$

为了训练语言模型，我们需要计算单词的概率，以及给定前面几个单词后出现某个单词的条件概率。这些概率本质上就是语言模型的参数。

这里，我们假设训练数据集是一个大型的文本语料库。比如，维基百科的所有条目、古登堡计划101，或者所有发布在网络上的文本。训练数据集中词的概率可以根据给定词的相对词频来计算。例如，可以将估计值  $\hat{P}(\text{deep})$  计算为任何以单词“deep”开头的句子的概率。一种（稍稍不太精确的）方法是统计单词“deep”在数据集中的出现次数，然后将其除以整个语料库中的单词总数。这种方法效果不错，特别是对于频繁出现的单词。接下来，我们可以尝试估计

$$
\hat {P} (\text {l e a n i n g} \mid \text {d e e p}) = \frac {n (\text {d e e p} , \text {l e a n i n g})}{n (\text {d e e p})}, \tag {8.3.4}
$$

其中  $n(x)$  和  $n(x, x')$  分别是单个单词和连续单词对的出现次数。不幸的是，由于连续单词对“deep learning”的出现频率要低得多，所以估计这类单词正确的概率要困难得多。特别是对于一些不常见的单词组合，要想找到足够的出现次数来获得准确的估计可能都不容易。而对于三个或者更多的单词组合，情况会变得更糟。许多合理的三个单词组合可能是存在的，但是在数据集中却找不到。除非我们提供某种解决方案，来将这些单词组合指定为非零计数，否则将无法在语言模型中使用它们。如果数据集很小，或者单词非常罕见，那么这类单词出现一次的机会可能都找不到。

一种常见的策略是执行某种形式的拉普拉斯平滑（Laplace smoothing），具体方法是在所有计数中添加一个小常量。用  $n$  表示训练集中的单词总数，用  $m$  表示唯一单词的数量。此解决方案有助于处理单元素问题，例如通过：

$$
\hat {P} (x) = \frac {n (x) + \epsilon_ {1} / m}{n + \epsilon_ {1}},
$$

$$
\hat {P} \left(x ^ {\prime} \mid x\right) = \frac {n \left(x , x ^ {\prime}\right) + \epsilon_ {2} \hat {P} \left(x ^ {\prime}\right)}{n (x) + \epsilon_ {2}}, \tag {8.3.5}
$$

$$
\hat {P} (x ^ {\prime \prime} \mid x, x ^ {\prime}) = \frac {n (x , x ^ {\prime} , x ^ {\prime \prime}) + \epsilon_ {3} \hat {P} (x ^ {\prime \prime})}{n (x , x ^ {\prime}) + \epsilon_ {3}}.
$$

其中， $\epsilon_1, \epsilon_2$  和  $\epsilon_3$  是超参数。以  $\epsilon_1$  为例：当  $\epsilon_1 = 0$  时，不应用平滑；当  $\epsilon_1$  接近正无穷大时， $\hat{P}(x)$  接近均匀概率分布  $1/m$ 。上面的公式是 (Wood et al., 2011) 的一个相当原始的变形。

然而，这样的模型很容易变得无效，原因如下：首先，我们需要存储所有的计数；其次，这完全忽略了单词的意思。例如，“猫”（cat）和“猫科动物”（feline）可能出现在相关的上下文中，但是想根据上下文调整这类模型其实是相当困难的。最后，长单词序列大部分是没出现过的，因此一个模型如果只是简单地统计先前“看到”的单词序列频率，那么模型面对这种问题肯定是表现不佳的。

# 8.3.2 马尔可夫模型与  $n$  元语法

在讨论包含深度学习的解决方案之前，我们需要了解更多的概念和术语。回想一下我们在 8.1 节中对马尔可夫模型的讨论，并且将其应用于语言建模。如果  $P(x_{t+1} \mid x_t, \dots, x_1) = P(x_{t+1} \mid x_t)$ ，则序列上的分布满足一阶马尔可夫性质。阶数越高，对应的依赖关系就越长。这种性质推导出了许多可以应用于序列建模的近似公式：

$$
P \left(x _ {1}, x _ {2}, x _ {3}, x _ {4}\right) = P \left(x _ {1}\right) P \left(x _ {2}\right) P \left(x _ {3}\right) P \left(x _ {4}\right),
$$

$$
P \left(x _ {1}, x _ {2}, x _ {3}, x _ {4}\right) = P \left(x _ {1}\right) P \left(x _ {2} \mid x _ {1}\right) P \left(x _ {3} \mid x _ {2}\right) P \left(x _ {4} \mid x _ {3}\right), \tag {8.3.6}
$$

$$
P (x _ {1}, x _ {2}, x _ {3}, x _ {4}) = P (x _ {1}) P (x _ {2} \mid x _ {1}) P (x _ {3} \mid x _ {1}, x _ {2}) P (x _ {4} \mid x _ {2}, x _ {3}).
$$

通常，涉及一个、两个和三个变量的概率公式分别被称为一元语法（unigram）、二元语法（bigram）和三元语法（trigram）模型。下面，我们将学习如何去设计更好的模型。

# 8.3.3 自然语言统计

我们看看在真实数据上如果进行自然语言统计。根据8.2节中介绍的时光机器数据集构建词表，并打印前10个最常用的（频率最高的）单词。

```python
import random   
import torch   
from d21 import torch as d21
```

```txt
tokens = d21Tokenizer(d21.read_time-machine())
#因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d21.Vocab(corpus)
vocab_token_freqs[:10]
```

```txt
[('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)]
```

正如我们所看到的，最流行的词看起来很无聊，这些词通常被称为停用词（stop words），因此可以被过滤掉。尽管如此，它们本身仍然是有意义的，我们仍然会在模型中使用它们。此外，还有个明显的问题是词频衰减

的速度相当地快。例如，最常用单词的词频对比，第10个还不到第1个的  $1 / 5$  。为了更好地理解，我们可以画出的词频图：

```python
freqs = [freq for token, freq in vocab_token_freqs]  
d21.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)'), xscale='log', yscale='log')
```

![](images/1d4616543e80d608075d2949d698e977ea89b624a7344df3ae3938fc13805d5c.jpg)

通过此图我们可以发现：词频以一种明确的方式迅速衰减。将前几个单词作为例外消除后，剩余的所有单词大致遵循双对数坐标图上的一条直线。这意味着单词的频率满足齐普夫定律（Zipf’slaw），即第  $i$  个最常用单词的频率  $n_{i}$  为：

$$
n _ {i} \propto \frac {1}{i ^ {\alpha}}, \tag {8.3.7}
$$

等价于

$$
\log n _ {i} = - \alpha \log i + c, \tag {8.3.8}
$$

其中  $\alpha$  是刻画分布的指数， $c$  是常数。这告诉我们想要通过计数统计和平滑来建模单词是不可行的，因为这样建模的结果会大大高估尾部单词的频率，也就是所谓的不常用单词。那么其他的词元组合，比如二元语法、三元语法等等，又会如何呢？我们来看看二元语法的频率是否与一元语法的频率表现出相同的行为方式。

```python
bigram_tokens = [pair for pair in zip(corpus[: -1], corpus[1:]])
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab_token_freqs[:10]
```

```txt
[('of', 'the'), 309),  
((in', 'the'), 169),  
((i', 'had'), 130),  
((i', 'was'), 112),  
((and', 'the'), 109),  
((the', 'time'), 102),  
((it', 'was'), 99),
```

(continues on next page)

```javascript
('to', 'the'), 85),  
('as', 'i'), 78),  
('of', 'a'), 73)]
```

这里值得注意：在十个最频繁的词对中，有九个是由两个停用词组成的，只有一个与“the time”有关。我们再进一步看看三元语法的频率是否表现出相同的行为方式。

```python
trigram_tokens = [triple for triple in zip(
    corpus[:2], corpus[1:-1], corpus[2:])
]  
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab_token_freqs[:10]
```

```javascript
[('the', 'time', 'traveller'), 59), ('the', 'time', 'machine'), 30), ('the', 'medical', 'man'), 24), ('it', 'seemed', 'to'), 16), ('it', 'was', 'a'), 15), ('here', 'and', 'there'), 15), ('seemed', 'to', 'me'), 14), ('i', 'did', 'not'), 14), ('i', 'saw', 'the'), 13), ('i', 'began', 'to'), 13)]
```

最后，我们直观地对比三种模型中的词元频率：一元语法、二元语法和三元语法。

```python
bigram_freqs = [freq for token, freq in bigram_vocab_token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab_token_freqs]
d21.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: n(x) ', xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])
```

![](images/aa9f4462d9b0c926386dae846a432f7d6b20d3460d4a3eda0846c497279c0c7e.jpg)

这张图非常令人振奋！原因有很多：

1. 除了一元语法词, 单词序列似乎也遵循齐普夫定律, 尽管公式 (8.3.7) 中的指数  $\alpha$  更小 (指数的大小受序列长度的影响);  
2. 词表中  $n$  元组的数量并没有那么大, 这说明语言中存在相当多的结构, 这些结构给了我们应用模型的希望;  
3. 很多  $n$  元组很少出现，这使得拉普拉斯平滑非常不适合语言建模。作为代替，我们将使用基于深度学习的模型。

# 8.3.4 读取长序列数据

由于序列数据本质上是连续的，因此我们在处理数据时需要解决这个问题。在 8.1 节中我们以一种相当特别的方式做到了这一点：当序列变得太长而不能被模型一次性全部处理时，我们可能希望拆分这样的序列方便模型读取。

在介绍该模型之前，我们看一下总体策略。假设我们将使用神经网络来训练语言模型，模型中的网络一次处理具有预定义长度（例如  $n$  个时间步）的一个小批量序列。现在的问题是如何随机生成一个小批量数据的特征和标签以供读取。

首先，由于文本序列可以是任意长的，例如整本《时光机器》（The Time Machine），于是任意长的序列可以被我们划分为具有相同时间步数的子序列。当训练我们的神经网络时，这样的小批量子序列将被输入到模型中。假设网络一次只处理具有  $n$  个时间步的子序列。图8.3.1画出了从原始文本序列获得子序列的所有不同的方式，其中  $n = 5$ ，并且每个时间步的词元对应于一个字符。请注意，因为我们可以选择任意偏移量来指示初始位置，所以我们有相当大的自由度。

![](images/8c168ae3557eb933701d8c78d713e4015dc2a1ffc1928a0cd0226c35012b3e0f.jpg)  
图8.3.1: 分割文本时, 不同的偏移量会导致不同的子序列

因此，我们应该从图8.3.1中选择哪一个呢？事实上，他们都一样的好。然而，如果我们只选择一个偏移量，那么用于训练网络的、所有可能的子序列的覆盖范围将是有限的。因此，我们可以从随机偏移量开始划分序列，以同时获得覆盖性（coverage）和随机性（randomness）。下面，我们将描述如何实现随机采样（random sampling）和顺序分区（sequential partitioning）策略。

# 随机采样

在随机采样中，每个样本都是在原始的长序列上任意捕获的子序列。在迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻。对于语言建模，目标是基于到目前为止我们看到的词元来预测下一个词元，因此标签是移位了一个词元的原始序列。

下面的代码每次可以从数据中随机生成一个小批量。在这里，参数batch_size指定了每个小批量中子序列样本的数目，参数num_steps是每个子序列中预定义的时间步数。

```python
def seq_data_iter_random(corpus, batch_size, num_steps): #save
    """使用随机抽样生成一个小批量子序列"
#从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
corpus = corpus[rand.randint(0, num_steps - 1)];
#减去1，是因为我们需要考虑标签
num_subseqs = (len(corpus) - 1) // num_steps
#长度为num_steps的子序列的起始索引
initial Indices = list(range(0, num subsequs * num_steps, num_steps))
#在随机抽样的迭代过程中，
#来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
randomshuffle(initialIndices)
def data(pos):
    #返回从pos位置开始的长度为num_steps的序列
    return corpus[pos: pos + num_steps]
num_batches = num_subseqs // batch_size
for i in range(0, batch_size * num_batches, batch_size):
    #在这里，initial Indices包含子序列的随机起始索引
    initial Indices_per_batch = initial Indices[i: i + batch_size]
X = [data(j) for j in initial Indices_per_batch]
Y = [data(j + 1) for j in initial Indices_per_batch]
yield torch.tensor(X), torch.tensor(Y)
```

下面我们生成一个从0到34的序列。假设批量大小为2，时间步数为5，这意味着可以生成  $\lfloor (35 - 1) / 5\rfloor = 6$  个“特征一标签”子序列对。如果设置小批量大小为2，我们只能得到3个小批量。

```python
my_seq = list(range(35))  
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):  
    print('X: ', X, '\nY: ', Y)
```

```javascript
X: tensor([[13, 14, 15, 16, 17], [28, 29, 30, 31, 32]])  
Y: tensor([[14, 15, 16, 17, 18],
```

(continues on next page)

# 8.3. 语言模型和数据集

```txt
[29, 30, 31, 32, 33]]  
X: tensor([[3, 4, 5, 6, 7], [18, 19, 20, 21, 22]])  
Y: tensor([[4, 5, 6, 7, 8], [19, 20, 21, 22, 23]])
X: tensor([[8, 9, 10, 11, 12], [23, 24, 25, 26, 27]])
Y: tensor([[9, 10, 11, 12, 13], [24, 25, 26, 27, 28]])
```

# 顺序分区

在迭代过程中，除了对原始序列可以随机抽样外，我们还可以保证两个相邻的小批量中的子序列在原始序列上也是相邻的。这种策略在基于小批量的迭代过程中保留了拆分的子序列的顺序，因此称为顺序分区。

```python
def seq_data_itersequential(corpus, batch_size, num_steps): #save
	"""使用顺序分区生成一个小批量子序列"
	#从随机偏移量开始划分序列
 offset = random.randint(0, num_steps)
 num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
 Xs = torch.tensor(corpus[offset: offset + num_tokens])
 Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
 Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
 num_batches = Xs.shape[1] // num_steps
 for i in range(0, num_steps * num_batches, num_steps):
 X = Xs[:, i: i + num_steps]
 Y = Ys[:, i: i + num_steps]
 yield X, Y
```

基于相同的设置，通过顺序分区读取每个小批量的子序列的特征X和标签Y。通过将它们打印出来可以发现：迭代期间来自两个相邻的小批量中的子序列在原始序列中确实是相邻的。

```python
for X, Y in seq_data_itersequential(my_seq, batch_size=2, num_steps=5): print('X: ', X, '\nY: ', Y)
```

```txt
X: tensor([[0, 1, 2, 3, 4], [17, 18, 19, 20, 21]])  
Y: tensor([[1, 2, 3, 4, 5], [18, 19, 20, 21, 22]])  
X: tensor([[5, 6, 7, 8, 9], [22, 23, 24, 25, 26]])
```

(continues on next page)

```javascript
Y: tensor([[6, 7, 8, 9, 10], [23, 24, 25, 26, 27]])  
X: tensor([[10, 11, 12, 13, 14], [27, 28, 29, 30, 31]])  
Y: tensor([[11, 12, 13, 14, 15], [28, 29, 30, 31, 32]])
```

现在，我们将上面的两个采样函数包装到一个类中，以便稍后可以将其用作数据迭代器。

```python
class SeqDataLoader: #@save
"''"加载序列数据的迭代器''
def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
    if use_random_iter:
        self.data_iter_fn = d2l(seq_data_iter_random
        else:
            self.data_iter_fn = d2l(seq_data_iterSequential
            self.corpus, self.vocab = d2l.load(corpus_time_MACHINE(max_tokens)
            self.batch_size, self.num_steps = batch_size, num_steps
def __iter__(self):
    return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

最后，我们定义了一个函数load_data_time-machine，它同时返回数据迭代器和词表，因此可以与其他带有load_data前缀的函数（如3.5节中定义的d21.load_data_fashion_mnist）类似地使用。

```txt
def load_data_time-machine(batch_size，num_steps，@save use_random_iter  $\equiv$  False，max_tokens  $= 10000$  ：   
""返回时光机器数据集的迭代器和词表""data_iter  $\equiv$  SeqDataLoader( batch_size，num_steps，use_random_iter，max_tokens) return data_iter，data_iter.vocab
```

# 小结

- 语言模型是自然语言处理的关键。  
-  $n$  元语法通过截断相关性，为处理长序列提供了一种实用的模型。  
- 长序列存在一个问题：它们很少出现或者从不出现。  
- 齐普夫定律支配着单词的分布，这个分布不仅适用于一元语法，还适用于其他  $n$  元语法。  
- 通过拉普拉斯平滑法可以有效地处理结构丰富而频率不足的低频词词组。

# 8.3. 语言模型和数据集

- 读取长序列的主要方式是随机采样和顺序分区。在迭代过程中，后者可以保证来自两个相邻的小批量中的子序列在原始序列上也是相邻的。

# 练习

1. 假设训练数据集中有 100,000 个单词。一个四元语法需要存储多少个词频和相邻多词频率?  
2. 我们如何对一系列对话建模？  
3. 一元语法、二元语法和三元语法的齐普夫定律的指数是不一样的，能设法估计么？  
4. 想一想读取长序列数据的其他方法？  
5. 考虑一下我们用于读取长序列的随机偏移量。

1. 为什么随机偏移量是个好主意？  
2. 它真的会在文档的序列上实现完美的均匀分布吗？  
3. 要怎么做才能使分布更均匀？

6. 如果我们希望一个序列样本是一个完整的句子，那么这在小批量抽样中会带来怎样的问题？如何解决? Discussions<sup>102</sup>

# 8.4 循环神经网络

在 8.3节中, 我们介绍了  $n$  元语法模型, 其中单词  $x_{t}$  在时间步  $t$  的条件概率仅取决于前面  $n - 1$  个单词。对于时间步  $t - (n - 1)$  之前的单词, 如果我们想将其可能产生的影响合并到  $x_{t}$  上, 需要增加  $n$ , 然而模型参数的数量也会随之呈指数增长, 因为词表  $\nu$  需要存储  $|\nu|^{n}$  个数字, 因此与其将  $P(x_{t} \mid x_{t-1}, \ldots, x_{t-n+1})$  模型化, 不如使用隐变量模型:

$$
P \left(x _ {t} \mid x _ {t - 1}, \dots , x _ {1}\right) \approx P \left(x _ {t} \mid h _ {t - 1}\right), \tag {8.4.1}
$$

其中  $h_{t-1}$  是隐状态（hidden state），也称为隐藏变量（hidden variable），它存储了到时间步  $t-1$  的序列信息。通常，我们可以基于当前输入  $x_t$  和先前隐状态  $h_{t-1}$  来计算时间步  $t$  处的任何时间的隐状态：

$$
h _ {t} = f \left(x _ {t}, h _ {t - 1}\right). \tag {8.4.2}
$$

对于 (8.4.2) 中的函数  $f$ ，隐变量模型不是近似值。毕竟  $h_t$  是可以仅仅存储到目前为止观察到的所有数据，然而这样的操作可能会使计算和存储的代价都变得昂贵。

回想一下，我们在4节中讨论过的具有隐藏单元的隐藏层。值得注意的是，隐藏层和隐状态指的是两个截然不同的概念。如上所述，隐藏层是在从输入到输出的路径上（以观测角度来理解）的隐藏的层，而隐状态则是在给定步骤所做的任何事情（以技术角度来定义）的输入，并且这些状态只能通过先前时间步的数据来计算。

循环神经网络（recurrent neural networks，RNNs）是具有隐状态的神经网络。在介绍循环神经网络模型之前，我们首先回顾4.1节中介绍的多层感知机模型。

# 8.4.1 无隐状态的神经网络

让我们来看一看只有单隐藏层的多层感知机。设隐藏层的激活函数为  $\phi$ ，给定一个小批量样本  $\mathbf{X} \in \mathbb{R}^{n \times d}$ ，其中批量大小为  $n$ ，输入维度为  $d$ ，则隐藏层的输出  $\mathbf{H} \in \mathbb{R}^{n \times h}$  通过下式计算：

$$
\mathbf {H} = \phi \left(\mathbf {X W} _ {x h} + \mathbf {b} _ {h}\right). \tag {8.4.3}
$$

在(8.4.3)中，我们拥有的隐藏层权重参数为  $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ ，偏置参数为  $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ ，以及隐藏单元的数目为  $h$ 。因此求和时可以应用广播机制（见2.1.3节）。接下来，将隐藏变量H用作输出层的输入。输出层由下式给出：

$$
\mathbf {O} = \mathbf {H} \mathbf {W} _ {h q} + \mathbf {b} _ {q}, \tag {8.4.4}
$$

其中， $\mathbf{0} \in \mathbb{R}^{n \times q}$  是输出变量， $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$  是权重参数， $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$  是输出层的偏置参数。如果是分类问题，我们可以用softmax(0)来计算输出类别的概率分布。

这完全类似于之前在8.1节中解决的回归问题，因此我们省略了细节。无须多言，只要可以随机选择“特征-标签”对，并且通过自动微分和随机梯度下降能够学习网络参数就可以了。

# 8.4.2 有隐状态的循环神经网络

有了隐状态后，情况就完全不同了。假设我们在时间步  $t$  有小批量输入  $\mathbf{X}_t \in \mathbb{R}^{n \times d}$  。换言之，对于  $n$  个序列样本的小批量， $\mathbf{X}_t$  的每一行对应于来自该序列的时间步  $t$  处的一个样本。接下来，用  $\mathbf{H}_t \in \mathbb{R}^{n \times h}$  表示时间步  $t$  的隐藏变量。与多层感知机不同的是，我们在这里保存了前一个时间步的隐藏变量  $\mathbf{H}_{t-1}$ ，并引入了一个新的权重参数  $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ ，来描述如何在当前时间步中使用前一个时间步的隐藏变量。具体地说，当前时间步隐藏变量由当前时间步的输入与前一个时间步的隐藏变量一起计算得出：

$$
\mathbf {H} _ {t} = \phi \left(\mathbf {X} _ {t} \mathbf {W} _ {x h} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h h} + \mathbf {b} _ {h}\right). \tag {8.4.5}
$$

与 (8.4.3)相比, (8.4.5)多添加了一项  $\mathbf{H}_{t-1}\mathbf{W}_{hh}$ , 从而实例化了 (8.4.2)。从相邻时间步的隐藏变量  $\mathbf{H}_t$  和  $\mathbf{H}_{t-1}$  之间的关系可知, 这些变量捕获并保留了序列直到其当前时间步的历史信息, 就如当前时间步下神经网络的状态或记忆, 因此这样的隐藏变量被称为隐状态 (hidden state)。由于在当前时间步中, 隐状态使用的定义与前一个时间步中使用的定义相同, 因此 (8.4.5)的计算是循环的 (recurrent)。于是基于循环计算的隐状态神经网络被命名为循环神经网络 (recurrent neural network)。在循环神经网络中执行 (8.4.5)计算的层称为循环层 (recurrent layer)。

有许多不同的方法可以构建循环神经网络，由(8.4.5)定义的隐状态的循环神经网络是非常常见的一种。对于时间步  $t$  ，输出层的输出类似于多层感知机中的计算：

$$
\mathbf {O} _ {t} = \mathbf {H} _ {t} \mathbf {W} _ {h q} + \mathbf {b} _ {q}. \tag {8.4.6}
$$

循环神经网络的参数包括隐藏层的权重  $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$  和偏置  $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ ，以及输出层的权重  $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$  和偏置  $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ 。值得一提的是，即使在不同的时间步，循环神经网络也总是使用这些模型参数。因此，循环神经网络的参数开销不会随着时间步的增加而增加。

图8.4.1展示了循环神经网络在三个相邻时间步的计算逻辑。在任意时间步  $t$ ，隐状态的计算可以被视为：

1. 拼接当前时间步  $t$  的输入  $\mathbf{X}_t$  和前一时间步  $t - 1$  的隐状态  $\mathbf{H}_{t - 1}$

# 8.4. 循环神经网络

2. 将拼接的结果送入带有激活函数  $\phi$  的全连接层。全连接层的输出是当前时间步  $t$  的隐状态  $\mathbf{H}_t$  。

在本例中, 模型参数是  $\mathbf{W}_{x h}$  和  $\mathbf{W}_{h h}$  的拼接, 以及  $\mathbf{b} _ { h }$  的偏置, 所有这些参数都来自 (8.4.5)。当前时间步  $t$  的隐状态  $\mathbf{H}_{t}$  将参与计算下一时间步  $t + 1$  的隐状态  $\mathbf{H}_{t + 1}$  。而且  $\mathbf{H}_{t}$  还将送入全连接输出层, 用于计算当前时间步  $t$  的输出  $\mathbf{O}_{t}$  。

![](images/be514429475d9fe1adffac77b3dc14fb976966bb10bac37edbaac1733090454e.jpg)  
图8.4.1: 具有隐状态的循环神经网络

我们刚才提到，隐状态中  $\mathbf{X}_t\mathbf{W}_{xh} + \mathbf{H}_{t - 1}\mathbf{W}_{hh}$  的计算，相当于  $\mathbf{X}_t$  和  $\mathbf{H}_{t - 1}$  的拼接与  $\mathbf{W}_{xh}$  和  $\mathbf{W}_{hh}$  的拼接的矩阵乘法。虽然这个性质可以通过数学证明，但在下面我们使用一个简单的代码来说明一下。首先，我们定义矩阵X、W_xh、H和W_hh，它们的形状分别为(3#1)、(1#4)、(3#4)和(4#4)。分别将X乘以W_xh，将H乘以W_hh，然后将这两个乘法相加，我们得到一个形状为(3#4)的矩阵。

```python
import torch
from d21 import torch as d21
```

```txt
X，W_xh = torch.normal(0,1,(3,1))，torch.normal(0,1,(1,4))  
H，W_hh = torch.normal(0,1,(3,4))，torch.normal(0,1,(4,4))  
torch/matmul(X,W_xh) + torch/matmul(H,W_hh)
```

```txt
tensor([-1.6506, -0.7309, 2.0021, -0.1055], [1.7334, 2.2035, -3.3148, -2.1629], [-2.0071, -1.0902, 0.2376, -1.3144])
```

现在，我们沿列（轴1）拼接矩阵X和H，沿行（轴0）拼接矩阵W_xh和W_hh。这两个拼接分别产生形状(3,5)和形状(5,4)的矩阵。再将这两个拼接的矩阵相乘，我们得到与上面相同形状(3,4)的输出矩阵。

```javascript
torch.cat((X,H),1)，torch.cat((W_xh,W_hh),0))
```

```javascript
tensor([[-1.6506, -0.7309, 2.0021, -0.1055], [1.7334, 2.2035, -3.3148, -2.1629],
```

```txt
(continues on next page)
```

[-2.0071，-1.0902，0.2376，-1.3144]]）

# 8.4.3 基于循环神经网络的字符级语言模型

回想一下8.3节中的语言模型，我们的目标是根据过去的和当前的词元预测下一个词元，因此我们将原始序列移位一个词元作为标签。Bengio等人首先提出使用神经网络进行语言建模(Bengio et al., 2003)。接下来，我们看一下如何使用循环神经网络来构建语言模型。设小批量大小为1，批量中的文本序列为“machine”。为了简化后续部分的训练，我们考虑使用字符级语言模型（character-level language model），将文本词元化为字符而不是单词。图8.4.2演示了如何通过基于字符级语言建模的循环神经网络，使用当前的和先前的字符预测下一个字符。

![](images/3e02f5d7a90ee00ddcb90c38cd7a04f0344a87663535647b174395af6d8428da.jpg)  
图8.4.2: 基于循环神经网络的字符级语言模型：输入序列和标签序列分别为“machin”和“achine”

在训练过程中，我们对每个时间步的输出层的输出进行softmax操作，然后利用交叉熵损失计算模型输出和标签之间的误差。由于隐藏层中隐状态的循环计算，图8.4.2中的第3个时间步的输出  $\mathbf{O}_3$  由文本序列“m”“a”和“c”确定。由于训练数据中这个文本序列的下一个字符是“h”，因此第3个时间步的损失将取决于下一个字符的概率分布，而下一个字符是基于特征序列“m”“a”“c”和这个时间步的标签“h”生成的。

在实践中，我们使用的批量大小为  $n > 1$  ，每个词元都由一个  $d$  维向量表示。因此，在时间步  $t$  输入  $\mathbf{X}_t$  将是一个  $n \times d$  矩阵，这与我们在8.4.2节中的讨论相同。

# 8.4.4 困惑度 (Perplexity)

最后，让我们讨论如何度量语言模型的质量，这将在后续部分中用于评估基于循环神经网络的模型。一个好的语言模型能够用高度准确的词元来预测我们接下来会看到什么。考虑一下由不同的语言模型给出的对“It is raining …”（“…下雨了”）的续写：

1. “It is raining outside”（外面下雨了）；  
2. “It is raining banana tree”（香蕉树下雨了）；  
3. “It is raining piouw; kcj pwepoiut” (piouw; kcj pwepoiut下雨了)。

就质量而言，例1显然是最合乎情理、在逻辑上最连贯的。虽然这个模型可能没有很准确地反映出后续词的语义，比如，“It is raining in San Francisco”（旧金山下雨了）和“It is raining in winter”（冬天下雨了）可能才是更完美的合理扩展，但该模型已经能够捕捉到跟在后面的是哪类单词。例2则要糟糕得多，因为其产生了一个无意义的续写。尽管如此，至少该模型已经学会了如何拼写单词，以及单词之间的某种程度的相关性。最后，例3表明了训练不足的模型是无法正确地拟合数据的。

我们可以通过计算序列的似然概率来度量模型的质量。然而这是一个难以理解、难以比较的数字。毕竟，较短的序列比较长的序列更有可能出现，因此评估模型产生托尔斯泰的巨著《战争与和平》的可能性不可避免地会比产生圣埃克苏佩里的中篇小说《小王子》可能性要小得多。而缺少的可能性值相当于平均数。

在这里，信息论可以派上用场了。我们在引入softmax回归（3.4.7节）时定义了熵、惊异和交叉熵，并在信息论的在线附录103中讨论了更多的信息论知识。如果想要压缩文本，我们可以根据当前词元集预测的下一个词元。一个更好的语言模型应该能让我们更准确地预测下一个词元。因此，它应该允许我们在压缩序列时花费更少的比特。所以我们可以通过一个序列中所有的  $n$  个词元的交叉熵损失的平均值来衡量：

$$
\frac {1}{n} \sum_ {t = 1} ^ {n} - \log P \left(x _ {t} \mid x _ {t - 1}, \dots , x _ {1}\right), \tag {8.4.7}
$$

其中  $P$  由语言模型给出,  $x_{t}$  是在时间步  $t$  从该序列中观察到的实际词元。这使得不同长度的文档的性能具有了可比性。由于历史原因, 自然语言处理的科学家更喜欢使用一个叫做困惑度（perplexity）的量。简而言之, 它是 (8.4.7) 的指数:

$$
\exp \left(- \frac {1}{n} \sum_ {t = 1} ^ {n} \log P \left(x _ {t} \mid x _ {t - 1}, \dots , x _ {1}\right)\right). \tag {8.4.8}
$$

困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”。我们看看一些案例。

- 在最好的情况下，模型总是完美地估计标签词元的概率为1。在这种情况下，模型的困惑度为1。  
- 在最坏的情况下，模型总是预测标签词元的概率为0。在这种情况下，困惑度是正无穷大。  
- 在基线上，该模型的预测是词表的所有可用词元上的均匀分布。在这种情况下，困惑度等于词表中唯一词元的数量。事实上，如果我们在没有任何压缩的情况下存储序列，这将是我们能做的最好的编码方式。因此，这种方式提供了一个重要的上限，而任何实际模型都必须超越这个上限。

在接下来的小节中，我们将基于循环神经网络实现字符级语言模型，并使用困惑度来评估这样的模型。

# 小结

- 对隐状态使用循环计算的神经网络称为循环神经网络（RNN）。  
- 循环神经网络的隐状态可以捕获直到当前时间步序列的历史信息。  
- 循环神经网络模型的参数数量不会随着时间步的增加而增加。  
- 我们可以使用循环神经网络创建字符级语言模型。  
- 我们可以使用困惑度来评价语言模型的质量。

# 练习

1. 如果我们使用循环神经网络来预测文本序列中的下一个字符，那么任意输出所需的维度是多少？  
2. 为什么循环神经网络可以基于文本序列中所有先前的词元，在某个时间步表示当前词元的条件概率？  
3. 如果基于一个长序列进行反向传播，梯度会发生什么状况？  
4. 与本节中描述的语言模型相关的问题有哪些？

Discussions<sup>104</sup>

# 8.5 循环神经网络的从零开始实现

本节将根据8.4节中的描述，从头开始基于循环神经网络实现字符级语言模型。这样的模型将在H.G.Wells的时光机器数据集上训练。和前面8.3节中介绍过的一样，我们先读取数据集。

```txt
%matplotlib inline  
import math  
import torch  
from torch import nn  
from torch.nn import functional as F  
from d21 import torch as d21
```

```python
batch_size, num_steps = 32, 35  
train_iter, vocab = d21.load_data_time-machine(batch_size, num_steps)
```

# 8.5.1 独热编码

回想一下，在train_iter中，每个词元都表示为一个数字索引，将这些索引直接输入神经网络可能会使学习变得困难。我们通常将每个词元表示为更具表现力的特征向量。最简单的表示称为独热编码(one-hot encoding)，它在3.4.1节中介绍过。

简言之, 将每个索引映射为相互不同的单位向量: 假设词表中不同词元的数目为  $N$  (即len(vocab)), 词元素引的范围为0到  $N - 1$  。如果词元的索引是整数  $i$ , 那么我们将创建一个长度为  $N$  的全0向量, 并将第  $i$  处的元素设置为1。此向量是原始词元的一个独热向量。索引为0和2的独热向量如下所示:

```txt
F.one-hot(torch.tensor([0,2]),len(vocab))
```

```txt
tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])
```

我们每次采样的小批量数据形状是二维张量：(批量大小，时间步数)。one-hot函数将这样一个小批量数据转换成三维张量，张量的最后一个维度等于词表大小（len(vocab))）。我们经常转换输入的维度，以便获得形状为（时间步数，批量大小，词表大小）的输出。这将使我们能够更方便地通过最外层的维度，一步一步地更新小批量数据的隐状态。

```txt
X = torch.arange(10).reshape((2, 5))  
F.one-hot(X.T, 28).shape
```

```txt
torch.Size([5, 2, 28])
```

# 8.5.2 初始化模型参数

接下来，我们初始化循环神经网络模型的模型参数。隐藏单元数num_hiddens是一个可调的超参数。当训练语言模型时，输入和输出来自相同的词表。因此，它们具有相同的维度，即词表的大小。

```python
def get.params(vocab_size, num_hiddens, device):
    num Inputs = num Outputs = vocab_size
    def normal(shape):
        return torch rand(size=shape, device=device) * 0.01
    #隐藏层参数
    W_xh = normal((num Inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    #输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    #附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        paramrequires_grad_(True)
    return params
```

# 8.5.3 循环神经网络模型

为了定义循环神经网络模型，我们首先需要一个init_rnn_state函数在初始化时返回隐状态。这个函数的返回是一个张量，张量全用0填充，形状为（批量大小，隐藏单元数）。在后面的章节中我们将会遇到隐状态包含多个变量的情况，而使用元组可以更容易地处理些。

```python
def init_rnn_state(batch_size, num_hiddens, device): return (torch.zeros((batch_size, num_hiddens), device=device), )
```

下面的rnn函数定义了如何在一个时间步内计算隐状态和输出。循环神经网络模型通过inputs最外层的维度实现循环，以便逐时间步更新小批量数据的隐状态H。此外，这里使用tanh函数作为激活函数。如4.1节所述，当元素在实数上满足均匀分布时，tanh函数的平均值为0。

```javascript
def rnn(input, state, params): # inputs的形状：（时间步数量，批量大小，词表大小） W_xh,W_hh,b_h,W_hq,b_q  $=$  params H,  $=$  state outputs  $\equiv$  [] #x的形状：（批量大小，词表大小） for X in inputs: H  $=$  torch.tanh(torch.mm(X,W_xh)+torch.mm(H,Whh)+b_h) Y  $=$  torch.mm(H,W_hq)  $^+$  b_q outputs.append(Y) return torch.cat(output, dim=0), (H,)
```

定义了所有需要的函数之后，接下来我们创建一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数。

```python
class RNNModelScratch: #save
    ""从零开始实现的循环神经网络模型''
    def __init__(self, vocab_size, num_hiddens, device,
                    get.params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get.params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        X = F.one-hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

让我们检查输出是否具有正确的形状。例如，隐状态的维数是否保持不变。

# 8.5. 循环神经网络的从零开始实现

```matlab
num_hiddens = 512  
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get.params, init_rnn_state, rnn)  
state = net.begin_state(X.shape[0], d2l.try_gpu())  
Y, new_state = net(X.to(d2l.try_gpu(), state)  
Y.shape, len(new_state), new_state[0].shape
```

```txt
torch.Size([10, 28]), 1, torch.Size([2, 512]))
```

我们可以看到输出形状是（时间步数  $\times$  批量大小，词表大小），而隐状态形状保持不变，即（批量大小，隐藏单元数）。

# 8.5.4 预测

让我们首先定义预测函数来生成prefix之后的新字符，其中的prefix是一个用户提供的包含多个字符的字符串。在循环遍历prefix中的开始字符时，我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。这被称为预热（warm-up）期，因为在此期间模型会自我更新（例如，更新隐状态），但不会进行预测。预热期结束后，隐状态的值通常比刚开始的初始值更适合预测，从而预测字符并输出它们。

```python
def predict_ch8(prefix, num_preds, net, vocab, device): #@save
    ""在prefix后面生成新字符''
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab Prefix[0]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1]: # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ""
    return ([vocab.idx_to_token[i] for i in outputs])
```

现在我们可以测试predict_ch8函数。我们将前缀指定为time traveller，并基于这个前缀生成10个后续字符。鉴于我们还没有训练网络，它会生成荒谬的预测结果。

```python
predict_ch8('time traveller', 10, net, vocab, d21.try_gpu())
```

```txt
'time traveller aaaaaaaa'
```

# 8.5.5 梯度裁剪

对于长度为  $T$  的序列, 我们在迭代中计算这  $T$  个时间步上的梯度, 将会在反向传播过程中产生长度为  $\mathcal{O}(T)$  的矩阵乘法链。如 4.8 节所述, 当  $T$  较大时, 它可能导致数值不稳定, 例如可能导致梯度爆炸或梯度消失。因此, 循环神经网络模型往往需要额外的方式来支持稳定训练。

一般来说，当解决优化问题时，我们对模型参数采用更新步骤。假定在向量形式的  $\mathbf{x}$  中，或者在小批量数据的负梯度  $\mathbf{g}$  方向上。例如，使用  $\eta > 0$  作为学习率时，在一次迭代中，我们将  $\mathbf{x}$  更新为  $\mathbf{x} - \eta \mathbf{g}$  。如果我们进一步假设目标函数  $f$  表现良好，即函数  $f$  在常数  $L$  下是利普希茨连续的（Lipschitz continuous）。也就是说，对于任意  $\mathbf{x}$  和  $\mathbf{y}$  我们有：

$$
\left| f (\mathbf {x}) - f (\mathbf {y}) \right| \leq L \| \mathbf {x} - \mathbf {y} \|. \tag {8.5.1}
$$

在这种情况下，我们可以安全地假设：如果我们通过  $\eta \mathbf{g}$  更新参数向量，则

$$
\left| f (\mathbf {x}) - f (\mathbf {x} - \eta \mathbf {g}) \right| \leq L \eta \| \mathbf {g} \|, \tag {8.5.2}
$$

这意味着我们不会观察到超过  $L\eta \|\mathbf{g}\|$  的变化。这既是坏事也是好事。坏的方面，它限制了取得进展的速度；好的方面，它限制了事情变糟的程度，尤其当我们朝着错误的方向前进时。

有时梯度可能很大, 从而优化算法可能无法收敛。我们可以通过降低  $\eta$  的学习率来解决这个问题。但是如果我们很少得到大的梯度呢? 在这种情况下, 这种做法似乎毫无道理。一个流行的替代方案是通过将梯度  $\mathbf{g}$  投影回给定半径（例如  $\theta$ ）的球来裁剪梯度  $\mathbf{g}$  。如下式:

$$
\mathbf {g} \leftarrow \min  \left(1, \frac {\theta}{\| \mathbf {g} \|}\right) \mathbf {g}. \tag {8.5.3}
$$

通过这样做，我们知道梯度范数永远不会超过  $\theta$  ，并且更新后的梯度完全与  $\mathbf{g}$  的原始方向对齐。它还有一个值得拥有的副作用，即限制任何给定的小批量数据（以及其中任何给定的样本）对参数向量的影响，这赋予了模型一定程度的稳定性。梯度裁剪提供了一个快速修复梯度爆炸的方法，虽然它并不能完全解决问题，但它是众多有效的技术之一。

下面我们定义一个函数来裁剪模型的梯度，模型是从零开始实现的模型或由高级API构建的模型。我们在此计算了所有模型参数的梯度的范数。

```python
def grad_clipping(net, theta): save
    """裁剪梯度"
    if isinstance(net, nnModule):
        params = [p for p in net.params() if prequires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

# 8.5.6 训练

在训练模型之前，让我们定义一个函数在一个迭代周期内训练模型。它与我们训练3.6节模型的方式有三个不同之处。

1. 序列数据的不同采样方法（随机采样和顺序分区）将导致隐状态初始化的差异。  
2. 我们在更新模型参数之前裁剪梯度。这样的操作的目的是，即使训练过程中某个点上发生了梯度爆炸，也能保证模型不会发散。  
3. 我们用困惑度来评价模型。如 8.4.4 节所述，这样的度量确保了不同长度的序列具有可比性。

具体来说，当使用顺序分区时，我们只在每个迭代周期的开始位置初始化隐状态。由于下一个小批量数据中的第  $i$  个子序列样本与当前第  $i$  个子序列样本相邻，因此当前小批量数据最后一个样本的隐状态，将用于初始化下一个小批量数据第一个样本的隐状态。这样，存储在隐状态中的序列的历史信息可以在一个迭代周期内流经相邻的子序列。然而，在任何一点隐状态的计算，都依赖于同一迭代周期中前面所有的小批量数据，这使得梯度计算变得复杂。为了降低计算量，在处理任何一个小批量数据之前，我们先分离梯度，使得隐状态的梯度计算总是限制在一个小批量数据的时间步内。

当使用随机抽样时，因为每个样本都是在一个随机位置抽样的，因此需要为每个迭代周期重新初始化隐状态。与3.6节中的train_epoch_ch3函数相同，updater是更新模型参数的常用函数。它既可以是从头开始实现的d21.sgd函数，也可以是深度学习框架中内置的优化函数。

```python
#save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    '''训练网络一个迭代周期（定义见第8章）''' state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和，词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                statedetach_
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    sdetach_
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state)
            l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optimnymixer):
            updater.zero_grad()
```

(continues on next page)

```txt
l.backup()  
grad_clipping(net, 1)  
 updater_step()  
else:  
    l.backup()  
    grad_clipping(net, 1)  
# 因为已经调用了mean函数  
 updater(batch_size=1)  
metric.add(1 * y.numel(), y.Numel())  
return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

循环神经网络模型的训练函数既支持从零开始实现，也可以使用高级API来实现。

```python
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    '''训练模型（定义见第8章）''
    loss = nn.CrossEntropyLoss()
    optimizer = d2lancesizer(xlabel='epoch', ylabel='perplexity',
                    legend=['train'], xlim=[10, num_epochs])
# 初始化
if isinstance(net, nn.Module):
    updater = torch.optim.SGD(net.params(), lr)
else:
    updater = lambda batch_size: d2l SGD(net.params, lr, batch_size)
predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
# 训练和预测
for epoch in range(num_epochs):
    ppl, speed = train_epoch_ch8(
        net, train_iter, loss, updater, device, use_random_iter)
    if (epoch + 1) % 10 == 0:
        print(predict('time traveller'))
        optimizer.add(epoeh + 1, [ppl])
print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device})')
print(predict('time traveller'))
print(predict('traveller'))
```

现在，我们训练循环神经网络模型。因为我们在数据集中只使用了10000个词元，所以模型需要更多的迭代周期来更好地收敛。

```txt
num_epochs, lr = 500, 1  
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

困惑度1.0，67212.6词元/秒cuda:0time traveller for so it will be convenient to speak of himwasetravelleryou can show black is white by argument said filby

![](images/c95c1b0cbf4953d6dad121a8ad607e054cc4219fbd13babc217e8ef87d310c04.jpg)

最后，让我们检查一下使用随机抽样方法的结果。

```txt
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get.params, init_rnn_state, rnn)  
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)
```

困惑度1.5，65222.3词元/秒cuda:0time traveller held in his hand was a glitteringmetallic framewotraveller but now you begin to seethe object of my investig

![](images/777e9add46efc122c6aa793094e37e51e83cd17c44dd2b79297deb74d623e53f.jpg)

从零开始实现上述循环神经网络模型，虽然有指导意义，但是并不方便。在下一节中，我们将学习如何改进循环神经网络模型。例如，如何使其实现地更容易，且运行速度更快。

# 小结

- 我们可以训练一个基于循环神经网络的字符级语言模型，根据用户提供的文本的前缀生成后续文本。  
- 一个简单的循环神经网络语言模型包括输入编码、循环神经网络模型和输出生成。  
- 循环神经网络模型在训练以前需要初始化状态，不过随机抽样和顺序划分使用初始化方法不同。  
- 当使用顺序划分时，我们需要分离梯度以减少计算量。  
- 在进行任何预测之前，模型通过预热期进行自我更新（例如，获得比初始值更好的隐状态）。  
- 梯度裁剪可以防止梯度爆炸，但不能应对梯度消失。

# 练习

1. 尝试说明独热编码等价于为每个对象选择不同的嵌入表示。  
2. 通过调整超参数（如迭代周期数、隐藏单元数、小批量数据的时间步数、学习率等）来改善困惑度。

- 困惑度可以降到多少？  
- 用可学习的嵌入表示替换独热编码，是否会带来更好的表现？  
- 如果用H.G.Wells的其他书作为数据集时效果如何，例如世界大战105？

3. 修改预测函数，例如使用采样，而不是选择最有可能的下一个字符。

- 会发生什么？  
- 调整模型使之偏向更可能的输出, 例如, 当  $\alpha > 1$ , 从  $q(x_{t} \mid x_{t - 1}, \ldots, x_{1}) \propto P(x_{t} \mid x_{t - 1}, \ldots, x_{1})^{\alpha}$  中采样。

4. 在不裁剪梯度的情况下运行本节中的代码会发生什么？  
5. 更改顺序划分，使其不会从计算图中分离隐状态。运行时间会有变化吗？困惑度呢？  
6. 用ReLU替换本节中使用的激活函数，并重复本节中的实验。我们还需要梯度裁剪吗？为什么？

Discussions<sup>106</sup>

# 8.6 循环神经网络的简洁实现

虽然8.5节对了解循环神经网络的实现方式具有指导意义，但并不方便。本节将展示如何使用深度学习框架的高级API提供的函数更有效地实现相同的语言模型。我们仍然从读取时光机器数据集开始。

```python
import torch   
from torch import nn   
from torch(nn import functional as F   
from d21 import torch as d21   
batch_size, num_steps = 32, 35   
train_iter, vocab = d21.load_data_time-machine(batch_size, num_steps)
```

# 8.6.1 定义模型

高级API提供了循环神经网络的实现。我们构造一个具有256个隐藏单元的单隐藏层的循环神经网络层rnn_layer。事实上，我们还没有讨论多层循环神经网络的意义（这将在9.3节中介绍）。现在仅需要将多层理解为一层循环神经网络的输出被用作下一层循环神经网络的输入就足够了。

```txt
num_hiddens = 256  
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

我们使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）。

```javascript
state  $=$  torch.zeros((1，batch_size，num_hiddens)) state.shape
```

```txt
torch.Size([1, 32, 256])
```

通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。需要强调的是，rnn_layer的“输出”(Y)不涉及输出层的计算：它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。

```txt
X = torch rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

```txt
torch.Size([35, 32, 256]), torch.Size([1, 32, 256]))
```

与8.5节类似，我们为一个完整的循环神经网络模型定义了一个RNNModel类。注意，rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层。

```txt
@save   
class RNNModel(nnModule): ""循环神经网络模型"" def__init__(self，rnn_layer，vocab_size，\*\*kwargs):
```

```txt
(continues on next page)
```

```python
super(RNNModel, self).__init__(**kwargs)  
self.rnn = rnn_layer  
self.vocab_size = vocab_size  
self.num_hiddens = self.rnn-hidden_size  
# 如果RNN是双向的（之后将介绍），num Directions应该是2，否则应该是1  
if not self.rnn bidirectional:  
    self.num Directions = 1  
    self.linear = nn.Linear(self.num_hiddens, self.vocab_size)  
else:  
    self.num Directions = 2  
    self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)  
def forward(self, inputs, state):  
    X = F.onehot(inputs.T.long(), self.vocab_size)  
    X = X.to(torch.float32)  
    Y, state = self.rnn(X, state)  
# 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)  
# 它的输出形状是(时间步数*批量大小,词表大小)。  
output = self.linear(Y.reshape((-1, Y.shape[-1]))  
return output, state  
def begin_state(self, device, batch_size=1):  
    if not isinstance(self.rnn, nn.LSTM):  
        # nn.GRU以张量作为隐状态  
        return torch.zeros((self.num Directions * self.rnn(num_layers, batch_size, self.num_hiddens), device=device)  
    else:  
        # nn.LSTM以元组作为隐状态  
        return (torch.zeros((self.num Directions * self.rnn(num_layers, batch_size, self.num_hiddens), device=device), torch.zeros((self.num Directions * self.rnn(num_layers, batch_size, self(num_hiddens), device=device))
```

# 8.6.2 训练与预测

在训练模型之前，让我们基于一个具有随机权重的模型进行预测。

```python
device = d21.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d21.predict_ch8('time traveller', 10, net, vocab, device)
```

```txt
'time travellerbbabbkabyg'
```

很明显，这种模型根本不能输出好的结果。接下来，我们使用8.5节中定义的超参数调用train_ch8，并且使用高级API训练模型。

```python
num_epochs, lr = 500, 1  
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```txt
perplexity 1.3, 404413.8 tokens/sec on CUDA:0  
time travellerit would be remarkably convenient for the historiatriavellery of il the hise rupt might and st was it loflers
```

![](images/51bcdc67857590c14ffa5a92dbefcd807a9301d8514a56f3a660f0c2fcccfdcf.jpg)

与上一节相比，由于深度学习框架的高级API对代码进行了更多的优化，该模型在较短的时间内达到了较低的困惑度。

# 小结

- 深度学习框架的高级API提供了循环神经网络层的实现。  
- 高级API的循环神经网络层返回一个输出和一个更新后的隐状态，我们还需要计算整个模型的输出层。  
- 相比从零开始实现的循环神经网络，使用高级API实现可以加速训练。

# 练习

1. 尝试使用高级API，能使循环神经网络模型过拟合吗？  
2. 如果在循环神经网络模型中增加隐藏层的数量会发生什么？能使模型正常工作吗？  
3. 尝试使用循环神经网络实现 8.1 节的自回归模型。

Discussions107

# 8.7 通过时间反向传播

到目前为止, 我们已经反复提到像梯度爆炸或梯度消失, 以及需要对循环神经网络分离梯度。例如, 在 8.5 节中,我们在序列上调用了 detach函数。为了能够快速构建模型并了解其工作原理, 上面所说的这些概念都没有得到充分的解释。本节将更深入地探讨序列模型反向传播的细节, 以及相关的数学原理。

当我们首次实现循环神经网络（8.5节）时，遇到了梯度爆炸的问题。如果做了练习题，就会发现梯度截断对于确保模型收敛至关重要。为了更好地理解此问题，本节将回顾序列模型梯度的计算方式，它的工作原理没有什么新概念，毕竟我们使用的仍然是链式法则来计算梯度。

我们在4.7节中描述了多层感知机中的前向与反向传播及相关的计算图。循环神经网络中的前向传播相对简单。通过时间反向传播（backpropagation through time，BPTT）(Werbos, 1990)实际上是循环神经网络中反向传播技术的一个特定应用。它要求我们将循环神经网络的计算图一次展开一个时间步，以获得模型变量和参数之间的依赖关系。然后，基于链式法则，应用反向传播来计算和存储梯度。由于序列可能相当长，因此依赖关系也可能相当长。例如，某个1000个字符的序列，其第一个词元可能会对最后位置的词元产生重大影响。这在计算上是不可行的（它需要的时间和内存都太多了），并且还需要超过1000个矩阵的乘积才能得到非常难以捉摸的梯度。这个过程充满了计算与统计的不确定性。在下文中，我们将阐明会发生什么以及如何在实践中解决它们。

# 8.7.1 循环神经网络的梯度分析

我们从一个描述循环神经网络工作原理的简化模型开始，此模型忽略了隐状态的特性及其更新方式的细节。这里的数学表示没有像过去那样明确地区分标量、向量和矩阵，因为这些细节对于分析并不重要，反而只会使本小节中的符号变得混乱。

在这个简化模型中,我们将时间步  $t$  的隐状态表示为  $h_{t}$ , 输入表示为  $x_{t}$ , 输出表示为  $o_{t}$  。回想一下我们在 8.4.2 节中的讨论, 输入和隐状态可以拼接后与隐藏层中的一个权重变量相乘。因此, 我们分别使用  $w_{h}$  和  $w_{o}$  来表示隐藏层和输出层的权重。每个时间步的隐状态和输出可以写为:

$$
h _ {t} = f \left(x _ {t}, h _ {t - 1}, w _ {h}\right), \tag {8.7.1}
$$

$$
o _ {t} = g (h _ {t}, w _ {o}),
$$

其中 \(f\) 和 \(g\) 分别是隐藏层和输出层的变换。因此，我们有一个链 \(\{...\), \left(x_{t-1}, h_{t-1}, o_{t-1}\right), \left(x_{t}, h_{t}, o_{t}\right), \ldots\}\)，它们通过循环计算彼此依赖。前向传播相当简单，一次一个时间步的遍历三元组 \(\left(x_{t}, h_{t}, o_{t}\right)\)，然后通过一个目标函数在所有 \(T\) 个时间步内评估输出 \(o_{t}\) 和对应的标签 \(y_{t}\) 之间的差异：

$$
L \left(x _ {1}, \dots , x _ {T}, y _ {1}, \dots , y _ {T}, w _ {h}, w _ {o}\right) = \frac {1}{T} \sum_ {t = 1} ^ {T} l \left(y _ {t}, o _ {t}\right). \tag {8.7.2}
$$

对于反向传播，问题则有点棘手，特别是当我们计算目标函数  $L$  关于参数  $w_{h}$  的梯度时。具体来说，按照链式法则：

$$
\begin{array}{l} \frac {\partial L}{\partial w _ {h}} = \frac {1}{T} \sum_ {t = 1} ^ {T} \frac {\partial l \left(y _ {t} , o _ {t}\right)}{\partial w _ {h}} \tag {8.7.3} \\ = \frac {1}{T} \sum_ {t = 1} ^ {T} \frac {\partial l \left(y _ {t} , o _ {t}\right)}{\partial o _ {t}} \frac {\partial g \left(h _ {t} , w _ {o}\right)}{\partial h _ {t}} \frac {\partial h _ {t}}{\partial w _ {h}}. \\ \end{array}
$$

在 (8.7.3)中乘积的第一项和第二项很容易计算，而第三项  $\partial h_{t} / \partial w_{h}$  是使事情变得棘手的地方，因为我们需要循环地计算参数  $w_{h}$  对  $h_t$  的影响。根据 (8.7.1)中的递归计算， $h_t$  既依赖于  $h_{t-1}$  又依赖于  $w_{h}$ ，其中  $h_{t-1}$  的计算也依赖于  $w_{h}$ 。因此，使用链式法则产生：

$$
\frac {\partial h _ {t}}{\partial w _ {h}} = \frac {\partial f \left(x _ {t} , h _ {t - 1} , w _ {h}\right)}{\partial w _ {h}} + \frac {\partial f \left(x _ {t} , h _ {t - 1} , w _ {h}\right)}{\partial h _ {t - 1}} \frac {\partial h _ {t - 1}}{\partial w _ {h}}. \tag {8.7.4}
$$

为了导出上述梯度，假设我们有三个序列  $\{a_t\}, \{b_t\}, \{c_t\}$ ，当  $t = 1,2,\ldots$  时，序列满足  $a_0 = 0$  且  $a_{t} = b_{t} + c_{t}a_{t - 1}$ 。对于  $t \geq 1$ ，就很容易得出：

$$
a _ {t} = b _ {t} + \sum_ {i = 1} ^ {t - 1} \left(\prod_ {j = i + 1} ^ {t} c _ {j}\right) b _ {i}. \tag {8.7.5}
$$

基于下列公式替换  $a_{t}$  、  $b_{t}$  和  $c_{t}$

$$
a _ {t} = \frac {\partial h _ {t}}{\partial w _ {h}},
$$

$$
b _ {t} = \frac {\partial f \left(x _ {t} , h _ {t - 1} , w _ {h}\right)}{\partial w _ {h}}, \tag {8.7.6}
$$

$$
c _ {t} = \frac {\partial f (x _ {t} , h _ {t - 1} , w _ {h})}{\partial h _ {t - 1}},
$$

公式(8.7.4)中的梯度计算满足  $a_{t} = b_{t} + c_{t}a_{t - 1}$  。因此，对于每个(8.7.5)，我们可以使用下面的公式移除(8.7.4)中的循环计算

$$
\frac {\partial h _ {t}}{\partial w _ {h}} = \frac {\partial f \left(x _ {t} , h _ {t - 1} , w _ {h}\right)}{\partial w _ {h}} + \sum_ {i = 1} ^ {t - 1} \left(\prod_ {j = i + 1} ^ {t} \frac {\partial f \left(x _ {j} , h _ {j - 1} , w _ {h}\right)}{\partial h _ {j - 1}}\right) \frac {\partial f \left(x _ {i} , h _ {i - 1} , w _ {h}\right)}{\partial w _ {h}}. \tag {8.7.7}
$$

虽然我们可以使用链式法则递归地计算  $\partial h_{t} / \partial w_{h}$ , 但当  $t$  很大时这个链就会变得很长。我们需要想想办法来处理这一问题.

# 完全计算

显然，我们可以仅仅计算 (8.7.7) 中的全部总和，然而，这样的计算非常缓慢，并且可能会发生梯度爆炸，因为初始条件的微小变化就可能会对结果产生巨大的影响。也就是说，我们可以观察到类似于蝴蝶效应的现象，即初始条件的很小变化就会导致结果发生不成比例的变化。这对于我们想要估计的模型而言是非常不可取的。毕竟，我们正在寻找的是能够很好地泛化高稳定性模型的估计器。因此，在实践中，这种方法几乎从未使用过。

# 截断时间步

或者，我们可以在  $\tau$  步后截断(8.7.7)中的求和计算。这是我们到目前为止一直在讨论的内容，例如在8.5节中分离梯度时。这会带来真实梯度的近似，只需将求和终止为  $\partial h_{t - \tau} / \partial w_h$  。在实践中，这种方式工作得很好。它通常被称为截断的通过时间反向传播(Jaeger,2002)。这样做导致该模型主要侧重于短期影响，而不是长期影响。这在现实中是可取的，因为它会将估计值偏向更简单和更稳定的模型。

# 随机截断

最后，我们可以用一个随机变量替换  $\partial h_t / \partial w_h$  ，该随机变量在预期中是正确的，但是会截断序列。这个随机变量是通过使用序列  $\xi_t$  来实现的，序列预定义了  $0 \leq \pi_t \leq 1$  ，其中  $P(\xi_t = 0) = 1 - \pi_t$  且  $P(\xi_t = \pi_t^{-1}) = \pi_t$  ，因此  $E[\xi_t] = 1$  。我们使用它来替换 (8.7.4)中的梯度  $\partial h_t / \partial w_h$  得到：

$$
z _ {t} = \frac {\partial f \left(x _ {t} , h _ {t - 1} , w _ {h}\right)}{\partial w _ {h}} + \xi_ {t} \frac {\partial f \left(x _ {t} , h _ {t - 1} , w _ {h}\right)}{\partial h _ {t - 1}} \frac {\partial h _ {t - 1}}{\partial w _ {h}}. \tag {8.7.8}
$$

从  $\xi_{t}$  的定义中推导出来  $E[z_{t}] = \partial h_{t} / \partial w_{h}$  。每当  $\xi_{t} = 0$  时，递归计算终止在这个  $t$  时间步。这导致了不同长度序列的加权和，其中长序列出现的很少，所以将适当地加大权重。这个想法是由塔莱克和奥利维尔 (Tallec and Ollivier, 2017) 提出的。

# 比较策略

![](images/9d127bfbcb67213bf3c59e2070b1e61ba0a73f3dbadb091990f30f1355f27701.jpg)  
图8.7.1: 比较RNN中计算梯度的策略, 3行自上而下分别为: 随机截断、常规截断、完整计算

图8.7.1说明了当基于循环神经网络使用通过时间反向传播分析《时间机器》书中前几个字符的三种策略：

- 第一行采用随机截断，方法是将文本划分为不同长度的片断；  
- 第二行采用常规截断，方法是将文本分解为相同长度的子序列。这也是我们在循环神经网络实验中一直在做的；  
- 第三行采用通过时间的完全反向传播，结果是产生了在计算上不可行的表达式。

遗憾的是，虽然随机截断在理论上具有吸引力，但很可能是由于多种因素在实践中并不比常规截断更好。首先，在对过去若干个时间步经过反向传播后，观测结果足以捕获实际的依赖关系。其次，增加的方差抵消了时间步数越多梯度越精确的事实。第三，我们真正想要的是只有短范围交互的模型。因此，模型需要的正是截断的通过时间反向传播方法所具备的轻度正则化效果。

# 8.7.2 通过时间反向传播的细节

在讨论一般性原则之后，我们看一下通过时间反向传播问题的细节。与8.7.1节中的分析不同，下面我们将展示如何计算目标函数相对于所有分解模型参数的梯度。为了保持简单，我们考虑一个没有偏置参数的循环神经网络，其在隐藏层中的激活函数使用恒等映射  $(\phi (x) = x)$  。对于时间步  $t$  ，设单个样本的输入及其对应的标签分别为  $\mathbf{x}_t\in \mathbb{R}^d$  和  $y_{t}$  。计算隐状态  $\mathbf{h}_t\in \mathbb{R}^h$  和输出  $\mathbf{o}_t\in \mathbb{R}^q$  的方式为：

$$
\mathbf {h} _ {t} = \mathbf {W} _ {h x} \mathbf {x} _ {t} + \mathbf {W} _ {h h} \mathbf {h} _ {t - 1}, \tag {8.7.9}
$$

$$
\mathbf {o} _ {t} = \mathbf {W} _ {q h} \mathbf {h} _ {t},
$$

其中权重参数为  $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$  、  $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$  和  $\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$  。用  $l(\mathbf{o}_t, y_t)$  表示时间步  $t$  处（即从序列开始起的超过  $T$  个时间步）的损失函数，则我们的目标函数的总体损失是：

$$
L = \frac {1}{T} \sum_ {t = 1} ^ {T} l \left(\mathbf {o} _ {t}, y _ {t}\right). \tag {8.7.10}
$$

为了在循环神经网络的计算过程中可视化模型变量和参数之间的依赖关系, 我们可以为模型绘制一个计算图,如图8.7.2所示。例如, 时间步3的隐状态  $\mathbf{h}_{3}$  的计算取决于模型参数  $\mathbf{W}_{hx}$  和  $\mathbf{W}_{hh}$ , 以及最终时间步的隐状态  $\mathbf{h}_{2}$  以及当前时间步的输入  $\mathbf{x}_{3}$  。

![](images/d35315e66e74c3388a9dac0406f85648206c3a43c6e63c8c80457ff35357d1b1.jpg)  
图8.7.2: 上图表示具有三个时间步的循环神经网络模型依赖关系的计算图。未着色的方框表示变量，着色的方框表示参数，圆表示运算符

正如刚才所说, 图8.7.2中的模型参数是  $\mathbf{W}_{hx} 、 \mathbf{W}_{hh}$  和  $\mathbf{W}_{qh}$  。通常, 训练该模型需要对这些参数进行梯度计算:  $\partial L / \partial \mathbf{W}_{hx} 、 \partial L / \partial \mathbf{W}_{hh}$  和  $\partial L / \partial \mathbf{W}_{qh}$  。根据图8.7.2中的依赖关系, 我们可以沿箭头的相反方向遍历计算图, 依次计算和存储梯度。为了灵活地表示链式法则中不同形状的矩阵、向量和标量的乘法, 我们继续使用如4.7节中所述的prod运算符。

首先，在任意时间步  $t$ ，目标函数关于模型输出的微分计算是相当简单的：

$$
\frac {\partial L}{\partial \mathbf {o} _ {t}} = \frac {\partial l (\mathbf {o} _ {t} , y _ {t})}{T \cdot \partial \mathbf {o} _ {t}} \in \mathbb {R} ^ {q}. \tag {8.7.11}
$$

现在，我们可以计算目标函数关于输出层中参数  $\mathbf{W}_{qh}$  的梯度： $\partial L / \partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ 。基于图8.7.2，目标函数  $L$  通过  $\mathbf{o}_1, \ldots, \mathbf{o}_T$  依赖于  $\mathbf{W}_{qh}$ 。依据链式法则，得到

$$
\frac {\partial L}{\partial \mathbf {W} _ {q h}} = \sum_ {t = 1} ^ {T} \operatorname {p r o d} \left(\frac {\partial L}{\partial \mathbf {o} _ {t}}, \frac {\partial \mathbf {o} _ {t}}{\partial \mathbf {W} _ {q h}}\right) = \sum_ {t = 1} ^ {T} \frac {\partial L}{\partial \mathbf {o} _ {t}} \mathbf {h} _ {t} ^ {\top}, \tag {8.7.12}
$$

其中  $\partial L / \partial \mathbf{o}_t$  是由(8.7.11)给出的。

接下来，如图8.7.2所示，在最后的时间步  $T$  ，目标函数  $L$  仅通过  $\mathbf{o}_T$  依赖于隐状态  $\mathbf{h}_T$  。因此，我们通过使用链式法可以很容易地得到梯度  $\partial L / \partial \mathbf{h}_T\in \mathbb{R}^h$  ：

$$
\frac {\partial L}{\partial \mathbf {h} _ {T}} = \operatorname {p r o d} \left(\frac {\partial L}{\partial \mathbf {o} _ {T}}, \frac {\partial \mathbf {o} _ {T}}{\partial \mathbf {h} _ {T}}\right) = \mathbf {W} _ {q h} ^ {\top} \frac {\partial L}{\partial \mathbf {o} _ {T}}. \tag {8.7.13}
$$

当目标函数  $L$  通过  $\mathbf{h}_{t + 1}$  和  $\mathbf{o}_t$  依赖  $\mathbf{h}_t$  时，对任意时间步  $t < T$  来说都变得更加棘手。根据链式法则，隐状态的梯度  $\partial L / \partial \mathbf{h}_t\in \mathbb{R}^h$  在任何时间步骤  $t < T$  时都可以递归地计算为：

$$
\frac {\partial L}{\partial \mathbf {h} _ {t}} = \operatorname {p r o d} \left(\frac {\partial L}{\partial \mathbf {h} _ {t + 1}}, \frac {\partial \mathbf {h} _ {t + 1}}{\partial \mathbf {h} _ {t}}\right) + \operatorname {p r o d} \left(\frac {\partial L}{\partial \mathbf {o} _ {t}}, \frac {\partial \mathbf {o} _ {t}}{\partial \mathbf {h} _ {t}}\right) = \mathbf {W} _ {h h} ^ {\top} \frac {\partial L}{\partial \mathbf {h} _ {t + 1}} + \mathbf {W} _ {q h} ^ {\top} \frac {\partial L}{\partial \mathbf {o} _ {t}}. \tag {8.7.14}
$$

为了进行分析，对于任何时间步  $1 \leq t \leq T$  展开递归计算得

$$
\frac {\partial L}{\partial \mathbf {h} _ {t}} = \sum_ {i = t} ^ {T} \left(\mathbf {W} _ {h h} ^ {\top}\right) ^ {T - i} \mathbf {W} _ {q h} ^ {\top} \frac {\partial L}{\partial \mathbf {o} _ {T + t - i}}. \tag {8.7.15}
$$

我们可以从 (8.7.15) 中看到, 这个简单的线性例子已经展现了长序列模型的一些关键问题: 它陷入到  $\mathbf{W}_{hh}^{\top}$  的潜在的非常大的幂。在这个幂中, 小于 1 的特征值将会消失, 大于 1 的特征值将会发散。这在数值上是不稳定的, 表现形式为梯度消失或梯度爆炸。解决此问题的一种方法是按照计算方便的需要截断时间步长的尺寸如

8.7.1节中所述。实际上，这种截断是通过在给定数量的时间步之后分离梯度来实现的。稍后，我们将学习更复杂的序列模型（如长短期记忆模型）是如何进一步缓解这一问题的。

最后，图8.7.2表明：目标函数  $L$  通过隐状态  $\mathbf{h}_1,\dots ,\mathbf{h}_T$  依赖于隐藏层中的模型参数  $\mathbf{W}_{hx}$  和  $\mathbf{W}_{hh}$  。为了计算有关这些参数的梯度  $\partial L / \partial \mathbf{W}_{hx}\in \mathbb{R}^{h\times d}$  和  $\partial L / \partial \mathbf{W}_{hh}\in \mathbb{R}^{h\times h}$ ，我们应用链式规则得：

$$
\frac {\partial L}{\partial \mathbf {W} _ {h x}} = \sum_ {t = 1} ^ {T} \operatorname {p r o d} \left(\frac {\partial L}{\partial \mathbf {h} _ {t}}, \frac {\partial \mathbf {h} _ {t}}{\partial \mathbf {W} _ {h x}}\right) = \sum_ {t = 1} ^ {T} \frac {\partial L}{\partial \mathbf {h} _ {t}} \mathbf {x} _ {t} ^ {\top}, \tag {8.7.16}
$$

$$
\frac {\partial L}{\partial \mathbf {W} _ {h h}} = \sum_ {t = 1} ^ {T} \operatorname {p r o d} \left(\frac {\partial L}{\partial \mathbf {h} _ {t}}, \frac {\partial \mathbf {h} _ {t}}{\partial \mathbf {W} _ {h h}}\right) = \sum_ {t = 1} ^ {T} \frac {\partial L}{\partial \mathbf {h} _ {t}} \mathbf {h} _ {t - 1} ^ {\top},
$$

其中  $\partial L / \partial \mathbf{h}_t$  是由(8.7.13)和(8.7.14)递归计算得到的，是影响数值稳定性的关键量。

正如我们在4.7节中所解释的那样，由于通过时间反向传播是反向传播在循环神经网络中的应用方式，所以训练循环神经网络交替使用前向传播和通过时间反向传播。通过时间反向传播依次计算并存储上述梯度。具体而言，存储的中间值会被重复使用，以避免重复计算，例如存储  $\partial L / \partial \mathbf{h}_t$  ，以便在计算  $\partial L / \partial \mathbf{W}_{hx}$  和  $\partial L / \partial \mathbf{W}_{hh}$  时使用。

# 小结

- “通过时间反向传播”仅仅适用于反向传播在具有隐状态的序列模型。  
- 截断是计算方便性和数值稳定性的需要。截断包括：规则截断和随机截断。  
- 矩阵的高次幂可能导致神经网络特征值的发散或消失，将以梯度爆炸或梯度消失的形式表现。  
- 为了计算的效率，“通过时间反向传播”在计算期间会缓存中间值。

# 练习

1. 假设我们拥有一个对称矩阵  $\mathbf{M} \in \mathbb{R}^{n \times n}$ ，其特征值为  $\lambda_{i}$ ，对应的特征向量是  $\mathbf{v}_{i}$ （ $i = 1, \dots, n$ ）。通常情况下，假设特征值的序列顺序为  $|\lambda_{i}| \geq |\lambda_{i+1}|$ 。

1. 证明  $\mathbf{M}^k$  拥有特征值  $\lambda_i^k$ 。  
2. 证明对于一个随机向量  $\mathbf{x} \in \mathbb{R}^n$ ， $\mathbf{M}^k\mathbf{x}$  将有较高概率与  $\mathbf{M}$  的特征向量  $\mathbf{v}_1$  在一条直线上。形式化这个证明过程。  
3. 上述结果对于循环神经网络中的梯度意味着什么？

2. 除了梯度截断，还有其他方法来应对循环神经网络中的梯度爆炸吗？

Discussions<sup>108</sup>

# 现代循环神经网络

前一章中我们介绍了循环神经网络的基础知识，这种网络可以更好地处理序列数据。我们在文本数据上实现了基于循环神经网络的语言模型，但是对于当今各种各样的序列学习问题，这些技术可能并不够用。

例如，循环神经网络在实践中一个常见问题是数值不稳定性。尽管我们已经应用了梯度裁剪等技巧来缓解这个问题，但是仍需要通过设计更复杂的序列模型来进一步处理它。具体来说，我们将引入两个广泛使用的网络，即门控循环单元（gated recurrent units, GRU）和长短期记忆网络（long short-term memory, LSTM）。然后，我们将基于一个单向隐藏层来扩展循环神经网络架构。我们将描述具有多个隐藏层的深层架构，并讨论基于前向和后向循环计算的双向设计。现代循环网络经常采用这种扩展。在解释这些循环神经网络的变体时，我们将继续考虑8节中的语言建模问题。

事实上，语言建模只揭示了序列学习能力的冰山一角。在各种序列学习问题中，如自动语音识别、文本到语音转换和机器翻译，输入和输出都是任意长度的序列。为了阐述如何拟合这种类型的数据，我们将以机器翻译为例介绍基于循环神经网络的“编码器—解码器”架构和束搜索，并用它们来生成序列。

# 9.1 门控循环单元（GRU）

在8.7节中，我们讨论了如何在循环神经网络中计算梯度，以及矩阵连续乘积可以导致梯度消失或梯度爆炸的问题。下面我们简单思考一下这种梯度异常在实践中的意义：

- 我们可能会遇到这样的情况：早期观测值对预测所有未来观测值具有非常重要的意义。考虑一个极端情况，其中第一个观测值包含一个校验和，目标是在序列的末尾辨别校验和是否正确。在这种情况下，第一个词元的影响至关重要。我们希望有某些机制能够在一个记忆元里存储重要的早期信息。如果没有这样的机制，我们将不得不给这个观测值指定一个非常大的梯度，因为它会影响所有后续的观测值。

- 我们可能会遇到这样的情况：一些词元没有相关的观测值。例如，在对网页内容进行情感分析时，可能有一些辅助HTML代码与网页传达的情绪无关。我们希望有一些机制来跳过隐状态表示中的此类词元。  
- 我们可能会遇到这样的情况：序列的各个部分之间存在逻辑中断。例如，书的章节之间可能会有过渡存在，或者证券的熊市和牛市之间可能会有过渡存在。在这种情况下，最好有一种方法来重置我们的内部状态表示。

在学术界已经提出了许多方法来解决这类问题。其中最早的方法是“长短期记忆”（long-short-term memory, LSTM）(Hochreiter and Schmidhuber, 1997)，我们将在9.2节中讨论。门控循环单元（gated recurrent unit, GRU）(Cho et al., 2014)是一个稍微简化的变体，通常能够提供同等的效果，并且计算(Chung et al., 2014)的速度明显更快。由于门控循环单元更简单，我们从它开始解读。

# 9.1.1 门控隐状态

门控循环单元与普通的循环神经网络之间的关键区别在于：前者支持隐状态的门控。这意味着模型有专门的机制来确定应该何时更新隐状态，以及应该何时重置隐状态。这些机制是可学习的，并且能够解决了上面列出的问题。例如，如果第一个词元非常重要，模型将学会在第一次观测之后不更新隐状态。同样，模型也可以学会跳过不相关的临时观测。最后，模型还将学会在需要的时候重置隐状态。下面我们将详细讨论各类门控。

# 重置门和更新门

我们首先介绍重置门（reset gate）和更新门（update gate）。我们把它们设计成(0,1)区间中的向量，这样我们就可以进行凸组合。重置门允许我们控制“可能还想记住”的过去状态的数量；更新门将允许我们控制新状态中有多少个是旧状态的副本。

我们从构造这些门控开始。图9.1.1描述了门控循环单元中的重置门和更新门的输入，输入是由当前时间步的输入和前一时间步的隐状态给出。两个门的输出是由使用sigmoid激活函数的两个全连接层给出。

![](images/75e7c5bb27dcea9f4dd8964172be1a1f8b8c61dbb7130924c0cdffa405d7d66c.jpg)  
图9.1.1: 在门控循环单元模型中计算重置门和更新门

我们来看一下门控循环单元的数学表达。对于给定的时间步  $t$ ，假设输入是一个小批量  $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ （样本个

数  $n$  ，输入个数  $d$  )，上一个时间步的隐状态是  $\mathbf{H}_{t - 1}\in \mathbb{R}^{n\times h}$  （隐藏单元个数  $h$  )。那么，重置门  $\mathbf{R}_t\in \mathbb{R}^{n\times h}$  和更新门  $\mathbf{Z}_t\in \mathbb{R}^{n\times h}$  的计算如下所示：

$$
\mathbf {R} _ {t} = \sigma \left(\mathbf {X} _ {t} \mathbf {W} _ {x r} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h r} + \mathbf {b} _ {r}\right),
$$

$$
\mathbf {Z} _ {t} = \sigma \left(\mathbf {X} _ {t} \mathbf {W} _ {x z} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h z} + \mathbf {b} _ {z}\right), \tag {9.1.1}
$$

其中  $\mathbf{W}_{xr},\mathbf{W}_{xz}\in \mathbb{R}^{d\times h}$  和  $\mathbf{W}_{hr},\mathbf{W}_{hz}\in \mathbb{R}^{h\times h}$  是权重参数，  $\mathbf{b}_r,\mathbf{b}_z\in \mathbb{R}^{1\times h}$  是偏置参数。请注意，在求和过程中会触发广播机制（请参阅2.1.3节）。我们使用sigmoid函数（如4.1节中介绍的）将输入值转换到区间(0,1)。

# 候选隐状态

接下来, 让我们将重置门  $\mathbf{R}_t$  与 (8.4.5) 中的常规隐状态更新机制集成, 得到在时间步  $t$  的候选隐状态 (candidate hidden state)  $\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ 。

$$
\tilde {\mathbf {H}} _ {t} = \tanh  \left(\mathbf {X} _ {t} \mathbf {W} _ {x h} + \left(\mathbf {R} _ {t} \odot \mathbf {H} _ {t - 1}\right) \mathbf {W} _ {h h} + \mathbf {b} _ {h}\right), \tag {9.1.2}
$$

其中  $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$  和  $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$  是权重参数， $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$  是偏置项，符号  $\odot$  是Hadamard积（按元素乘积）运算符。在这里，我们使用tanh非线性激活函数来确保候选隐状态中的值保持在区间(-1,1)中。

与 (8.4.5)相比, (9.1.2)中的  $\mathbf{R}_{t}$  和  $\mathbf{H}_{t-1}$  的元素相乘可以减少以往状态的影响。每当重置门  $\mathbf{R}_{t}$  中的项接近 1 时, 我们恢复一个如 (8.4.5)中的普通的循环神经网络。对于重置门  $\mathbf{R}_{t}$  中所有接近 0 的项, 候选隐状态是以  $\mathbf{X}_{t}$  作为输入的多层感知机的结果。因此, 任何预先存在的隐状态都会被重置为默认值。

图9.1.2说明了应用重置门之后的计算流程。

![](images/be01944f24b0af205294d83bbc6de1841eba198cb957ba16bb54fa88faf5a056.jpg)  
图9.1.2: 在门控循环单元模型中计算候选隐状态

# 隐状态

上述的计算结果只是候选隐状态，我们仍然需要结合更新门  $\mathbf{Z}_t$  的效果。这一步确定新的隐状态  $\mathbf{H}_t \in \mathbb{R}^{n \times h}$  在多大程度上来自旧的状态  $\mathbf{H}_{t-1}$  和新的候选状态  $\tilde{\mathbf{H}}_t$  。更新门  $\mathbf{Z}_t$  仅需要在  $\mathbf{H}_{t-1}$  和  $\tilde{\mathbf{H}}_t$  之间进行按元素的凸组合就可以实现这个目标。这就得出了门控循环单元的最终更新公式：

$$
\mathbf {H} _ {t} = \mathbf {Z} _ {t} \odot \mathbf {H} _ {t - 1} + (1 - \mathbf {Z} _ {t}) \odot \tilde {\mathbf {H}} _ {t}. \tag {9.1.3}
$$

每当更新门  $\mathbf{Z}_{t}$  接近 1 时, 模型就倾向只保留旧状态。此时, 来自  $\mathbf{X}_{t}$  的信息基本上被忽略, 从而有效地跳过了依赖链条中的时间步  $t$  。相反, 当  $\mathbf{Z}_{t}$  接近 0 时, 新的隐状态  $\mathbf{H}_{t}$  就会接近候选隐状态  $\tilde{\mathbf{H}}_{t}$  。这些设计可以帮助我们处理循环神经网络中的梯度消失问题, 并更好地捕获时间步距离很长的序列的依赖关系。例如, 如果整个子序列的所有时间步的更新门都接近于 1 , 则无论序列的长度如何, 在序列起始时间步的旧隐状态都将很容易保留并传递到序列结束。

![](images/16a5a942791e08d35d42b584e883610d149c6b06ae03d9ffd29cc1491693f9c2.jpg)  
图9.1.3说明了更新门起作用后的计算流。  
图9.1.3: 计算门控循环单元模型中的隐状态

总之，门控循环单元具有以下两个显著特征：

- 重置门有助于捕获序列中的短期依赖关系；  
- 更新门有助于捕获序列中的长期依赖关系。

# 9.1.2 从零开始实现

为了更好地理解门控循环单元模型，我们从零开始实现它。首先，我们读取8.5节中使用的时间机器数据集：

```python
import torch  
from torch import nn  
from d2l import torch as d2l
```

(continues on next page)

```python
batch_size, num_steps = 32, 35  
train_iter, vocab = d2l.load_data_time-machine(batch_size, num_steps)
```

# 初始化模型参数

下一步是初始化模型参数。我们从标准差为0.01的高斯分布中提取权重，并将偏置项设为0，超参数num_hiddens定义隐藏单元的数量，实例化与更新门、重置门、候选隐状态和输出层相关的所有权重和偏置。

```python
def get.params(vocab_size, num_hiddens, device):
    num Inputs = num Outputs = vocab_size
    def normal(shape):
        return torch rand(size=shape, device=device)*0.01
    def three():
        return (normal((num Inputs, num_hiddens)), 
                    normal((num_hiddens, num_hiddens)), 
                    torch.zeros(num_hiddens, device=device))
    W_xz, W_hz, b_z = three() # 更新门参数
    W_xr, W_hr, b_r = three() # 重置门参数
    W_xh, W_hh, b_h = three() # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        paramrequires_grad_(True)
    return params
```

# 定义模型

现在我们将定义隐状态的初始化函数init_gru_state。与8.5节中定义的init_rnn_state函数一样，此函数返回一个形状为（批量大小，隐藏单元个数）的张量，张量的值全部为零。

```python
def init_gru_state(batch_size, num_hiddens, device): return (torch.zeros((batch_size, num_hiddens), device=device), )
```

现在我们准备定义门控循环单元模型，模型的架构与基本的循环神经网络单元是相同的，只是权重更新公式更为复杂。

```python
def gru(input, state, params): W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params H, = state outputs = [] for X in inputs: Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z) R = torchsigmoid((X @ W_xr) + (H @ W_hr) + b_r) H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h) H = Z * H + (1 - Z) * H_tilda Y = H @ W_hq + b_q outputs.append(Y) return torch.cat(output, dim=0), (H,)
```

# 训练与预测

训练和预测的工作方式与8.5节完全相同。训练结束后，我们分别打印输出训练集的困惑度，以及前缀“traveler”和“traveler”的预测序列上的困惑度。

```lua
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get.params,
init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```txt
perplexity 1.1, 19911.5 tokens/sec onuda:0 time traveller firenis i heidfile soak at i jomer and sugard are travelleryou can show black is white by argument said filby
```

![](images/2eaf1cc3a2e282e2f24a07472a9ee4f575ec1e57858a0acd1f80e21e5e92182f.jpg)

# 9.1.3 简洁实现

高级API包含了前文介绍的所有配置细节，所以我们可以直接实例化门控循环单元模型。这段代码的运行速度要快得多，因为它使用的是编译好的运算符而不是Python来处理之前阐述的许多细节。

```txt
num Inputs = vocab_size  
gru_layer = nn.GRU(num_entries, num_hiddens)  
model = d21.RNNModel(gru_layer, len(vocab))  
model = model.to(device)  
d21.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```txt
perplexity 1.0, 109423.8 tokens/sec onuda:0 time traveller you can show black is white by argument said filby traveller with a slight accession ofcheerfulness really thi
```

![](images/d7d0825f67a41f2965c0ff730ac3188f50997260263a95ac48214a805bb8243d.jpg)

# 小结

- 门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。  
- 重置门有助于捕获序列中的短期依赖关系。  
- 更新门有助于捕获序列中的长期依赖关系。  
- 重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。

# 练习

1. 假设我们只想使用时间步  $t'$  的输入来预测时间步  $t > t'$  的输出。对于每个时间步，重置门和更新门的最佳值是什么？  
2. 调整和分析超参数对运行时间、困惑度和输出顺序的影响。  
3. 比较rnn.RNN和rnn.GRU的不同实现对运行时间、困惑度和输出字符串的影响。  
4. 如果仅仅实现门控循环单元的一部分，例如，只有一个重置门或一个更新门会怎样？

Discussions109

# 9.2 长短期记忆网络（LSTM）

长期以来，隐变量模型存在着长期信息保存和短期输入缺失的问题。解决这一问题的最早方法之一是长短期存储器（long short-term memory, LSTM）(Hochreiter and Schmidhuber, 1997)。它有许多与门控循环单元（9.1节）一样的属性。有趣的是，长短期记忆网络的设计比门控循环单元稍微复杂一些，却比门控循环单元早诞生了近20年。

# 9.2.1 门控记忆元

可以说，长短期记忆网络的设计灵感来自于计算机的逻辑门。长短期记忆网络引入了记忆元（memory cell），或简称为单元（cell）。有些文献认为记忆元是隐状态的一种特殊类型，它们与隐状态具有相同的形状，其设计目的是用于记录附加的信息。为了控制记忆元，我们需要许多门。其中一个门用来从单元中输出条目，我们将其称为输出门（output gate）。另外一个门用来决定何时将数据读入单元，我们将其称为输入门（input gate）。我们还需要一种机制来重置单元的内容，由遗忘门（forget gate）来管理，这种设计的动机与门控循环单元相同，能够通过专用机制决定什么时候记忆或忽略隐状态中的输入。让我们看看这在实践中是如何运作的。

# 输入门、忘记门和输出门

就如在门控循环单元中一样，当前时间步的输入和前一个时间步的隐状态作为数据送入长短期记忆网络的门中，如图9.2.1所示。它们由三个具有sigmoid激活函数的全连接层处理，以计算输入门、遗忘门和输出门的值。因此，这三个门的值都在(0,1)的范围内。

![](images/ad3806fd1fe4a31a47330f5a5568050b2ebd72fb6af67b4817e695b9713ea19e.jpg)  
图9.2.1: 长短期记忆模型中的输入门、遗忘门和输出门

我们来细化一下长短期记忆网络的数学表达。假设有  $h$  个隐藏单元，批量大小为  $n$  ，输入数为  $d$  。因此，输入为  $\mathbf{X}_t \in \mathbb{R}^{n \times d}$  ，前一时间步的隐状态为  $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$  。相应地，时间步  $t$  的门被定义如下：输入门是  $\mathbf{I}_t \in \mathbb{R}^{n \times h}$  ，遗忘门是  $\mathbf{F}_t \in \mathbb{R}^{n \times h}$  ，输出门是  $\mathbf{O}_t \in \mathbb{R}^{n \times h}$  。它们的计算方法如下：

$$
\mathbf {I} _ {t} = \sigma \left(\mathbf {X} _ {t} \mathbf {W} _ {x i} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h i} + \mathbf {b} _ {i}\right),
$$

$$
\mathbf {F} _ {t} = \sigma \left(\mathbf {X} _ {t} \mathbf {W} _ {x f} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h f} + \mathbf {b} _ {f}\right), \tag {9.2.1}
$$

$$
\mathbf {O} _ {t} = \sigma \left(\mathbf {X} _ {t} \mathbf {W} _ {x o} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h o} + \mathbf {b} _ {o}\right),
$$

其中  $\mathbf{W}_{xi},\mathbf{W}_{xf},\mathbf{W}_{xo}\in \mathbb{R}^{d\times h}$  和  $\mathbf{W}_{hi},\mathbf{W}_{hf},\mathbf{W}_{ho}\in \mathbb{R}^{h\times h}$  是权重参数，  $\mathbf{b}_i,\mathbf{b}_f,\mathbf{b}_o\in \mathbb{R}^{1\times h}$  是偏置参数。

# 候选记忆元

由于还没有指定各种门的操作，所以先介绍候选记忆元（candidate memory cell） $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$ 。它的计算与上面描述的三个门的计算类似，但是使用tanh函数作为激活函数，函数的值范围为  $(-1, 1)$ 。下面导出在时间步  $t$  处的方程：

$$
\tilde {\mathbf {C}} _ {t} = \tanh  \left(\mathbf {X} _ {t} \mathbf {W} _ {x c} + \mathbf {H} _ {t - 1} \mathbf {W} _ {h c} + \mathbf {b} _ {c}\right), \tag {9.2.2}
$$

其中  $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$  和  $\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$  是权重参数， $\mathbf{b}_c \in \mathbb{R}^{1 \times h}$  是偏置参数。

候选记亿元的如图9.2.2所示。

![](images/1fbb3466bfd86aed15c06df37dbc8fc4383e7c60d917628ee421db8b575ba029.jpg)  
图9.2.2: 长短期记忆模型中的候选记元

# 记忆元

在门控循环单元中, 有一种机制来控制输入和遗忘 (或跳过)。类似地, 在长短期记忆网络中, 也有两个门用于这样的目的: 输入门  $\mathbf{I}_{t}$  控制采用多少来自  $\tilde{\mathbf{C}}_{t}$  的新数据, 而遗忘门  $\mathbf{F}_{t}$  控制保留多少过去的记忆元  $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$  的内容。使用按元素乘法, 得出:

$$
\mathbf {C} _ {t} = \mathbf {F} _ {t} \odot \mathbf {C} _ {t - 1} + \mathbf {I} _ {t} \odot \tilde {\mathbf {C}} _ {t}. \tag {9.2.3}
$$

如果遗忘门始终为1且输入门始终为0，则过去的记忆元  $\mathbf{C}_{t-1}$  将随时间被保存并传递到当前时间步。引入这种设计是为了缓解梯度消失问题，并更好地捕获序列中的长距离依赖关系。

这样我们就得到了计算记亿元的流程图，如图9.2.3。

![](images/21c5a9db9020b66d0906fe4dda2af700fa5dd517d69d6d228f7ab21f4df626f5.jpg)  
图9.2.3: 在长短期记忆网络模型中计算记亿元

# 隐状态

最后，我们需要定义如何计算隐状态  $\mathbf{H}_t\in \mathbb{R}^{n\times h}$ ，这就是输出门发挥作用的地方。在长短期记忆网络中，它仅仅是记忆元的tanh的门控版本。这就确保了  $\mathbf{H}_t$  的值始终在区间(-1,1)内：

$$
\mathbf {H} _ {t} = \mathbf {O} _ {t} \odot \tanh  \left(\mathbf {C} _ {t}\right). \tag {9.2.4}
$$

只要输出门接近1，我们就能够有效地将所有记忆信息传递给预测部分，而对于输出门接近0，我们只保留记忆元内的所有信息，而不需要更新隐状态。

![](images/71a08297ff9f2b2bf9a880ab5b63e82e6d095c5e6601112d88dec80f22d75155.jpg)  
图9.2.4提供了数据流的图形化演示。  
图9.2.4: 在长短期记忆模型中计算隐状态

# 9.2.2 从零开始实现

现在，我们从零开始实现长短期记忆网络。与8.5节中的实验相同，我们首先加载时光机器数据集。

```python
import torch   
from torch import nn   
from d2l import torch as d21   
batch_size，num_steps  $= 32$  ，35   
train_iter，vocab  $\equiv$  d21.load_data_time-machine(batch_size，num_steps)
```

# 初始化模型参数

接下来，我们需要定义和初始化模型参数。如前所述，超参数num_hiddens定义隐藏单元的数量。我们按照标准差0.01的高斯分布初始化权重，并将偏置项设为0。

```python
def get LSM params(vocab_size, num_hiddens, device):
    num Inputs = num Outputs = vocab_size
    def normal(shape):
        return torch Randn(size=shape, device=device)*0.01
    def three():
        return (normal((num Inputs, num_hiddens)), 
                    normal((num_hiddens, num_hiddens)), 
                    torch.zeros(num_hiddens, device=device))
    W(xi, W_hi, b_i = three() # 输入门参数
    W_xf, W_hf, b_f = three() # 遗忘门参数
    W_xo, W_ho, b_o = three() # 输出门参数
    W_xc, W_hc, b_c = three() # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W(xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                    b_c, W_hq, b_q]
    for param in params:
        paramrequires_grad_(True)
    return params
```

# 定义模型

在初始化函数中，长短期记忆网络的隐状态需要返回一个额外的记忆元，单元的值为0，形状为（批量大小，隐藏单元数）。因此，我们得到以下的状态初始化。

```python
def init_lstm_state(batch_size, num_hiddens, device): return (torch.zeros((batch_size, num_hiddens), device=device), torch.zeros((batch_size, num_hiddens), device=device))
```

实际模型的定义与我们前面讨论的一样：提供三个门和一个额外的记忆元。请注意，只有隐状态才会传递到输出层，而记忆元  $\mathbf{C}_t$  不直接参与输出计算。

```python
def lstm(input, state, params): [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params (H, C) = state outputs = [] for X in inputs: I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i) F = torchsigmoid((X @ W_xf) + (H @ W_hf) + b_f) O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o) C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c) C = F * C + I * C_tilda H = 0 * torch.tanh(C) Y = (H @ W_hq) + b_q outputs.append(Y) return torch.cat(output, dim=0), (H, C)
```

# 训练和预测

让我们通过实例化8.5节中引入的RNNModelScratch类来训练一个长短期记忆网络，就如我们在9.1节中所做的一样。

```lua
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm.params,
init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```txt
perplexity 1.3, 17736.0 tokens/sec onuda:0 time traveller for so it will leong go it we melenot ir cove i s traveller care be can so i ngrecpely as along the time dime
```

![](images/0f0a88ac71e1bfcb42fa1766034c137f53e4c2b2af794e3a7643b985a8926909.jpg)

# 9.2.3 简洁实现

使用高级API，我们可以直接实例化LSTM模型。高级API封装了前文介绍的所有配置细节。这段代码的运行速度要快得多，因为它使用的是编译好的运算符而不是Python来处理之前阐述的许多细节。

```python
num Inputs = vocab_size  
lstm_layer = nn.LSTM(num Inputs, num_hiddens)  
model = d21.RNNModel(lstm_layer, len(vocab))  
model = model.to(device)  
d21.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```txt
perplexity 1.1, 234815.0 tokens/sec onuda:0 time traveller for so it will be convenient to speak of himwas e travelleryou can show black is white by argument said filby
```

![](images/41c0d8bd0b6c5694765084cb17ca753bd118af28e4bf7848bef37ee9056f60ca.jpg)

长短期记忆网络是典型的具有重要状态控制的隐变量自回归模型。多年来已经提出了其许多变体，例如，多层、残差连接、不同类型的正则化。然而，由于序列的长距离依赖性，训练长短期记忆网络和其他序列模型（例如门控循环单元）的成本是相当高的。在后面的内容中，我们将讲述更高级的替代模型，如Transformer。

# 小结

- 长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。  
- 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。  
- 长短期记忆网络可以缓解梯度消失和梯度爆炸。

# 练习

1. 调整和分析超参数对运行时间、困惑度和输出顺序的影响。  
2. 如何更改模型以生成适当的单词，而不是字符序列？  
3. 在给定隐藏层维度的情况下，比较门控循环单元、长短期记忆网络和常规循环神经网络的计算成本。要特别注意训练和推断成本。  
4. 既然候选记忆元通过使用tanh函数来确保值范围在  $(-1,1)$  之间，那么为什么隐状态需要再次使用tanh函数来确保输出值范围在  $(-1,1)$  之间呢？  
5. 实现一个能够基于时间序列进行预测而不是基于字符序列进行预测的长短期记忆网络模型。

Discussions<sup>110</sup>

# 9.3 深度循环神经网络

到目前为止，我们只讨论了具有一个单向隐藏层的循环神经网络。其中，隐变量和观测值与具体的函数形式的交互方式是相当随意的。只要交互类型建模具有足够的灵活性，这就不是一个大问题。然而，对一个单层来说，这可能具有相当的挑战性。之前在线性模型中，我们通过添加更多的层来解决这个问题。而在循环神经网络中，我们首先需要确定如何添加更多的层，以及在哪里添加额外的非线性，因此这个问题有点棘手。

事实上，我们可以将多层循环神经网络堆叠在一起，通过对几个简单层的组合，产生了一个灵活的机制。特别是，数据可能与不同层的堆叠有关。例如，我们可能希望保持有关金融市场状况（熊市或牛市）的宏观数据可用，而微观数据只记录较短期的时间动态。

图9.3.1描述了一个具有  $L$  个隐藏层的深度循环神经网络，每个隐状态都连续地传递到当前层的下一个时间步和下一层的当前时间步。

![](images/4e4993d9d53051decf97f9da2cc80fc8512e58626c2baf587dbb28f6de3fbd2a.jpg)  
图9.3.1: 深度循环神经网络结构

# 9.3.1 函数依赖关系

我们可以将深度架构中的函数依赖关系形式化，这个架构是由图9.3.1中描述了  $L$  个隐藏层构成。后续的讨论主要集中在经典的循环神经网络模型上，但是这些讨论也适应于其他序列模型。

假设在时间步  $t$  有一个小批量的输入数据  $\mathbf{X}_t \in \mathbb{R}^{n \times d}$  （样本数：  $n$ ，每个样本中的输入数：  $d$ ）。同时，将  $l^{\text{th}}$  隐藏层  $(l = 1, \dots, L)$  的隐状态设为  $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$  （隐藏单元数：  $h$ ），输出层变量设为  $\mathbf{O}_t \in \mathbb{R}^{n \times q}$  （输出数：  $q$ ）。设置  $\mathbf{H}_t^{(0)} = \mathbf{X}_t$ ，第  $l$  个隐藏层的隐状态使用激活函数  $\phi_l$ ，则：

$$
\mathbf {H} _ {t} ^ {(l)} = \phi_ {l} \left(\mathbf {H} _ {t} ^ {(l - 1)} \mathbf {W} _ {x h} ^ {(l)} + \mathbf {H} _ {t - 1} ^ {(l)} \mathbf {W} _ {h h} ^ {(l)} + \mathbf {b} _ {h} ^ {(l)}\right), \tag {9.3.1}
$$

其中，权重  $\mathbf{W}_{xh}^{(l)}\in \mathbb{R}^{h\times h}$  ，  $\mathbf{W}_{hh}^{(l)}\in \mathbb{R}^{h\times h}$  和偏置  $\mathbf{b}_h^{(l)}\in \mathbb{R}^{1\times h}$  都是第  $l$  个隐藏层的模型参数。

最后，输出层的计算仅基于第  $l$  个隐藏层最终的隐状态：

$$
\mathbf {O} _ {t} = \mathbf {H} _ {t} ^ {(L)} \mathbf {W} _ {h q} + \mathbf {b} _ {q}, \tag {9.3.2}
$$

其中，权重  $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$  和偏置  $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$  都是输出层的模型参数。

与多层感知机一样，隐藏层数目  $L$  和隐藏单元数目  $h$  都是超参数。也就是说，它们可以由我们调整的。另外，用门控循环单元或长短期记忆网络的隐状态来代替 (9.3.1) 中的隐状态进行计算，可以很容易地得到深度门控循环神经网络或深度长短期记忆神经网络。

# 9.3.2 简洁实现

实现多层循环神经网络所需的许多逻辑细节在高级API中都是现成的。简单起见，我们仅示范使用此类内置函数的实现方式。以长短期记忆网络模型为例，该代码与之前在9.2节中使用的代码非常相似，实际上唯一的区别是我们指定了层的数量，而不是使用单一层这个默认值。像往常一样，我们从加载数据集开始。

```python
import torch   
from torch import nn   
from d2l import torch as d21   
batch_size，num_steps  $= 32$  ，35   
train_iter，vocab  $\equiv$  d21.load_data_time-machine(batch_size，num_steps)
```

像选择超参数这类架构决策也跟9.2节中的决策非常相似。因为我们有不同的词元，所以输入和输出都选择相同数量，即vocab_size。隐藏单元的数量仍然是256。唯一的区别是，我们现在通过num_layers的值来设定隐藏层数。

```python
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2  
num Inputs = vocab_size  
device = d2l.trygpu()  
lstm_layer = nn.LSTM(num Inputs, num_hiddens, num_layers)  
model = d2l.RNNModel(lstm_layer, len(vocab))  
model = model.to(device)
```

# 9.3.3 训练与预测

由于使用了长短期记忆网络模型来实例化两个层，因此训练速度被大大降低了。

```txt
num_epochs, lr = 500, 2  
d21.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
```

```txt
perplexity 1.0, 186005.7 tokens/sec onuda:0 time traveller for so it will be convenient to speak of himwas e travelleryou can show black is white by argument said filby
```

![](images/643ac7ab5f4148e71e43b3bb2604a8319953e2ca4b7900dbcbb1c15d4d44eaca.jpg)

# 小结

- 在深度循环神经网络中，隐状态的信息被传递到当前层的下一时间步和下一层的当前时间步。  
- 有许多不同风格的深度循环神经网络，如长短期记忆网络、门控循环单元、或经典循环神经网络。这些模型在深度学习框架的高级API中都有涵盖。  
- 总体而言，深度循环神经网络需要大量的调参（如学习率和修剪）来确保合适的收敛，模型的初始化也需要谨慎。

# 练习

1. 基于我们在8.5节中讨论的单层实现，尝试从零开始实现两层循环神经网络。  
2. 在本节训练模型中，比较使用门控循环单元替换长短期记忆网络后模型的精确度和训练速度。  
3. 如果增加训练数据，能够将困惑度降到多低？  
4. 在为文本建模时，是否可以将不同作者的源数据合并？有何优劣呢？

```twig
Discussions<sup>111</sup>
```

```txt
111 https://discuss.d2l.ai/t/2770
```

# 9.4 双向循环神经网络

在序列学习中，我们以往假设的目标是：在给定观测的情况下（例如，在时间序列的上下文中或在语言模型的上下文中），对下一个输出进行建模。虽然这是一个典型情景，但不是唯一的。还可能发生什么其它的情况呢？我们考虑以下三个在文本序列中填空的任务。

·我____。  
·我____饿了。  
- 我____饿了，我可以吃半头猪。

根据可获得的信息量, 我们可以用不同的词填空, 如 “很高兴” (“happy”)、“不” (“not”) 和 “非常” (“very”). 很明显, 每个短语的 “下文” 传达了重要信息 (如果有的话), 而这些信息关乎到选择哪个词来填空, 所以无法利用这一点的序列模型将在相关任务上表现不佳。例如, 如果要做好命名实体识别 (例如, 识别 “Green” 指的是 “格林先生” 还是绿色), 不同长度的上下文范围重要性是相同的。为了获得一些解决问题的灵感, 让我们先迁回到概率图模型。

# 9.4.1 隐马尔可夫模型中的动态规划

这一小节是用来说明动态规划问题的，具体的技术细节对于理解深度学习模型并不重要，但它有助于我们思考为什么要使用深度学习，以及为什么要选择特定的架构。

如果我们想用概率图模型来解决这个问题，可以设计一个隐变量模型：在任意时间步  $t$  ，假设存在某个隐变量  $h_t$  通过概率  $P(x_{t} \mid h_{t})$  控制我们观测到的  $x_{t}$  。此外，任何  $h_t \rightarrow h_{t+1}$  转移都是由一些状态转移概率  $P(h_{t+1} \mid h_t)$  给出。这个概率图模型就是一个隐马尔可夫模型（hidden Markov model，HMM），如图9.4.1所示。

![](images/d980fab7c81dea248c1fc0f05a139da06e7b5afe4e00203b521ca798419d9883.jpg)  
图9.4.1: 隐马尔可夫模型

因此，对于有  $T$  个观测值的序列，我们在观测状态和隐状态上具有以下联合概率分布：

$$
P \left(x _ {1}, \dots , x _ {T}, h _ {1}, \dots , h _ {T}\right) = \prod_ {t = 1} ^ {T} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right), \text {w h e r e} P \left(h _ {1} \mid h _ {0}\right) = P \left(h _ {1}\right). \tag {9.4.1}
$$

现在，假设我们观测到所有的  $x_{i}$ ，除了  $x_{j}$ ，并且我们的目标是计算  $P(x_{j} \mid x_{-j})$ ，其中  $x_{-j} = (x_{1}, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$ 。由于  $P(x_{j} \mid x_{-j})$  中没有隐变量，因此我们考虑对  $h_{1}, \ldots, h_{T}$  选择构成的所有可能的组合进行求和。如果任何  $h_{i}$  可以接受  $k$  个不同的值（有限的状态数），这意味着我们需要对  $k^{T}$  个项求和，这个任务显然难于登天。幸运的是，有个巧妙的解决方案：动态规划（dynamic programming）。

要了解动态规划的工作方式，我们考虑对隐变量  $h_1, \ldots, h_T$  的依次求和。根据(9.4.1)，将得出：

$$
\begin{array}{l} P (x _ {1}, \dots , x _ {T}) \\ = \sum_ {h _ {1}, \dots , h _ {T}} P \left(x _ {1}, \dots , x _ {T}, h _ {1}, \dots , h _ {T}\right) \\ = \sum_ {h _ {1}, \dots , h _ {T}} \prod_ {t = 1} ^ {T} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right) \\ = \sum_ {h _ {2}, \dots , h _ {T}} \underbrace {\left[ \sum_ {h _ {1}} P \left(h _ {1}\right) P \left(x _ {1} \mid h _ {1}\right) P \left(h _ {2} \mid h _ {1}\right) \right]} _ {\pi_ {2} \left(h _ {2}\right) \stackrel {\text {d e f}} {=} 3} P \left(x _ {2} \mid h _ {2}\right) \prod_ {t = 3} ^ {T} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right) \tag {9.4.2} \\ = \sum_ {h _ {3}, \dots , h _ {T}} \underbrace {\left[ \sum_ {h _ {2}} \pi_ {2} (h _ {2}) P \left(x _ {2} \mid h _ {2}\right) P \left(h _ {3} \mid h _ {2}\right) \right]} _ {\pi_ {3} \left(h _ {3}\right) \stackrel {{\mathrm {d e f}}} {{=}}} P \left(x _ {3} \mid h _ {3}\right) \prod_ {t = 4} ^ {T} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right) \\ = \dots \\ = \sum_ {h _ {T}} \pi_ {T} \left(h _ {T}\right) P \left(x _ {T} \mid h _ {T}\right). \\ \end{array}
$$

通常，我们将前向递归（forward recursion）写为：

$$
\pi_ {t + 1} \left(h _ {t + 1}\right) = \sum_ {h _ {t}} \pi_ {t} \left(h _ {t}\right) P \left(x _ {t} \mid h _ {t}\right) P \left(h _ {t + 1} \mid h _ {t}\right). \tag {9.4.3}
$$

递归被初始化为  $\pi_1(h_1) = P(h_1)$  。符号简化，也可以写成  $\pi_{t + 1} = f(\pi_t,x_t)$  ，其中  $f$  是一些可学习的函数。这看起来就像我们在循环神经网络中讨论的隐变量模型中的更新方程。

与前向递归一样，我们也可以使用后向递归对同一组隐变量求和。这将得到：

$$
\begin{array}{l} P (x _ {1}, \dots , x _ {T}) \\ = \sum_ {h _ {1}, \dots , h _ {T}} P \left(x _ {1}, \dots , x _ {T}, h _ {1}, \dots , h _ {T}\right) \\ = \sum_ {h _ {1}, \dots , h _ {T}} \prod_ {t = 1} ^ {T - 1} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right) \cdot P \left(h _ {T} \mid h _ {T - 1}\right) P \left(x _ {T} \mid h _ {T}\right) \\ = \sum_ {h _ {1}, \dots , h _ {T - 1}} \prod_ {t = 1} ^ {T - 1} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right) \cdot \underbrace {\left[ \sum_ {h _ {T}} P \left(h _ {T} \mid h _ {T - 1}\right) P \left(x _ {T} \mid h _ {T}\right) \right]} _ {\rho_ {T - 1} \left(h _ {T - 1}\right) \stackrel {\text {d e f}} {=} 0} \tag {9.4.4} \\ = \sum_ {h _ {1}, \dots , h _ {T - 2}} \prod_ {t = 1} ^ {T - 2} P (h _ {t} \mid h _ {t - 1}) P (x _ {t} \mid h _ {t}) \cdot \underbrace {\left[ \sum_ {h _ {T - 1}} P (h _ {T - 1} \mid h _ {T - 2}) P (x _ {T - 1} \mid h _ {T - 1}) \rho_ {T - 1} (h _ {T - 1}) \right]} _ {\rho_ {T - 2} (h _ {T - 2}) \stackrel {\mathrm {d e f}} {=}} \\ = \dots \\ = \sum_ {h _ {1}} P \left(h _ {1}\right) P \left(x _ {1} \mid h _ {1}\right) \rho_ {1} \left(h _ {1}\right). \\ \end{array}
$$

# 9.4.双向循环神经网络

因此，我们可以将后向递归（backward recursion）写为：

$$
\rho_ {t - 1} \left(h _ {t - 1}\right) = \sum_ {h _ {t}} P \left(h _ {t} \mid h _ {t - 1}\right) P \left(x _ {t} \mid h _ {t}\right) \rho_ {t} \left(h _ {t}\right), \tag {9.4.5}
$$

初始化  $\rho_{T}(h_{T}) = 1$  。前向和后向递归都允许我们对  $T$  个隐变量在  $\mathcal{O}(kT)$  （线性而不是指数）时间内对  $(h_1,\dots ,h_T)$  的所有值求和。这是使用图模型进行概率推理的巨大好处之一。它也是通用消息传递算法(Aji and McEliece, 2000)的一个非常特殊的例子。结合前向和后向递归，我们能够计算

$$
P \left(x _ {j} \mid x _ {- j}\right) \propto \sum_ {h _ {j}} \pi_ {j} \left(h _ {j}\right) \rho_ {j} \left(h _ {j}\right) P \left(x _ {j} \mid h _ {j}\right). \tag {9.4.6}
$$

因为符号简化的需要，后向递归也可以写为  $\rho_{t - 1} = g(\rho_t,x_t)$  ，其中  $g$  是一个可以学习的函数。同样，这看起来非常像一个更新方程，只是不像我们在循环神经网络中看到的那样前向运算，而是后向计算。事实上，知道未来数据何时可用对隐马尔可夫模型是有益的。信号处理学家将是否知道未来观测这两种情况区分为内插和外推，有关更多详细信息，请参阅 (Doucet et al., 2001)。

# 9.4.2 双向模型

如果我们希望在循环神经网络中拥有一种机制，使之能够提供与隐马尔可夫模型类似的前瞻能力，我们就需要修改循环神经网络的设计。幸运的是，这在概念上很容易，只需要增加一个“从最后一个词元开始从后向前运行”的循环神经网络，而不是只有一个在前向模式下“从第一个词元开始运行”的循环神经网络。双向循环神经网络（bidirectional RNNs）添加了反向传递信息的隐藏层，以便更灵活地处理此类信息。图9.4.2描述了具有单个隐藏层的双向循环神经网络的架构。

![](images/a18098c2b78a8be34e7ae55b527f19f894c14fc3c280f3747ffdbe1adeb0c422.jpg)  
图9.4.2: 双向循环神经网络架构

事实上，这与隐马尔可夫模型中的动态规划的前向和后向递归没有太大区别。其主要区别是，在隐马尔可夫模型中的方程具有特定的统计意义。双向循环神经网络没有这样容易理解的解释，我们只能把它们当作通用的、可学习的函数。这种转变集中体现了现代深度网络的设计原则：首先使用经典统计模型的函数依赖类型，然后将其参数化为通用形式。

# 定义

双向循环神经网络是由 (Schuster and Paliwal, 1997) 提出的，关于各种架构的详细讨论请参阅 (Graves and Schmidhuber, 2005)。让我们看看这样一个网络的细节。

对于任意时间步  $t$ , 给定一个小批量的输入数据  $\mathbf{X}_t \in \mathbb{R}^{n \times d}$  (样本数  $n$ , 每个示例中的输入数  $d$ ), 并且令隐藏层激活函数为  $\phi$  。在双向架构中, 我们设该时间步的前向和反向隐状态分别为  $\vec{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$  和  $\vec{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ , 其中  $h$  是隐藏单元的数目。前向和反向隐状态的更新如下:

$$
\overrightarrow {\mathbf {H}} _ {t} = \phi \left(\mathbf {X} _ {t} \mathbf {W} _ {x h} ^ {(f)} + \overrightarrow {\mathbf {H}} _ {t - 1} \mathbf {W} _ {h h} ^ {(f)} + \mathbf {b} _ {h} ^ {(f)}\right), \tag {9.4.7}
$$

$$
\overleftarrow {\mathbf {H}} _ {t} = \phi \left(\mathbf {X} _ {t} \mathbf {W} _ {x h} ^ {(b)} + \overleftarrow {\mathbf {H}} _ {t + 1} \mathbf {W} _ {h h} ^ {(b)} + \mathbf {b} _ {h} ^ {(b)}\right),
$$

其中，权重  $\mathbf{W}_{xh}^{(f)}\in \mathbb{R}^{d\times h},\mathbf{W}_{hh}^{(f)}\in \mathbb{R}^{h\times h},\mathbf{W}_{xh}^{(b)}\in \mathbb{R}^{d\times h},\mathbf{W}_{hh}^{(b)}\in \mathbb{R}^{h\times h}$  和偏置  $\mathbf{b}_h^{(f)}\in \mathbb{R}^{1\times h},\mathbf{b}_h^{(b)}\in \mathbb{R}^{1\times h}$  都是模型参数。

接下来，将前向隐状态  $\overline{\mathbf{H}}_t$  和反向隐状态  $\widehat{\mathbf{H}}_t$  连接起来，获得需要送入输出层的隐状态  $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$  。在具有多个隐藏层的深度双向循环神经网络中，该信息作为输入传递到下一个双向层。最后，输出层计算得到的输出为  $\mathbf{O}_t \in \mathbb{R}^{n \times q}$  （ $q$  是输出单元的数目）：

$$
\mathbf {O} _ {t} = \mathbf {H} _ {t} \mathbf {W} _ {h q} + \mathbf {b} _ {q}. \tag {9.4.8}
$$

这里，权重矩阵  $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$  和偏置  $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$  是输出层的模型参数。事实上，这两个方向可以拥有不同数量的隐藏单元。

# 模型的计算代价及其应用

双向循环神经网络的一个关键特性是：使用来自序列两端的信息来估计输出。也就是说，我们使用来自过去和未来的观测信息来预测当前的观测。但是在对下一个词元进行预测的情况中，这样的模型并不是我们所需的。因为在预测下一个词元时，我们终究无法知道下一个词元的下文是什么，所以将不会得到很好的精度。具体地说，在训练期间，我们能够利用过去和未来的数据来估计现在空缺的词；而在测试期间，我们只有过去的数据，因此精度将会很差。下面的实验将说明这一点。

另一个严重问题是，双向循环神经网络的计算速度非常慢。其主要原因是网络的前向传播需要在双向层中进行前向和后向递归，并且网络的反向传播还依赖于前向传播的结果。因此，梯度求解将有一个非常长的链。

双向层的使用在实践中非常少，并且仅仅应用于部分场合。例如，填充缺失的单词、词元注释（例如，用于命名实体识别）以及作为序列处理流水线中的一个步骤对序列进行编码（例如，用于机器翻译）。在14.8节和15.2节中，我们将介绍如何使用双向循环神经网络编码文本序列。

# 9.4.3 双向循环神经网络的错误应用

由于双向循环神经网络使用了过去的和未来的数据，所以我们不能盲目地将这一语言模型应用于任何预测任务。尽管模型产出的困惑度是合理的，该模型预测未来词元的能力却可能存在严重缺陷。我们用下面的示例代码引以为戒，以防在错误的环境中使用它们。

```python
import torch  
from torch import nn  
from d2l import torch as d2l
```

# 加载数据

```python
batch_size, num_steps, device = 32, 35, d21.trygpu()  
train_iter, vocab = d21.load_data_time-machine(batch_size, num_steps)  
# 通过设置“bidirective=True”来定义双向LSTM模型  
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2  
num Inputs = vocab_size  
lstm_layer = nn.LSTM(num Inputs, num_hiddens, num_layers, bidirectional=True)  
model = d21.RNNModel(lstm_layer, len(vocab))  
model = model.to(device)  
# 训练模型  
num_epochs, lr = 500, 1  
d21.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```txt
perplexity 1.1, 131129.2 tokens/sec on Juda:0 time travellerererererererererererererererererererererererererererererererererererererererererererererererererererererer
```

![](images/c9de8080faf4ae9f69695ee281e6ab3e206796c20dd5d7665004317b77099cf4.jpg)

上述结果显然令人瞠目结舌。关于如何更有效地使用双向循环神经网络的讨论，请参阅15.2节中的情感分类应用。

# 小结

- 在双向循环神经网络中，每个时间步的隐状态由当前时间步的前后数据同时决定。  
- 双向循环神经网络与概率图模型中的“前向-后向”算法具有相似性。  
- 双向循环神经网络主要用于序列编码和给定双向上下文的观测估计。  
- 由于梯度链更长，因此双向循环神经网络的训练代价非常高。

# 练习

1. 如果不同方向使用不同数量的隐藏单位,  $\mathbf{H}_{\mathrm{t}}$  的形状会发生怎样的变化?  
2. 设计一个具有多个隐藏层的双向循环神经网络。  
3. 在自然语言中一词多义很常见。例如，“bank”一词在不同的上下文“i went to the bank to deposit cash”和“i went to the bank to sit down”中有不同的含义。如何设计一个神经网络模型，使其在给定上下文序列和单词的情况下，返回该单词在此上下文中的向量表示？哪种类型的神经网络架构更适合处理一词多义？

Discussions<sup>112</sup>

# 9.5 机器翻译与数据集

语言模型是自然语言处理的关键，而机器翻译是语言模型最成功的基准测试。因为机器翻译正是将输入序列转换成输出序列的序列转换模型（sequence transduction）的核心问题。序列转换模型在各类现代人工智能应用中发挥着至关重要的作用，因此我们将其做为本章剩余部分和10节的重点。为此，本节将介绍机器翻译问题及其后文需要使用的数据集。

机器翻译（machine translation）指的是将序列从一种语言自动翻译成另一种语言。事实上，这个研究领域可以追溯到数字计算机发明后不久的20世纪40年代，特别是在第二次世界大战中使用计算机破解语言编码。几十年来，在使用神经网络进行端到端学习的兴起之前，统计学方法在这一领域一直占据主导地位 (Brown et al., 1990, Brown et al., 1988)。因为统计机器翻译（statistical machine translation）涉及了翻译模型和语言模型等组成部分的统计分析，因此基于神经网络的方法通常被称为神经机器翻译（neural machine translation），用于将两种翻译模型区分开来。

本书的关注点是神经网络机器翻译方法，强调的是端到端的学习。与8.3节中的语料库是单一语言的语言模型问题存在不同，机器翻译的数据集是由源语言和目标语言的文本序列对组成的。因此，我们需要一种完全不同的方法来预处理机器翻译数据集，而不是复用语言模型的预处理程序。下面，我们看一下如何将预处理后的数据加载到小批量中用于训练。

```txt
import os  
import torch  
from d2l import torch as d2l
```

# 9.5.1 下载和预处理数据集

首先，下载一个由Tatoeba项目的双语句子对113组成的“英一法”数据集，数据集中的每一行都是制表符分隔的文本序列对，序列对由英文文本序列和翻译后的法语文本序列组成。请注意，每个文本序列可以是一个句子，也可以是包含多个句子的一个段落。在这个将英语翻译成法语的机器翻译问题中，英语是源语言（source language），法语是目标语言（target language）。

```python
@save
d21.DATA_HUB['fra-eng'] = (d21.DATA_URL + 'fra-eng.zip',
                     '94646ad1522d915e7b0f9296181140edcf86a4f5')
#save
def read_data_nmt():
    '''载入“英语一法语”数据集''''
    data_dir = d21.download.extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
                    encoding='utf-8') as f:
        return f.read()
raw_text = read_data_nmt()
print(raw_text[:75])
```

```txt
Downloading ./data/fra-eng.zip from http://d2l-data.s3-accelerate.amazon.com/fra-eng.zip...  
Go. Va!  
Hi. Salut!  
Run! Cours!  
Run! Courez!  
Who? Qui?  
Wow! Za alors!
```

下载数据集后，原始文本数据需要经过几个预处理步骤。例如，我们用空格代替不间断空格（non-breaking space），使用小写字母替换大写字母，并在单词和标点符号之间插入空格。

```batch
@save defpreprocess_nmt(text):
```

(continues on next page)

```python
```
```
def no_space(char, prev_char):
    return char in set('', ' ', ' )
# 使用空格替换不间断空格
# 使用小写字母替换大写字母
text = text.replace('\\u202f', ' ', '.').replace('\\xa0', ' ', '.').lower()
# 在单词和标点符号之间插入空格
out = [' ', + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
return '.join(out)
text = preprocess_nmt(raw_text)
print(text[:80])
```

```txt
go . va!  
hi . salute!  
run ! courses!  
run ! courez !  
who ? qui ?  
wow !ça alors !
```

# 9.5.2 词元化

与8.3节中的字符级词元化不同，在机器翻译中，我们更喜欢单词级词元化（最先进的模型可能使用更高级的词元化技术）。下面的tokenize_nmt函数对前numexamples个文本序列对进行词元，其中每个词元要么是一个词，要么是一个标点符号。此函数返回两个词元列表：source和target：source[i]是源语言（这里是英语）第i个文本序列的词元列表，target[i]是目标语言（这里是法语）第i个文本序列的词元列表。

```txt
@save   
deftokenizer_nmt(text，numexamples  $\equiv$  None)： """词元化“英语一法语”数据数据集"" source,target  $= []$  ，[] fori,line in enumerate(text.split('n'))： if numexamplesand  $\mathrm{i}>$  numexamples: break parts  $=$  line.split('t') iflen-parts）  $= = 2$  ： source.appendparts[0].split（‘）） target.appendparts[1].split（'）） return source，target
```

(continues on next page)

```txt
source, target = tokenize_nmt(text)  
source[:6], target[:6]
```

```latex
[ \begin{array}{l} \left[\text{'go '}, \text{'. '}\right], \\ \left[\text{'hi '}, \text{'. '}\right], \\ \left[\text{'run '}, \text{'! '}\right], \\ \left[\text{'run '}, \text{'! '}\right], \\ \left[\text{'who '}, \text{'? '}\right], \\ \left[\text{'wow '}, \text{'! '}\right], \\ \left[\text{'va '}, \text{'! '}\right], \\ \left[\text{'salut '}, \text{'! '}\right], \\ \left[\text{'cours '}, \text{'! '}\right], \\ \left[\text{'courez '}, \text{'! '}\right], \\ \left[\text{'qui '}, \text{'? '}\right], \\ \left[\text{'ca '}, \text{'alors '}, \text{'! '}\right] \end{array} ]
```

让我们绘制每个文本序列所包含的词元数量的直方图。在这个简单的“英一法”数据集中，大多数文本序列的词元数量少于20个。

```txt
@save   
def show_list_len_pair歷史(legend,xlabel,ylabel,xlist,ylist): ""绘制列表长度对的直方图"""" d21.set_figsize() _,_,patches  $=$  d21plt.hist( [[len(1)for1inxlist],[len(1)for1in ylist]]） d21pltxlabel(xlabel) d21pltylabel(ylabel) for patch in patches[1].patches: patch.set_hatch('/') d21pltlegend(legend)   
show_list_len_pair歷史(['source','target'],'# tokens per sequence', count'，source,target);
```

![](images/f653d7d74092ace1741978af8dd6b490fe79d1417c3473bca5d905b739ffbb5f.jpg)

# 9.5.3 词表

由于机器翻译数据集由语言对组成，因此我们可以分别为源语言和目标语言构建两个词表。使用单词级词元化时，词表大小将明显大于使用字符级词元化时的词表大小。为了缓解这一问题，这里我们将出现次数少于2次的低频率词元视为相同的未知（“<unk>”）词元。除此之外，我们还指定了额外的特定词元，例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”），以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）。这些特殊词元在自然语言处理任务中比较常用。

```python
src_vocab = d21.Vocab.source, min_freq=2,  
reserved_tokens=['<pad>'', ''<bos>'', '<eos>'']  
len(src_vocab)
```

10012

# 9.5.4 加载数据集

回想一下，语言模型中的序列样本都有一个固定的长度，无论这个样本是一个句子的一部分还是跨越了多个句子的一个片断。这个固定长度是由 8.3 节中的 num_steps（时间步数或词元数量）参数指定的。在机器翻译中，每个样本都是由源和目标组成的文本序列对，其中的每个文本序列可能具有不同的长度。

为了提高计算效率，我们仍然可以通过截断（truncation）和填充（padding）方式实现一次只处理一个小批量的文本序列。假设同一个小批量中的每个序列都应该具有相同的长度num_steps，那么如果文本序列的词元数目少于num_steps时，我们将继续在其末尾添加特定的“<pad>”词元，直到其长度达到num_steps；反之，我们将截断文本序列时，只取其前num_steps个词元，并且丢弃剩余的词元。这样，每个文本序列将具有相同的长度，以便以相同形状的小批量进行加载。

如前所述，下面的truncate_pad函数将截断或填充文本序列。

```txt
@save def truncate_pad(line, num_steps, padding_token):
```

(continues on next page)

```python
""截断或填充文本序列''
if len(line) > num_steps:
    return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充
truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

```json
[47，4，1，1，1，1，1，1，1，1]
```

现在我们定义一个函数，可以将文本序列转换成小批量数据集用于训练。我们将特定的“<eos>”词元添加到所有序列的末尾，用于表示序列的结束。当模型通过一个词元接一个词元地生成序列进行预测时，生成的“<eos>”词元说明完成了序列输出工作。此外，我们还记录了每个文本序列的长度，统计长度时排除了填充词元，在稍后将要介绍的一些模型会需要这个长度信息。

```txt
@save   
def build_array_nmt lines,vocab,num_steps): ""将机器翻译的文本序列转换成小批量""" lines  $=$  [vocab[1]for1inlines] lines  $= [1 + [\mathrm{vocab}[' <   \mathrm{eos}]']$  for1 in lines] array  $=$  torch.tensor([truncate_pad( 1，num_steps,vocab['<pad>]）for1 in lines]) valid_len  $=$  (array！  $=$  vocab['<pad>]).type(torch.int32).sum(1) return array,valid_len
```

# 9.5.5 训练模型

最后，我们定义load_data_nmt函数来返回数据迭代器，以及源语言和目标语言的两种词表。

```txt
@save   
def load_data_nmt(batch_size，num_steps，num/examples=600): ""返回翻译数据集的迭代器和词表"" text  $=$  preprocess_nmt(read_data_nmt()) source，target  $\equiv$  tokenize_nmt(text，num/examples) src_vocab  $\equiv$  d21.Vocabsource,min_freq  $= 2$  reserved_tokens  $\coloneqq$  ['<pad>'，'<bos>'，'<eos>]) tgt_vocab  $\equiv$  d21.Vocab(target,min_freq  $= 2$  reserved_tokens  $\coloneqq$  ['<pad>'，'<bos>'，'<eos>]) src_array，src_valid_len  $\equiv$  build_array_nmt.source，src_vocab，num_steps) tgt_array，tgt_valid_len  $\equiv$  build_array_nmt(target，tgt_vocab，num_steps) data_arrays  $=$  (src_array，src_valid_len，tgt_array，tgt_valid_len)
```

```txt
(continues on next page)
```

```txt
data_iter = d2l.load_array(data_arrays, batch_size)  
return data_iter, src_vocab, tgt_vocab
```

下面我们读出“英语一法语”数据集中的第一个小批量数据。

```python
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)  
for X, X_valid_len, Y, Y_valid_len in train_iter:  
    print('X: ', X.type(torch.int32))  
    print('X的有效长度: ', X_valid_len)  
    print('Y: ', Y.type(torch.int32))  
    print('Y的有效长度: ', Y_valid_len)  
break
```

```txt
X: tensor([[7, 43, 4, 3, 1, 1, 1, 1], [44, 23, 4, 3, 1, 1, 1, 1]], dtype=torch.int32)  
X的有效长度：tensor([[4, 4])  
Y: tensor([[6, 7, 40, 4, 3, 1, 1, 1], [0, 5, 3, 1, 1, 1, 1]], dtype=torch.int32)  
Y的有效长度：tensor([[5, 3])
```

# 小结

- 机器翻译指的是将文本序列从一种语言自动翻译成另一种语言。  
- 使用单词级词元化时的词表大小，将明显大于使用字符级词元化时的词表大小。为了缓解这一问题，我们可以将低频词元视为相同的未知词元。  
- 通过截断和填充文本序列，可以保证所有的文本序列都具有相同的长度，以便以小批量的方式加载。

# 练习

1. 在load_data_nmt函数中尝试不同的numexamples参数值。这对源语言和目标语言的词表大小有何影响？  
2. 某些语言（例如中文和日语）的文本没有单词边界指示符（例如空格）。对于这种情况，单词级词元化仍然是个好主意吗？为什么？

```txt
Discussions114
```

```txt
114 https://discuss.d2l.ai/t/2776
```

# 9.6 编码器-解码器架构

正如我们在9.5节中所讨论的，机器翻译是序列转换模型的一个核心问题，其输入和输出都是长度可变的序列。为了处理这种类型的输入和输出，我们可以设计一个包含两个主要组件的架构：第一个组件是一个编码器 encoder)：它接受一个长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。第二个组件是解码器 (decoder)：它将固定形状的编码状态映射到长度可变的序列。这被称为编码器-解码器 (encoder-decoder) 架构，如图9.6.1所示。

![](images/9ba13e38b70d7ae06ada8cccc766fcf5301623eddda5f186f7cf2a7bc158b06b.jpg)  
图9.6.1: 编码器-解码器架构

我们以英语到法语的机器翻译为例：给定一个英文的输入序列：“They”“are”“watching”“.”。首先，这种“编码器一解码器”架构将长度可变的输入序列编码成一个“状态”，然后对该状态进行解码，一个词元接着一个词元地生成翻译后的序列作为输出：“Ils”“regordent”“.”。由于“编码器一解码器”架构是形成后续章节中不同序列转换模型的基础，因此本节将把这个架构转换为接口方便后面的代码实现。

# 9.6.1 编码器

在编码器接口中，我们只指定长度可变的序列作为编码器的输入X。任何继承这个Encoder基类的模型将完成代码实现。

```python
from torch import nn   
@save   
class Encoder(nnModule): ""编码器-解码器架构的基本编码器接口""" def__init__(self，\*\*kwargs): super(Encoder，self).__init__(\*\*kwargs) def forward(self，X，\*args): raise NotImplementedError
```

# 9.6.2 解码器

在下面的解码器接口中，我们新增一个init_state函数，用于将编码器的输出（enc_outputs）转换为编码后的状态。注意，此步骤可能需要额外的输入，例如：输入序列的有效长度，这在9.5.4节中进行了解释。为了逐个地生成长度可变的词元序列，解码器在每个时间步都会将输入（例如：在前一时间步生成的词元）和编码后的状态映射成当前时间步的输出词元。

```python
#include <class>   
class Decoder(nnModule):   
    ""编码器-解码器架构的基本解码器接口""   
def __init__(self, **kwargs):   
    super(Decoder, self).__init__(**kwargs)   
def init_state(self, enc_outputs, *args):   
    raise NotImplementedError   
def forward(self, X, state):   
    raise NotImplementedError
```

# 9.6.3 合并编码器和解码器

总而言之，“编码器-解码器”架构包含了一个编码器和一个解码器，并且还拥有可选的额外的参数。在前向传播中，编码器的输出用于生成编码状态，这个状态又被解码器作为其输入的一部分。

```python
#include <class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self encoder = encoder
        selfdecoder = decoder
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self encoder(enc_X, *args)
        dec_state = self decoder.init_state(enc_outputs, *args)
        return self decoder(dec_X, dec_state)
```

“编码器一解码器”体系架构中的术语状态会启发人们使用具有状态的神经网络来实现该架构。在下一节中，我们将学习如何应用循环神经网络，来设计基于“编码器一解码器”架构的序列转换模型。

# 9.6. 编码器-解码器架构

# 小结

- “编码器一解码器”架构可以将长度可变的序列作为输入和输出，因此适用于机器翻译等序列转换问题。  
- 编码器将长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。  
- 解码器将具有固定形状的编码状态映射为长度可变的序列。

# 练习

1. 假设我们使用神经网络来实现“编码器—解码器”架构，那么编码器和解码器必须是同一类型的神经网络吗？  
2. 除了机器翻译，还有其它可以适用于”编码器一解码器“架构的应用吗？

Discussions<sup>115</sup>

# 9.7 序列到序列学习（seq2seq）

正如我们在9.5节中看到的，机器翻译中的输入序列和输出序列都是长度可变的。为了解决这类问题，我们在9.6节中设计了一个通用的”编码器一解码器“架构。本节，我们将使用两个循环神经网络的编码器和解码器，并将其应用于序列到序列（sequence to sequence，seq2seq）类的学习任务(Cho et al., 2014, Sutskever et al., 2014)。

遵循编码器一解码器架构的设计原则，循环神经网络编码器使用长度可变的序列作为输入，将其转换为固定形状的隐状态。换言之，输入序列的信息被编码到循环神经网络编码器的隐状态中。为了连续生成输出序列的词元，独立的循环神经网络解码器是基于输入序列的编码信息和输出序列已经看见的或者生成的词元来预测下一个词元。图9.7.1演示了如何在机器翻译中使用两个循环神经网络进行序列到序列学习。

![](images/709a40732675c60ab76588ac7110333a5a0523a9b49ce05ccf8ccfc2283a9520.jpg)  
图9.7.1: 使用循环神经网络编码器和循环神经网络解码器的序列到序列学习

在图9.7.1中，特定的“<eos>”表示序列结束词元。一旦输出序列生成此词元，模型就会停止预测。在循环神经网络解码器的初始化时间步，有两个特定的设计决定：首先，特定的“<bos>”表示序列开始词元，它是解码器的输入序列的第一个词元。其次，使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态。例如，在(Sutskever et al., 2014)的设计中，正是基于这种设计将输入序列的编码信息送入到解码器中来生成输

出序列的。在其他一些设计中 (Cho et al., 2014), 如图9.7.1所示, 编码器最终的隐状态在每一个时间步都作为解码器的输入序列的一部分。类似于8.3节中语言模型的训练, 可以允许标签成为原始的输出序列, 从源序列词元 “<bos>” “Ils” “regardent” “.” 到新序列词元 “Ils” “regardent” “.” “<eos>” 来移动预测的位置。

下面，我们动手构建图9.7.1的设计，并将基于9.5节中介绍的“英一法”数据集来训练这个机器翻译模型。

```python
import collections  
import math  
import torch  
from torch import nn  
from d2l import torch as d2l
```

# 9.7.1 编码器

从技术上讲，编码器将长度可变的输入序列转换成形状固定的上下文变量c，并且将输入序列的信息在该上下文变量中进行编码。如图9.7.1所示，可以使用循环神经网络来设计编码器。

考虑由一个序列组成的样本（批量大小是1）。假设输入序列是  $x_{1},\ldots ,x_{T}$  ，其中  $x_{t}$  是输入文本序列中的第  $t$  个词元。在时间步  $t$  ，循环神经网络将词元  $x_{t}$  的输入特征向量  $\mathbf{x}_t$  和  $\mathbf{h}_{t - 1}$  （即上一时间步的隐状态）转换为  $\mathbf{h}_t$  （即当前步的隐状态)。使用一个函数  $f$  来描述循环神经网络的循环层所做的变换：

$$
\mathbf {h} _ {t} = f \left(\mathbf {x} _ {t}, \mathbf {h} _ {t - 1}\right). \tag {9.7.1}
$$

总之，编码器通过选定的函数  $q$ ，将所有时间步的隐状态转换为上下文变量：

$$
\mathbf {c} = q \left(\mathbf {h} _ {1}, \dots , \mathbf {h} _ {T}\right). \tag {9.7.2}
$$

比如，当选择  $q(\mathbf{h}_1,\dots ,\mathbf{h}_T) = \mathbf{h}_T$  时（就像图9.7.1中一样)，上下文变量仅仅是输入序列在最后时间步的隐状态  $\mathbf{h}_T$  。

到目前为止，我们使用的是一个单向循环神经网络来设计编码器，其中隐状态只依赖于输入子序列，这个子序列是由输入序列的开始位置到隐状态所在的时间步的位置（包括隐状态所在的时间步）组成。我们也可以使用双向循环神经网络构造编码器，其中隐状态依赖于两个输入子序列，两个子序列是由隐状态所在的时间步的位置之前的序列和之后的序列（包括隐状态所在的时间步），因此隐状态对整个序列的信息都进行了编码。

现在，让我们实现循环神经网络编码器。注意，我们使用了嵌入层（embedding layer）来获得输入序列中每个词元的特征向量。嵌入层的权重是一个矩阵，其行数等于输入词表的大小（vocab_size），其列数等于特征向量的维度（embed_size）。对于任意输入词元的索引  $i$ ，嵌入层获取权重矩阵的第  $i$  行（从0开始）以返回其特征向量。另外，本文选择了一个多层门控循环单元来实现编码器。

```txt
@save   
classSeq2SeqEncoder(d21.Encoder): ""用于序列到序列学习的循环神经网络编码器""
```

(continues on next page)

```python
def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs): super(Seq2SeqEncoder, self).__init__(**kwargs) # 嵌入层 self_embedding = nn.Embedding(vocab_size, embed_size) self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout) def forward(self, X, *args): # 输出‘X'的形状：(batch_size,num_steps,embed_size) X = self.Embedding(X) # 在循环神经网络模型中，第一个轴对应于时间步 X = X.permute(1, 0, 2) # 如果未提及状态，则默认为0 output, state = self.rnn(X) # output的形状：(num_steps,batch_size,num_hiddens) # state的形状：(num_layers,batch_size,num_hiddens) return output, state
```

循环层返回变量的说明可以参考8.6节。

下面，我们实例化上述编码器的实现：我们使用一个两层门控循环单元编码器，其隐藏单元数为16。给定一小批量的输入序列x（批量大小为4，时间步为7)。在完成所有时间步后，最后一层的隐状态的输出是一个张量（output由编码器的循环层返回)，其形状为（时间步数，批量大小，隐藏单元数)。

```txt
encoder  $=$  Seq2SeqEncoder(vocab_size  $\coloneqq 10$  ,embed_size  $= 8$  ,num_hiddens  $= 16$  num_layers  $= 2$    
encoder.eval()   
X  $=$  torch.zeros((4,7),dtype  $\equiv$  torch.long)   
output, state  $=$  encoder(X)   
output.shape
```

```txt
torch.Size([7, 4, 16])
```

由于这里使用的是门控循环单元，所以在最后一个时间步的多层隐状态的形状是（隐藏层的数量，批量大小，隐藏单元的数量)。如果使用长短期记忆网络，state中还将包含记忆单元信息。

```txt
state.shape
```

```txt
torch.Size([2, 4, 16])
```

# 9.7.2 解码器

正如上文提到的，编码器输出的上下文变量c对整个输入序列  $x_{1},\ldots ,x_{T}$  进行编码。来自训练数据集的输出序列  $y_{1},y_{2},\ldots ,y_{T^{\prime}}$  ，对于每个时间步  $t^\prime$  （与输入序列或编码器的时间步  $t$  不同)，解码器输出  $y_{t'}$  的概率取决于先前的输出子序列  $y_{1},\ldots ,y_{t^{\prime} - 1}$  和上下文变量c，即  $P(y_{t^{\prime}}\mid y_1,\dots ,y_{t^{\prime} - 1},\mathbf{c})$  。

为了在序列上模型化这种条件概率，我们可以使用另一个循环神经网络作为解码器。在输出序列上的任意时间步  $t'$ ，循环神经网络将来自上一时间步的输出  $y_{t' - 1}$  和上下文变量  $\mathbf{c}$  作为其输入，然后在当前时间步将它们和上一隐状态  $\mathbf{s}_{t' - 1}$  转换为隐状态  $\mathbf{s}_{t'}$ 。因此，可以使用函数  $g$  来表示解码器的隐藏层的变换：

$$
\mathbf {s} _ {t ^ {\prime}} = g \left(y _ {t ^ {\prime} - 1}, \mathbf {c}, \mathbf {s} _ {t ^ {\prime} - 1}\right). \tag {9.7.3}
$$

在获得解码器的隐状态之后，我们可以使用输出层和softmax操作来计算在时间步  $t^{\prime}$  时输出  $y_{t^{\prime}}$  的条件概率分布  $P(y_{t^{\prime}}\mid y_1,\dots ,y_{t^{\prime} - 1},\mathbf{c})$  。

根据图9.7.1，当实现解码器时，我们直接使用编码器最后一个时间步的隐状态来初始化解码器的隐状态。这就要求使用循环神经网络实现的编码器和解码器具有相同数量的层和隐藏单元。为了进一步包含经过编码的输入序列的信息，上下文变量在所有的时间步与解码器的输入进行拼接（concatenate）。为了预测输出词元的概率分布，在循环神经网络解码器的最后一层使用全连接层来变换隐状态。

```python
class Seq2SeqDecoder(d2l.Decoder):
    ""用于序列到序列学习的循环神经网络解码器''
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self_embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc Outputs, *args):
        return enc_outputs[1]
    def forward(self, X, state):
        # 输出'X'的形状: (batch_size, num_steps, embed_size)
        X = self_embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状: (batch_size, num_steps, vocab_size)
        # state的形状: (num_layers, batch_size, num_hiddens)
        return output, state
```

下面，我们用与前面提到的编码器中相同的超参数来实例化解码器。如我们所见，解码器的输出形状变为（批

量大小，时间步数，词表大小)，其中张量的最后一个维度存储预测的词元分布。

```prolog
decoder  $=$  Seq2SeqDecoder(vocab_size  $\coloneqq 10$  ,embed_size  $= 8$  ,num_hiddens  $= 16$  num_layers  $= 2$    
decoder.eval()   
state  $=$  decoder.init_state encoder(X))   
output, state  $=$  decoder(X, state)   
output.shape, state.shape
```

```txt
torch.Size([4,7，10])，torch.Size([2，4，16]))
```

总之，上述循环神经网络“编码器一解码器”模型中的各层如图9.7.2所示。

![](images/fed177daf9f9121c1535f9d0624a9b2544075fd46f9c94cd70e28f15c78bc83e.jpg)  
图9.7.2: 循环神经网络编码器-解码器模型中的层

# 9.7.3 损失函数

在每个时间步，解码器预测了输出词元的概率分布。类似于语言模型，可以使用softmax来获得分布，并通过计算交叉熵损失函数来进行优化。回想一下9.5节中，特定的填充词元被添加到序列的末尾，因此不同长度的序列可以以相同形状的小批量加载。但是，我们应该将填充词元的预测排除在损失函数的计算之外。

为此，我们可以使用下面的sequence_mask函数通过零值化屏蔽不相关的项，以便后面任何不相关预测的计算都是与零的乘积，结果都等于零。例如，如果两个序列的有效长度（不包括填充词元）分别为1和2，则第一个序列的第一项和第二个序列的前两项之后的剩余项将被清除为零。

```txt
@save   
def sequence_mask(X，valid_len，value  $= 0$  ：   
"""在序列中屏蔽不相关的项""""   
maxlen  $\equiv$  X.size(1)   
mask  $\equiv$  torch.arange((maxlen)，dtype  $\equiv$  torch.float32, device  $\equiv$  X_device)[None，：]  $<$  valid_len[：，None]   
X[~mask]  $\equiv$  value   
return X
```

```txt
(continues on next page)
```

```python
X = torch.tensor([[1, 2, 3], [4, 5, 6]]) sequence_mask(X, torch.tensor([1, 2]))
```

```txt
tensor([1, 0, 0], [4, 5, 0])
```

我们还可以使用此函数屏蔽最后几个轴上的所有项。如果愿意，也可以使用指定的非零值来替换这些项。

```txt
X = torch.ones(2, 3, 4)  
sequence_mask(X, torch.tensord([1, 2]), value=-1)
```

```txt
tensor([[1.，1.，1.，1.],[-1.，-1.，-1.，-1.],[-1.，-1.，-1.，-1.]],[[1.，1.，1.，1.],[1.，1.，1.，1.],[-1.，-1.，-1.，-1.]]])
```

现在，我们可以通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。最初，所有预测词元的掩码都设置为1。一旦给定了有效长度，与填充词元对应的掩码将被设置为0。最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。

```python
@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    ""带遮蔽的softmax交叉熵损失函数''
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like标签)
        weights = sequence_mask(weights, valid_len)
        self.reduce='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0,2,1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

我们可以创建三个相同的序列来进行代码健全性检查，然后分别指定这些序列的有效长度为4、2和0。结果就是，第一个序列的损失应为第二个序列的两倍，而第三个序列的损失应为零。

```javascript
loss  $=$  MaskedSoftmaxCELoss()   
loss(torch.ones(3,4，10)，torch.ones((3，4)，dtype  $\equiv$  torch.long), torch.tensor([4，2，0]))
```

```txt
tensor([2.3026, 1.1513, 0.0000])
```

# 9.7.4 训练

在下面的循环训练过程中，如图9.7.1所示，特定的序列开始词元（“<bos>”）和原始的输出序列（不包括序列结束词元“<eos>”）拼接在一起作为解码器的输入。这被称为强制教学（teacher forcing），因为原始的输出序列（词元的标签）被送入解码器。或者，将来自上一个时间步的预测得到的词元作为解码器的当前输入。

```python
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    '''训练序列到序列模型'''''`
    def xavier_initweights(m):
        if type(m) == nn.Linear:
            nn.init.xavier.uniform(m.weight)
        if type(m) == nn.GRU:
            for param in m._flatweights_names:
                if "weight" in param:
                    nn.init.xavier.uniform(m._parameters[param])
            net.apply(xavier_initweights)
        net.to(device)
        optimizer = torch.optim.Adam(net.params(), lr=lr)
        loss = MaskedSoftmaxCELoss()
        net.train()
        optimizer = d2lancesnet.Label('epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']) * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :, -1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
```

(continues on next page)

```python
l-sum().backward() # 损失函数的标量进行“反向传播”  
d21.grad_clipping(net, 1)  
num_tokens = Y_valid_len.sum()  
optimizer step()  
with torch.no_grad():  
    metric.add(l,sum(), num_tokens)  
if (epoch + 1) % 10 == 0:  
    optimizer.add(epoch + 1, (metric[0] / metric[1],))  
print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '  
f'tokens/sec on {str(device)}')
```

现在，在机器翻译数据集上，我们可以创建和训练一个循环神经网络“编码器—解码器”模型用于序列到序列的学习。

```txt
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1  
batch_size, num_steps = 64, 10  
lr, num_epochs, device = 0.005, 300, d2l.trygpu()  
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)  
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)  
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)  
net = d2l EncoderDecoder encoder, decoder)  
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

loss 0.019, 12745.1 tokens/sec onuda:0

![](images/b4ad1244ade195dc81d4624d708208c240567e1ca7eab7f941a82cc8a3e8fd83.jpg)

# 9.7.5 预测

为了采用一个接着一个词元的方式预测输出序列，每个解码器当前时间步的输入都将来自于前一时间步的预测词元。与训练类似，序列开始词元（“<bos>”）在初始时间步被输入到解码器中。该预测过程如图9.7.3所示，当输出序列的预测遇到序列结束词元（“<eos>”）时，预测就结束了。

![](images/d7137127c1af45737f6ea0515d076dbbb3174742030b418e1669ee590512eabb.jpg)  
图9.7.3: 使用循环神经网络编码器-解码器逐词元地预测输出序列。

我们将在9.8节中介绍不同的序列生成策略。

```python
def predict_seq2seq(net, srcsentence, src_vocab, tgt_vocab, num_steps, device, saveattentionweights=False):
    '''序列到序列模型的预测''' # 在预测时将net设置为评估模式 net.eval()
    src_tokens = src_vocab[srcsentence.lower().split('')]+
        src_vocab['<eos>']
    enc_valid_len = torch.tensor(len(src_tokens)], device=device)
    src_tokens = d21.truncate_pad(src_tokens, num_steps, src_vocab['<pad>])
# 添加批量轴 enc_X = torch unsqueeze(
    torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
enc_outputs = net encoder(enc_X, enc_valid_len)
dec_state = net decoder.init_state(ENC Outputs, enc_valid_len)
# 添加批量轴 dec_X = torch unsqueeze(torch.tensor(
    [tgt_vocab['<bos>'], dtype=torch.long, device=device), dim=0)
output_seq, attention_weight_seq = [], []
for _ in range(num_steps):
    Y, dec_state = netdecoder(dec_X, dec_state)
    # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
    dec_X = Y.argmax(dim=2)
    pred = dec_X.squeeze(dim=0).type(torch.int32).item()
    # 保存注意力权重（稍后讨论）
    if saveattentionweights:
        attention_weight_seq.append(netDecoder attendsitionweights)
```

(continues on next page)

```txt
一旦序列结束词元被预测，输出序列的生成就完成了  
if pred == tgt_vocab['<eos>]:  
    break  
output_seq.append(pred)  
return '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

# 9.7.6 预测序列的评估

我们可以通过与真实的标签序列进行比较来评估预测序列。虽然(Papineni et al., 2002)提出的BLEU(bilingual evaluation understudy）最先是用于评估机器翻译的结果，但现在它已经被广泛用于测量许多应用的输出序列的质量。原则上说，对于预测序列中的任意  $n$  元语法（n-grams），BLEU的评估都是这个  $n$  元语法是否出现在标签序列中。

我们将BLEU定义为：

$$
\exp \left(\min  \left(0, 1 - \frac {\operatorname {l e n} _ {\text {l a b e l}}}{\operatorname {l e n} _ {\text {p r e d}}}\right)\right) \prod_ {n = 1} ^ {k} p _ {n} ^ {1 / 2 ^ {n}}, \tag {9.7.4}
$$

其中len_label表示标签序列中的词元数和len_pred表示预测序列中的词元数， $k$ 是用于匹配的最长的n元语法。另外，用 $p_n$ 表示n元语法的精确度，它是两个数量的比值：第一个是预测序列与标签序列中匹配的n元语法的数量，第二个是预测序列中n元语法的数量的比率。具体地说，给定标签序列A、B、C、D、E、F和预测序列A、B、B、C、D，我们有 $p_1 = 4 / 5$ 、 $p_2 = 3 / 4$ 、 $p_3 = 1 / 3$ 和 $p_4 = 0$ 。

根据(9.7.4)中BLEU的定义，当预测序列与标签序列完全相同时，BLEU为1。此外，由于  $n$  元语法越长则匹配难度越大，所以BLEU为更长的  $n$  元语法的精确度分配更大的权重。具体来说，当  $p_n$  固定时， $p_n^{1/2^n}$  会随着  $n$  的增长而增加（原始论文使用  $p_n^{1/n}$ ）。而且，由于预测的序列越短获得的  $p_n$  值越高，所以(9.7.4)中乘法项之前的系数用于惩罚较短的预测序列。例如，当  $k = 2$  时，给定标签序列  $A$ 、 $B$ 、 $C$ 、 $D$ 、 $E$ 、 $F$  和预测序列  $A$ 、 $B$ ，尽管  $p_1 = p_2 = 1$ ，惩罚因子  $\exp(1 - 6/2) \approx 0.14$  会降低BLEU。

BLEU的代码实现如下。

```python
defbleu(pred_seq，label_seq，k):@save   
"""计算BLEU"""   
pred_tokens，label_tokens  $\equiv$  pred_seq.split('')，label_seq.split('')   
len_pred，len_label  $=$  len(pred_tokens)，len.label_tokens)   
score  $\equiv$  math.exp(min(0,1-len_label/len_pred))   
forn in range(1,k+1): nummatches，label_sub  $= 0$  ，collections.defaultdict(int) fori in range(len_label-n+1): label_subse['.join.label_tokens[i:i+n]）  $+ = 1$  fori in range(len_pred-n+1): iflabel_subse['.join(pred_tokens[i:i+n])  $] > 0$  ： nummatches  $+ = 1$
```

(continues on next page)

```python
label_subs[''.join(pred_tokens[i: i + n]]) -= 1  
score *= math.pow(num_MATCHes / (len_pred - n + 1), math.pow(0.5, n))  
return score
```

最后，利用训练好的循环神经网络“编码器—解码器”模型，将几个英语句子翻译成法语，并计算BLEU的最终结果。

```python
engs = ['go ', 'i lost ', 'he\s calm ', 'i'm home.']  
fras = ['va!', 'j\ai perdu ', 'il est calme ', 'je suis chez moi.']  
for eng, fra in zip(engs, fras):  
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)  
print(f'\{eng\} => {translation}, bleu{bleu(translation, fra, k=2):.3f}')
```

```txt
go.  $\Rightarrow$  va!,bleu 1.000 i lost.  $\Rightarrow$  j'ai perdu.,bleu 1.000 he's calm.  $\Rightarrow$  il est riche.,bleu 0.658 i'm home.  $\Rightarrow$  je suis en retard?,bleu 0.447
```

# 小结

- 根据“编码器-解码器”架构的设计，我们可以使用两个循环神经网络来设计一个序列到序列学习的模型。  
- 在实现编码器和解码器时，我们可以使用多层循环神经网络。  
- 我们可以使用遮蔽来过滤不相关的计算，例如在计算损失时。  
- 在“编码器一解码器”训练中，强制教学方法将原始输出序列（而非预测结果）输入解码器。  
- BLEU是一种常用的评估方法，它通过测量预测序列和标签序列之间的  $n$  元语法的匹配度来评估预测。

# 练习

1. 试着通过调整超参数来改善翻译效果。  
2. 重新运行实验并在计算损失时不使用遮蔽，可以观察到什么结果？为什么会有这个结果？  
3. 如果编码器和解码器的层数或者隐藏单元数不同，那么如何初始化解码器的隐状态？  
4. 在训练中，如果用前一时间步的预测输入到解码器来代替强制教学，对性能有何影响？  
5. 用长短期记忆网络替换门控循环单元重新运行实验  
6. 有没有其他方法来设计解码器的输出层？

# 9.8 束搜索

在9.7节中，我们逐个预测输出序列，直到预测序列中出现特定的序列结束词元“<eos>”。本节将首先介绍贪心搜索（greedy search）策略，并探讨其存在的问题，然后对比其他替代策略：穷举搜索（exhaustive search）和束搜索（beam search）。

在正式介绍贪心搜索之前，我们使用与9.7节中相同的数学符号定义搜索问题。在任意时间步  $t'$ ，解码器输出  $y_{t'}$  的概率取决于时间步  $t'$  之前的输出子序列  $y_1, \ldots, y_{t' - 1}$  和对输入序列的信息进行编码得到的上下文变量  $\mathbf{c}$  。为了量化计算代价，用  $\mathcal{V}$  表示输出词表，其中包含“ $\langle \mathrm{eos} \rangle$ ”，所以这个词汇集合的基数  $|\mathcal{V}|$  就是词表的大小。我们还将输出序列的最大词元数指定为  $T'$  。因此，我们的目标是从所有  $\mathcal{O}(|\mathcal{V}|^{T'})$  个可能的输出序列中寻找理想的输出。当然，对于所有输出序列，在“ $\langle \mathrm{eos} \rangle$ ”之后的部分（非本句）将在实际输出中丢弃。

# 9.8.1 贪心搜索

首先，让我们看看一个简单的策略：贪心搜索，该策略已用于9.7节的序列预测。对于输出序列的每一时间步  $t'$ ，我们都将基于贪心搜索从  $\mathcal{Y}$  中找到具有最高条件概率的词元，即：

$$
y _ {t ^ {\prime}} = \underset {y \in \mathcal {Y}} {\operatorname {a r g m a x}} P (y \mid y _ {1}, \dots , y _ {t ^ {\prime} - 1}, \mathbf {c}) \tag {9.8.1}
$$

一旦输出序列包含了“<eos>”或者达到其最大长度  $T'$ ，则输出完成。

![](images/9b439a352fb495246a6c51c21af4526922e6738fb2a915434f9bb1d49fcb0f6c.jpg)  
图9.8.1: 在每个时间步, 贪心搜索选择具有最高条件概率的词元

如图9.8.1中，假设输出中有四个词元“A”“B”“C”和“<eos>”。每个时间步下的四个数字分别表示在该时间步生成“A”“B”“C”和“<eos>”的条件概率。在每个时间步，贪心搜索选择具有最高条件概率的词元。因此，将在图9.8.1中预测输出序列“A”“B”“C”和“<eos>”。这个输出序列的条件概率是  $0.5 \times 0.4 \times 0.4 \times 0.6 = 0.048$ 。那么贪心搜索存在的问题是什么呢？现实中，最优序列（optimal sequence）应该是最大化  $\prod_{t'=1}^{T'} P(y_{t'} | y_1, \ldots, y_{t'-1}, \mathbf{c})$  值的输出序列，这是基于输入序列生成输出序列的条件概率。然而，贪心搜索无法保证得到最优序列。

![](images/e57a5743ff256ab429ec62027fa6a602c6af10b077ff3800eb2bdf3643f70288.jpg)  
图9.8.2: 在时间步2, 选择具有第二高条件概率的词元 “C” (而非最高条件概率的词元)

图9.8.2中的另一个例子阐述了这个问题。与图9.8.1不同，在时间步2中，我们选择图9.8.2中的词元“C”，它具有第二高的条件概率。由于时间步3所基于的时间步1和2处的输出子序列已从图9.8.1中的“A”和“B”改变为图9.8.2中的“A”和“C”，因此时间步3处的每个词元的条件概率也在图9.8.2中改变。假设我们在时间步3选择词元“B”，于是当前的时间步4基于前三个时间步的输出子序列“A”“C”和“B”为条件，这与图9.8.1中的“A”“B”和“C”不同。因此，在图9.8.2中的时间步4生成每个词元的条件概率也不同于图9.8.1中的条件概率。结果，图9.8.2中的输出序列“A”“C”“B”和“<eos>”的条件概率为  $0.5 \times 0.3 \times 0.6 \times 0.6 = 0.054$  ，这大于图9.8.1中的贪心搜索的条件概率。这个例子说明：贪心搜索获得的输出序列“A”“B”“C”和“<eos>”不一定是最佳序列。

# 9.8.2 穷举搜索

如果目标是获得最优序列，我们可以考虑使用穷举搜索（exhaustive search）：穷举地列举所有可能的输出序列及其条件概率，然后计算输出条件概率最高的一个。

虽然我们可以使用穷举搜索来获得最优序列, 但其计算量  $\mathcal{O}(|\mathcal{Y}|^{T^{\prime}})$  可能高的惊人。例如, 当  $|\mathcal{Y}| = 10000$  和  $T^{\prime} = 10$  时, 我们需要评估  $10000^{10} = 10^{40}$  序列, 这是一个极大的数, 现有的计算机几乎不可能计算它。然而, 贪心搜索的计算量  $\mathcal{O}(|\mathcal{Y}|T^{\prime})$  通它要显著地小于穷举搜索。例如, 当  $|\mathcal{Y}| = 10000$  和  $T^{\prime} = 10$  时, 我们只需要评估  $10000 \times 10 = 10^{5}$  个序列。

# 9.8.3 束搜索

那么该选取哪种序列搜索策略呢？如果精度最重要，则显然是穷举搜索。如果计算成本最重要，则显然是贪心搜索。而束搜索的实际应用则介于这两个极端之间。

束搜索（beam search）是贪心搜索的一个改进版本。它有一个超参数，名为束宽（beam size） $k$ 。在时间步1，我们选择具有最高条件概率的 $k$ 个词元。这 $k$ 个词元将分别是 $k$ 个候选输出序列的第一个词元。在随后的每个时间步，基于上一时间步的 $k$ 个候选输出序列，我们将继续从 $k|\mathcal{Y}|$ 个可能的选择中挑出具有最高条件概率的 $k$ 个候选输出序列。

![](images/d9cac14d1a9bf9fa3e6c12d7f88163455da22a9990c7a81306593c4389c76896.jpg)  
图9.8.3: 束搜索过程 (束宽: 2, 输出序列的最大长度: 3)。候选输出序列是  $A 、 C 、 A B 、 C E 、 A B D$  和  $C E D$

图9.8.3演示了束搜索的过程。假设输出的词表只包含五个元素:  $\mathcal{Y} = \{A, B, C, D, E\}$ , 其中有一个是 “<eos>”。设置束宽为 2 , 输出序列的最大长度为 3 。在时间步 1 , 假设具有最高条件概率  $P(y_{1} \mid \mathbf{c})$  的词元是 A 和 C 。在时间步 2 , 我们计算所有  $y_{2} \in \mathcal{Y}$  为:

$$
P \left(A, y _ {2} \mid \mathbf {c}\right) = P (A \mid \mathbf {c}) P \left(y _ {2} \mid A, \mathbf {c}\right), \tag {9.8.2}
$$

$$
P (C, y _ {2} \mid \mathbf {c}) = P (C \mid \mathbf {c}) P (y _ {2} \mid C, \mathbf {c}),
$$

从这十个值中选择最大的两个，比如  $P(A, B \mid \mathbf{c})$  和  $P(C, E \mid \mathbf{c})$  。然后在时间步3，我们计算所有  $y_{3} \in \mathcal{V}$  为：

$$
P (A, B, y _ {3} \mid \mathbf {c}) = P (A, B \mid \mathbf {c}) P (y _ {3} \mid A, B, \mathbf {c}), \tag {9.8.3}
$$

$$
P (C, E, y _ {3} \mid \mathbf {c}) = P (C, E \mid \mathbf {c}) P (y _ {3} \mid C, E, \mathbf {c}),
$$

从这十个值中选择最大的两个，即  $P(A,B,D\mid \mathbf{c})$  和  $P(C,E,D\mid \mathbf{c})$  ，我们会得到六个候选输出序列：（1）A；(2)C；（3）A,B；（4）C,E；（5）A,B,D；（6）C,E,D。

最后，基于这六个序列（例如，丢弃包括“<eos>”和之后的部分），我们获得最终候选输出序列集合。然后我们选择其中条件概率乘积最高的序列作为输出序列：

$$
\frac {1}{L ^ {\alpha}} \log P \left(y _ {1}, \dots , y _ {L} \mid \mathbf {c}\right) = \frac {1}{L ^ {\alpha}} \sum_ {t ^ {\prime} = 1} ^ {L} \log P \left(y _ {t ^ {\prime}} \mid y _ {1}, \dots , y _ {t ^ {\prime} - 1}, \mathbf {c}\right), \tag {9.8.4}
$$

其中  $L$  是最终候选序列的长度，  $\alpha$  通常设置为0.75。因为一个较长的序列在(9.8.4)的求和中会有更多的对数项，因此分母中的  $L^{\alpha}$  用于惩罚长序列。

束搜索的计算量为  $\mathcal{O}(k|\mathcal{Y}|T^{\prime})$  ，这个结果介于贪心搜索和穷举搜索之间。实际上，贪心搜索可以看作一种束宽为1的特殊类型的束搜索。通过灵活地选择束宽，束搜索可以在正确率和计算代价之间进行权衡。

# 小结

- 序列搜索策略包括贪心搜索、穷举搜索和束搜索。  
- 贪心搜索所选取序列的计算量最小，但精度相对较低。  
- 穷举搜索所选取序列的精度最高，但计算量最大。  
- 束搜索通过灵活选择束宽，在正确率和计算代价之间进行权衡。

# 练习

1. 我们可以把穷举搜索看作一种特殊的束搜索吗？为什么？  
2. 在9.7节的机器翻译问题中应用束搜索。束宽是如何影响预测的速度和结果的？  
3. 在 8.5节中，我们基于用户提供的前缀，通过使用语言模型来生成文本。这个例子中使用了哪种搜索策略？可以改进吗？

Discussions<sup>117</sup>

# 10

# 注意力机制

灵长类动物的视觉系统接受了大量的感官输入，这些感官输入远远超过了大脑能够完全处理的程度。然而，并非所有刺激的影响都是相等的。意识的聚集和专注使灵长类动物能够在复杂的视觉环境中将注意力引向感兴趣的物体，例如猎物和天敌。只关注一小部分信息的能力对进化更加有意义，使人类得以生存和成功。

自19世纪以来，科学家们一直致力于研究认知神经科学领域的注意力。本章的很多章节将涉及到一些研究。

首先回顾一个经典注意力框架,解释如何在视觉场景中展开注意力。受此框架中的注意力提示(attention cues)的启发，我们将设计能够利用这些注意力提示的模型。1964年的Nadaraya-Waston核回归（kernel regression）正是具有注意力机制（attention mechanism）的机器学习的简单演示。

然后继续介绍的是注意力函数，它们在深度学习的注意力模型设计中被广泛使用。具体来说，我们将展示如何使用这些函数来设计Bahdanau注意力。Bahdanau注意力是深度学习中的具有突破性价值的注意力模型，它双向对齐并且可以微分。

最后将描述仅仅基于注意力机制的Transformer架构，该架构中使用了多头注意力（multi-head attention）和自注意力（self-attention）。自2017年横空出世，Transformer一直都普遍存在于现代的深度学习应用中，例如语言、视觉、语音和强化学习领域。