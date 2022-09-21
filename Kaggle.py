import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import matplotlib.pyplot as plt

# 从数据集中取出数据
train_data = pd.read_csv('california-house-prices/train.csv')  # 训练的数据集,包含了所有的特征和售价
test_x = pd.read_csv('california-house-prices/test.csv')  # 测试的数据集,只有特征,没有售价

# 取出训练特征以及售价,将其存储至两个变量中
train_x = train_data.iloc[:, 3:40]  # 取出特征(该数据集有41列,前面都是一些文本的数据还有售价在第三列,所以从下标3,也就是第四行开始取后面的特征)
train_x_numeric_features = train_x.dtypes[
    train_x.dtypes != 'object'].index  # 将所有数字类型的特征的(不是object那就是数字)index取出来,方便后面取出数字类型的数据
train_x = train_x[train_x_numeric_features]  # 取出数字类型的数据
# 数据预处理,将得到的数字数据进行标准化处理,将其变成一个均值为0,方差为1的标准正态分布
train_x = train_x.apply(
    lambda x: (x - x.mean()) / x.std()  # 将所有的数据减去数据集的均值,那么这个数据集均值就为0了,再除以原本的方差,新的数据集方差就为1
)
train_x = train_x.fillna(0)  # 将取出的数据集中没有数字的地方填满0(0不会影响均值和方差,不改变原来的数据分布)
train_y = train_data.iloc[:, 2]  # 将售价取出来

# 对测试的数据进行同样的标准化处理
test_x = test_x.iloc[:, 1:39]
test_x_numeric_features = test_x.dtypes[test_x.dtypes != 'object'].index
test_x = test_x[test_x_numeric_features]
test_x = test_x.apply(
    lambda x: (x - x.mean()) / x.std()
)
test_x = train_x.fillna(0)

# 将刚刚预处理好的数据转变为tensor格式,方便后面计算
train_x = torch.tensor(train_x.values, dtype=torch.float32)
train_y = torch.tensor(train_y.values, dtype=torch.float32)
test_x = torch.tensor(test_x.values, dtype=torch.float32)

# 输入的维度和输出的维度和loss
num_inputs, num_outputs = train_x.shape[1], 1  # 输入维度就是数据集一条数据的维度,输出维度因为是回归问题所以是1

loss = nn.MSELoss()


# 定义损失函数
def log_rmse(y_hat, y):
    clipped_preds = torch.clamp(y_hat, 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(y)))
    return rmse


# 定义一个函数方便我们重新生成一个神经网络
def get_net():
    net = nn.Sequential(nn.Linear(num_inputs, num_outputs))  # 网络的定义
    net.apply(init_weights)  # 初始化权重
    return net


# 权重初始化的函数
def init_weights(m):
    if type(m) == nn.Module:
        nn.init.normal_(m.weight, mean=0, std=1)  # 将weight初始化为高斯分布
        nn.init.zeros_(m.bias)  # bias为全零


# k折交叉验证中取数据,将第i份作为验证集,其他的作为数据集返回
def get_k_fold_data(k, i, train_x, train_y):
    assert k > 1

    fold_size = train_x.shape[0] // k  # 一份的大小为数据集的数据量除以k

    # 训练集和验证集初始化为None
    cross_train_x, cross_train_y = None, None
    cross_validate_x, cross_validate_y = None, None

    for j in range(k):

        # 因为分了k份,所以要一份份取数据,当前的份数为j,下标从j * fold_size到(j + 1) * fold_size
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = train_x[idx], train_y[idx]

        # 如果j等于i,那么这份就应当当做验证集
        if j == i:
            cross_validate_x, cross_validate_y = x_part, y_part
        # 其余的情况当做训练集
        elif cross_train_x is None:
            cross_train_x, cross_train_y = x_part, y_part  # 当训练集为空也就是当前还没有份数分为训练集时,直接让训练集等于当前的切片
        else:  # 已经有数据集了,合并当前的数据集和当前的切片
            cross_train_x = torch.cat([cross_train_x, x_part], dim=0)
            cross_train_y = torch.cat([cross_train_y, y_part], dim=0)

    # 返回训练集和验证集
    return cross_train_x, cross_train_y, cross_validate_x, cross_validate_y


# k折交叉验证
def k_fold(k, train_x, train_y, cross_batch_size, epochs, lr, weight_decay, net):
    cross_train_l_sum, cross_validate_l_sum = 0.0, 0.0  # 训练损失和验证损失的总和(要做k次,先将每次的损失相加,最后除以k取均值返回)

    # k折交叉验证
    for i in range(k):
        # 先取出训练集和验证集
        cross_train_x, cross_train_y, cross_validate_x, cross_validate_y = get_k_fold_data(k, i, train_x, train_y)

        # dataloader
        cross_train_iter = d2l.load_array((cross_train_x, cross_train_y), batch_size=cross_batch_size)
        cross_validate_iter = d2l.load_array((cross_validate_x, cross_validate_y), batch_size=cross_batch_size,
                                             is_train=False)

        # 重新初始化网络,k折交叉验证一轮数据会跑k次,每次都需要重新初始化一下,否则后面的训练就是在前面的基础上继续训练,那么就没有对比的意义了
        cross_net = net

        # 将训练误差和验证误差累加
        cross_train_l_sum += train(epochs, cross_train_iter, lr, weight_decay, cross_net, True)
        cross_validate_l_sum += test(cross_validate_iter, cross_net)

    # 返回k次的误差均值
    return cross_train_l_sum / k, cross_validate_l_sum / k


# 训练函数
def train(epochs, train_iter, lr, weight_decay, net, is_cross_train=False):
    # 更新权重的updater,方法和Adam,和梯度下降类似,只不过不怎么需要调学习率
    updater = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # loss列表和epoch列表,方便画图
    loss_list = []
    epoch_list = []

    # 一共跑epochs轮数据
    for epoch in range(epochs):

        # 累加器,方便计算loss
        metric = d2l.Accumulator(2)

        # 从dataloader中取出数据(我起名叫iter是因为习惯问题)
        for x, y in train_iter:
            # 将数据移动到GPU上,用GPU进行训练
            x = x.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))

            # 权重更新
            updater.zero_grad()  # 每次进行loss计算时需要先将梯度清零
            y_hat = net(x)  # 计算y_hat
            l = log_rmse(y_hat, y.reshape(y_hat.shape))  # loss
            l.sum().backward()  # 反向传播算法
            updater.step()  # 梯度更新

            # 这一步不需要计算梯度,所以用torch.no_grad()括起来
            with torch.no_grad():
                metric.add(l.sum(), l.numel())  # 将当前batch的loss和数据量累加起来,因为我们的loss要求均值,等会整个epoch跑完后用总的loss除以数据量就得到了平均loss

        # loss均值
        epoch_loss = metric[0] / metric[1]
        # 将这个epoch跑完后的loss和epoch添加到刚刚的列表中,方便后面画图
        loss_list.append(epoch_loss)
        epoch_list.append(epoch + 1)

    # 如果不是交叉验证,就画一张图出来看看效果
    if not is_cross_train:
        plt.plot(epoch_list, loss_list)
        plt.show()

    return epoch_loss


# 测试,原理和训练类似,只是不用做梯度更新,只用计算loss
def test(test_iter, net):
    metric = d2l.Accumulator(2)

    for x, y in test_iter:
        with torch.no_grad():
            x = x.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))

            y_hat = net(x)
            l = log_rmse(y_hat, y.reshape(y_hat.shape))

            metric.add(l.sum(), l.numel())

    return metric[0] / metric[1]


# 一些参数的设置
cross_epochs = 100  # 训练的轮数
weight_decay = 0  # 权重衰退, 就是正则化项加的那个lambda
lr = 1300  # 学习率
cross_batch_size = 128  # 批量大小

# ----------------------------------------------------------------------------------------------------------------------


# # 交叉验证
#
#
# # 生成神经网络
# net = get_net()
# net.to(torch.device('cuda'))  # 将网络移动到GPU上
#
# # 进行5折交叉验证
# cross_train_loss, cross_validate_loss = k_fold(5, train_x, train_y, cross_batch_size, cross_epochs, lr, weight_decay,
#                                                net)
# # 打印loss
# print('lr:', lr, 'batch_size:', cross_batch_size)
# print('train loss:', cross_train_loss)
# print('validate loss:', cross_validate_loss)

# ----------------------------------------------------------------------------------------------------------------------


# # 普通的设单次验证集的验证
#
#
# fold_size = train_x.shape[0] // 2
# cross_train_x, cross_train_y = train_x[slice(0 * fold_size, 1 * fold_size)], train_y[
#     slice(0 * fold_size, 1 * fold_size)]
# cross_validate_x, cross_validate_y = train_x[slice(1 * fold_size, 2 * fold_size)], train_y[
#     slice(1 * fold_size, 2 * fold_size)]
#
# cross_train_iter = d2l.load_array((cross_train_x, cross_train_y), cross_batch_size)
# cross_validate_iter = d2l.load_array((cross_validate_x, cross_validate_y), cross_batch_size, False)
#
# net = get_net()
# net.to(torch.device('cuda'))
#
# print(train(cross_epochs, cross_train_iter, lr, weight_decay, net, True))
# print(test(cross_validate_iter, net))

# ----------------------------------------------------------------------------------------------------------------------


# 真正的训练模型


batch_size, epochs = 128, 100
lr = 1300

train_iter = d2l.load_array((train_x, train_y), batch_size)

net = get_net()
net.to(torch.device('cuda'))

print('loss:', train(epochs, train_iter, lr, weight_decay, net))

# ----------------------------------------------------------------------------------------------------------------------

# 保存预测的结果

test_x = test_x.to(torch.device('cuda'))
predict = net(test_x)
predict = predict.to(torch.device('cpu'))
predict = predict.detach().numpy()

sample_submission = pd.read_csv('california-house-prices/sample_submission.csv')
sample_submission['Sold Price'] = pd.Series(predict.reshape(1, -1)[0])
submission = pd.concat([sample_submission['Id'], sample_submission['Sold Price']], axis=1)
submission.to_csv('california-house-prices/submission.csv', index=False)
