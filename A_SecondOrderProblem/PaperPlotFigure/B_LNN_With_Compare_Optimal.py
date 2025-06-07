import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import math
import matplotlib as mpl

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

mpl.rcParams['text.usetex'] = True  # 使用Latex语法
mpl.rcParams['font.family'] = 'simsun'  # 解决中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号

mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴

# mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 8  # legend默认size

mpl.rcParams['xtick.labelsize'] = 9  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 9  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 9  # 轴标题默认size

class ValueNet(nn.Module):
    def __init__(self, input_dim=2, name='value_net'):
        super(ValueNet, self).__init__()
        self.name = name
        self.input_dim = input_dim

        self.generator = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.input_dim * (self.input_dim + 1) // 2),
        )
        self.indices = [(i, j) for i in range(self.input_dim) for j in range(i + 1)]

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = x - torch.Tensor([0, 0])
        batch_size = x.shape[0]
        elements = self.generator(x)
        L = torch.zeros(batch_size, self.input_dim, self.input_dim)
        for idx, (i, j) in enumerate(self.indices):
            L[:, i, j] = elements[:, idx]

        L_T = L.transpose(1, 2)
        intermediate = torch.bmm(x.unsqueeze(1), L_T).squeeze(1)
        value = (intermediate ** 2).sum(dim=1)
        value = value.unsqueeze(dim=1)
        return value

    def save_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' saved.')
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' loaded.')
        self.load_state_dict(torch.load(filename))


class ValueNet2(nn.Module):
    def __init__(self, input_dim=2, name='value_net'):
        super(ValueNet2, self).__init__()
        self.name = name
        self.input_dim = input_dim

        self.dnet = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        zero_point = torch.zeros_like(x)
        value = torch.sum(torch.square(self.dnet(x) - self.dnet(zero_point)), dim=1)
        value = value.unsqueeze(dim=1)
        return value

    def save_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' saved.')
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' loaded.')
        self.load_state_dict(torch.load(filename))


class DNN(nn.Module):
    def __init__(self, input_dim=2, name='dnn'):
        super(DNN, self).__init__()
        self.name = name
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        value = self.net(x)
        return value

    def save_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' saved.')
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' loaded.')
        self.load_state_dict(torch.load(filename))


def ValueNetTrain():
    value_net = ValueNet(input_dim=2)
    # value_net.load_checkpoint()
    total_epochs = 50000  # 训练轮次
    batch_size = 512       # 小批量
    loss_list = []

    DATA = np.load("../data/DATA.npz")
    dataset = DATA['dataset']

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, -1])
    # plt.show()


    index_pool = np.random.permutation(dataset.shape[0])  # 打乱顺序
    for i in range(total_epochs):
        # 随机取数据
        batch_order = i % (dataset.shape[0] // batch_size)
        index = index_pool[batch_size * batch_order:batch_size * (batch_order + 1)]
        x1_array = dataset[index, 0:1]
        x2_array = dataset[index, 1:2]
        input_array = np.hstack((x1_array, x2_array))
        input_data = torch.tensor(input_array, dtype=torch.float)
        output_data = torch.tensor(dataset[index, -1:], dtype=torch.float)

        pred_out = value_net.forward(input_data)
        loss = value_net.loss(pred_out, output_data)
        value_net.zero_grad()
        loss.backward()
        loss_list.append(loss.detach().numpy())
        value_net.optimizer.step()

        print('\r' 'epoch:%d, ' % i + "loss:%6f" % loss, end='')

        if i % 1000 == 0 and i > 0:
            value_net.save_checkpoint()


def ValueNetTest():
    value_net = ValueNet(input_dim=2)
    value_net.load_checkpoint()
    value_net.eval()

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_tensor = torch.FloatTensor(np.column_stack((x_grid.ravel(), y_grid.ravel())))
    with torch.no_grad():
        z = abs(value_net(grid_tensor).numpy()[:, 0] - (0.5*np.square(x_grid.ravel()) + np.square(y_grid.ravel())))
    z_grid = z.reshape(x_grid.shape)

    # 创建3D曲面图
    fig = plt.figure(figsize=(3.5, 2.9))
    ax = fig.add_subplot(111)

    # 绘制曲面
    surf = ax.contourf(x_grid, y_grid, z_grid,
                           cmap='coolwarm',  # 改用高对比度plasma色图
                           alpha=1,  # 曲面半透明
                           antialiased=True)

    r0 = 3.667
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(r0*np.cos(theta), r0*np.sin(theta), color='black', linestyle='--')

    # 添加坐标轴标签
    ax.set_xlabel('$x_1$', usetex=True)
    ax.set_ylabel('$x_2$', usetex=True)
    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')
    ax.axis('equal')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.03)
    cbar.set_label(r'$\left| V_{net}-V^* \right|$', rotation=270, labelpad=13, usetex=True)


    fig.savefig('image/value_function_SOP_Compare.pdf', bbox_inches='tight')
    plt.show()


class ActionNet(nn.Module):
    def __init__(self, input_dim=2, name='action_net'):
        super(ActionNet, self).__init__()
        self.name = name
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        action = self.net(x)
        return action

    def save_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' saved.')
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self):
        filename = '../model/' + self.name
        print(filename + ' loaded.')
        self.load_state_dict(torch.load(filename))


def ActionNetTrain():
    action_net = ActionNet(input_dim=2)
    # action_net.load_checkpoint()
    total_epochs = 50000
    batch_size = 512
    loss_list = []

    DATA = np.load("../data/DATA.npz")
    dataset = DATA['dataset']

    index_pool = np.random.permutation(dataset.shape[0])  # 打乱顺序
    for i in range(total_epochs):
        # 随机取数据
        batch_order = i % (dataset.shape[0] // batch_size)
        index = index_pool[batch_size * batch_order:batch_size * (batch_order + 1)]
        x1_array = dataset[index, 0:1]
        x2_array = dataset[index, 1:2]
        input_array = np.hstack((x1_array, x2_array))
        input_data = torch.tensor(input_array, dtype=torch.float)
        output_data = torch.tensor(dataset[index, 4:5], dtype=torch.float)

        pred_out = action_net.forward(input_data)
        loss = action_net.loss(pred_out, output_data)
        action_net.zero_grad()
        loss.backward()
        loss_list.append(loss.detach().numpy())
        action_net.optimizer.step()

        print('\r' 'epoch:%d, ' % i + "loss:%6f" % loss, end='')

        if i % 1000 == 0 and i > 0:
            action_net.save_checkpoint()


def ActionNetTest():
    action_net = ActionNet(input_dim=2)
    action_net.load_checkpoint()
    action_net.eval()

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_tensor = torch.FloatTensor(np.column_stack((x_grid.ravel(), y_grid.ravel())))
    with torch.no_grad():
        z = abs(action_net(grid_tensor).numpy()[:, 0] - (-(np.cos(2*x_grid.ravel()) + 2) * y_grid.ravel()))
    z_grid = z.reshape(x_grid.shape)

    # 创建3D曲面图
    fig = plt.figure(figsize=(3.5, 2.9))
    ax = fig.add_subplot(111)

    # 绘制曲面
    surf = ax.contourf(x_grid, y_grid, z_grid,
                       cmap='coolwarm',  # 改用高对比度plasma色图
                       alpha=1,  # 曲面半透明
                       levels=10,
                       antialiased=True)

    r0 = 3.667
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(r0 * np.cos(theta), r0 * np.sin(theta), color='black', linestyle='--')

    # 添加坐标轴标签
    ax.set_xlabel('$x_1$', usetex=True)
    ax.set_ylabel('$x_2$', usetex=True)
    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')
    ax.axis('equal')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.03)
    cbar.set_label(r'$\left| u_{net}-u^* \right|$', rotation=270, labelpad=13, usetex=True)

    fig.savefig('image/action_SOP_Compare.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # ValueNetTrain()
    ValueNetTest()

    # ActionNetTrain()
    ActionNetTest()