import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import math
# import tyro
# from config import Args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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

        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = x - torch.zeros_like(x)
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

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
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
    value_net = ValueNet(input_dim=6)
    # value_net.load_checkpoint()
    total_epochs = 10000  # 训练轮次
    batch_size = 512       # 小批量
    loss_list = []

    DATA = np.load("../data/DATA.npz")
    dataset = DATA['dataset']
    dataset[:, -1] = dataset[:, -1] / np.max(np.abs(dataset[:, -1]))

    # index = np.random.choice(dataset.shape[0], 20000, replace=False)
    # dataset = dataset[index, :]
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, -1])
    # plt.show()

    index_pool = np.random.permutation(dataset.shape[0])  # 打乱顺序
    for i in range(total_epochs):
        # 随机取数据
        batch_order = i % (dataset.shape[0] // batch_size)
        index = index_pool[batch_size * batch_order:batch_size * (batch_order + 1)]
        input_array = dataset[index, 0:6]
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
    value_net = ValueNet(input_dim=6)
    value_net.load_checkpoint()
    value_net.eval()


    # 创建3D曲面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    DATA = np.load("../data/DATA.npz")
    dataset = DATA['dataset']
    dataset[:, -1] = dataset[:, -1] / np.max(np.abs(dataset[:, -1]))
    index = np.random.choice(dataset.shape[0], 20000, replace=False)
    dataset = dataset[index, :]
    ax.scatter(dataset[:, 0], dataset[:, 3], dataset[:, -1])

    input_tensor = torch.FloatTensor(dataset[:, 0:6])
    output_tensor = value_net(input_tensor).detach().numpy()
    ax.scatter(dataset[:, 0], dataset[:, 3], output_tensor[:, -1], c='r')

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
            nn.Linear(64, 3),
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
    action_net = ActionNet(input_dim=6)
    # action_net.load_checkpoint()
    total_epochs = 10000
    batch_size = 512
    loss_list = []

    DATA = np.load("../data/DATA.npz")
    dataset = DATA['dataset']

    dataset[:, -4] = dataset[:, -4] / np.max(np.abs(dataset[:, -4]))
    dataset[:, -3] = dataset[:, -3] / np.max(np.abs(dataset[:, -3]))
    dataset[:, -2] = dataset[:, -2] / np.max(np.abs(dataset[:, -2]))

    # index = np.random.choice(dataset.shape[0], 20000, replace=False)
    # dataset = dataset[index, :]
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(dataset[:, 0], dataset[:, 3], dataset[:, -4])
    # plt.show()

    index_pool = np.random.permutation(dataset.shape[0])  # 打乱顺序
    for i in range(total_epochs):
        # 随机取数据
        batch_order = i % (dataset.shape[0] // batch_size)
        index = index_pool[batch_size * batch_order:batch_size * (batch_order + 1)]
        input_array = dataset[index, 0:6]
        input_data = torch.tensor(input_array, dtype=torch.float)
        output_data = torch.tensor(dataset[index, -4:-1], dtype=torch.float)

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
    action_net = ActionNet(input_dim=6)
    action_net.load_checkpoint()
    action_net.eval()

    # 创建3D曲面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    DATA = np.load("../data/DATA.npz")
    dataset = DATA['dataset']
    dataset[:, -4] = dataset[:, -4] / np.max(np.abs(dataset[:, -4]))
    dataset[:, -3] = dataset[:, -3] / np.max(np.abs(dataset[:, -3]))
    dataset[:, -2] = dataset[:, -2] / np.max(np.abs(dataset[:, -2]))

    index = np.random.choice(dataset.shape[0], 20000, replace=False)
    dataset = dataset[index, :]
    ax.scatter(dataset[:, 0], dataset[:, 3], dataset[:, -2])

    input_tensor = torch.FloatTensor(dataset[:, 0:6])
    output_tensor = action_net(input_tensor).detach().numpy()
    ax.scatter(dataset[:, 0], dataset[:, 3], output_tensor[:, 2], c='r')

    plt.show()


if __name__ == '__main__':

    # ValueNetTrain()
    # ValueNetTest()

    # ActionNetTrain()
    ActionNetTest()





















