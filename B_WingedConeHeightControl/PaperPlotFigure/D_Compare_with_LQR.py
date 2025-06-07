import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import torch
import matplotlib.pyplot as plt
from B_LNN_Learning_PaperPlot import ValueNet, ActionNet

from config import Args
import tyro

mpl.rcParams['text.usetex'] = True  # 使用Latex语法
mpl.rcParams['font.family'] = 'simsun'  # 解决中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号

mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴

mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 9  # legend默认size

mpl.rcParams['xtick.labelsize'] = 9  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 9  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 9  # 轴标题默认size


args = tyro.cli(Args)
R = args.R
Q = args.Q
K = args.K
alpha0_rad = args.alpha0_rad
alpha_max_rad = args.alpha_max_rad

action_net = ActionNet()
action_net.load_checkpoint()
value_net = ValueNet()
value_net.load_checkpoint()


DATA = np.load("../data/DATA.npz")
dataset = DATA['dataset']
delta_x1_max = np.max(np.abs(dataset[:, 0] - 110000))
delta_x2_max = np.max(np.abs(dataset[:, 1]))
V_max = np.max(dataset[:, -1])
u_max = np.max(np.abs(dataset[:, -2]))
lambda_max = np.max(np.abs(dataset[:, 3]))

def NetController(x):
    x1, x2 = x

    x = (x - np.array([110000, 0])) / np.array([delta_x1_max, delta_x2_max])
    x = torch.Tensor(np.array([x])).requires_grad_(True)

    value = value_net(x)
    dVdx = torch.autograd.grad(value, x, torch.ones_like(value))
    dVdx = dVdx[0].detach().numpy() * V_max / np.array([[delta_x1_max, delta_x2_max]])

    fx = np.array([
        [x2],
        [- 20.685555 * (1 - (x2 / 15060) ** 2)]
    ])

    gx = np.array([
        [0],
        [64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2)]
    ])

    a_net = action_net(x)
    a_net = a_net.detach().numpy()[0][0] * u_max

    Ldot = dVdx @ (fx + gx * a_net)
    if Ldot >= 0:
        delta_u = (-0.000001*np.abs((dVdx @ gx)) * np.linalg.norm([x1-110000, x2]) - Ldot) / (dVdx @ gx)  # \Delta u=\frac{-k\lVert x \rVert _2-\frac{\text{d}V}{\text{d}x}\left( f\left( x \right) +g\left( x \right) a_{net} \right)}{\frac{\text{d}V}{\text{d}x}g\left( x \right)}
        delta_u = delta_u[0][0]
    else:
        delta_u = 0
    alpha = np.clip(a_net + delta_u, -alpha_max_rad, alpha_max_rad)
    return alpha


def LQR_Controller(x):
    alpha = - K @ (x - np.array([110000, 0])) + alpha0_rad
    alpha = alpha[0]
    alpha = np.clip(alpha, -alpha_max_rad, alpha_max_rad)
    return alpha


fig1 = plt.figure()
ax1_1 = fig1.add_subplot(311)
ax1_2 = fig1.add_subplot(312)
ax1_3 = fig1.add_subplot(313)

fig2 = plt.figure()
ax2_1 = fig2.add_subplot(111)
ax2_1.set_xlabel("test case")
ax2_1.set_ylabel("performance index")
ax2_1.set_xticklabels([])


def Plot(x1, x2, controller):

    V = 0
    t = 0
    dt = 0.005

    data = []
    while (t < 50):
        alpha_star = controller(np.array([x1, x2]))

        x1_dot = x2
        x2_dot = 64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * alpha_star - 20.685555 * (1 - (x2 / 15060) ** 2)

        x1 = x1 + x1_dot * dt
        x2 = x2 + x2_dot * dt
        t = t + dt
        V = V + (Q[0, 0] * (x1-110000)**2 + Q[1, 1] * x2**2 + R * (alpha_star-alpha0_rad)**2)*dt

        data.append([t, x1, x2, alpha_star, V])


    # print("V:", V)
    data = np.array(data)
    return data


r0 = 1561.6167350492176
theta0 = -0.18762762291098825
x1_limit = np.abs(r0 * np.cos(theta0))
x2_limit = np.abs(r0 * np.sin(theta0))
NetData = []
LQRData = []
for count, theta in enumerate(np.linspace(theta0, theta0+2*np.pi, 20)):
    print("test_case:", count)
    delta_x1 = np.clip(r0 * np.cos(theta), -x1_limit, x1_limit)
    delta_x2 = np.clip(r0 * np.sin(theta), -x2_limit, x2_limit)


    x1 = 110000 + delta_x1
    x2 = delta_x2
    data = Plot(x1, x2, controller=NetController)
    NetData.append(data)

    ax1_1.plot(data[:, 0], data[:, 1])
    ax1_2.plot(data[:, 0], data[:, 2])
    ax1_3.plot(data[:, 0], np.rad2deg(data[:, -2]))
    ax2_1.scatter(theta, data[-1, -1], color='red')


    delta_x1 = np.clip(r0 * np.cos(theta), -1500, 1500)
    delta_x2 = np.clip(r0 * np.sin(theta), -280, 280)
    delta_x1 = np.clip(r0 * np.cos(theta), -x1_limit, x1_limit)
    delta_x2 = np.clip(r0 * np.sin(theta), -x2_limit, x2_limit)


    x1 = 110000 + delta_x1
    x2 = delta_x2
    data = Plot(x1, x2, controller=LQR_Controller)
    LQRData.append(data)

    ax1_1.plot(data[:, 0], data[:, 1])
    ax1_2.plot(data[:, 0], data[:, 2])
    ax1_3.plot(data[:, 0], np.rad2deg(data[:, -2]))
    ax2_1.scatter(theta, data[-1, -1], color='blue')

    plt.pause(0.01)

NetData = np.array(NetData)
LQRData = np.array(LQRData)
np.savez('../data/Simulation.npz', NetData=NetData, LQRData=LQRData)



DATA = np.load('../data/Simulation.npz')
NetData = DATA['NetData']
LQRData = DATA['LQRData']

Net_J = NetData[:, -1, -1]
LQR_J = LQRData[:, -1, -1]

# 计算误差指标
absolute_error = Net_J - LQR_J
relative_error = (absolute_error / LQR_J) * 100  # 百分比形式

# 配置可视化参数
colors = plt.cm.Paired.colors  # 提取Paired色板
absolute_color = '#1f77b4'
relative_color = '#ff7f0e'

bar_width = 0.4  # 加宽柱宽
case_num = len(Net_J)
x = np.arange(case_num)  # 生成x轴位置

# 创建画布和双轴
fig, ax1 = plt.subplots(figsize=(7, 3))
ax2 = ax1.twinx()

# 绘制绝对误差柱状图（左轴）
bars1 = ax1.bar(x , -absolute_error, bar_width,
                color=absolute_color, edgecolor='black',
                label='Absolute Error')

# 绘制相对误差柱状图（右轴）
bars2 = ax2.bar(x, relative_error, bar_width,
                color=relative_color, edgecolor='black',
                label='Relative Error (\%)')

# 设置左轴样式
ax1.set_ylabel(r'$J_{LQR} - J_{Ours}$', color=absolute_color)
ax1.tick_params(axis='y', colors=absolute_color)

# 设置右轴样式
ax2.set_ylabel('Cost Reduction (\%)', color=relative_color)
ax2.tick_params(axis='y', colors=relative_color)

# 统一横轴设置
ax1.set_xlabel('Test Cases')
ax1.set_xticks(x)
ax1.set_xticklabels([])  # 隐藏具体case标签
ax1.set_xlim(-0.5, case_num-0.5)  # 精确控制x轴范围

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2,
           loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.15),
           frameon=False)

# 添加辅助网格线
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# # 两轴居中
ax1.set_ylim([-260, 260])
ax2.set_ylim([-35, 35])

plt.tight_layout()

fig.savefig('image/comparison_WCC.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()