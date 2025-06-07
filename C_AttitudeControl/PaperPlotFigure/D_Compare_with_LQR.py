import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import tyro
import matplotlib.pyplot as plt
from B_LNN_Learning_PaperPlot import ValueNet, ActionNet
from config import Args


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
dt = args.dt
Ixx = args.Ixx
Iyy = args.Iyy
Izz = args.Izz
tau_max = args.tau_max
r1 = args.R[0, 0]
r2 = args.R[1, 1]
r3 = args.R[2, 2]
q1 = args.Q[0, 0]
q2 = args.Q[1, 1]
q3 = args.Q[2, 2]
q4 = args.Q[3, 3]
q5 = args.Q[4, 4]
q6 = args.Q[5, 5]
S = args.S_matrix
Q = args.Q
R = args.R
A_matrix = args.A
B_matrix = args.B
K = args.K


action_net = ActionNet(input_dim=6)
action_net.load_checkpoint()
value_net = ValueNet(input_dim=6)
value_net.load_checkpoint()

DATA = np.load("../data/DATA.npz")
dataset = DATA['dataset']
u1_max = np.max(np.abs(dataset[:, -4]))
u2_max = np.max(np.abs(dataset[:, -3]))
u3_max = np.max(np.abs(dataset[:, -2]))
V_max = np.max(np.abs(dataset[:, -1]))

def NetController(x):
    phi, theta, psi, p, q, r = x
    state = np.array([phi, theta, psi, p, q, r]).reshape((6, 1))

    x = torch.Tensor([x]).requires_grad_(True)

    value = value_net(x)
    dVdx = torch.autograd.grad(value, x, torch.ones_like(value))
    dVdx = dVdx[0].detach().numpy() * V_max


    fx = np.array([
        [p + np.tan(theta) * np.sin(phi) * q + np.tan(theta) * np.cos(phi) * r],
        [np.cos(phi) * q + (-np.sin(phi) * r)],
        [np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r],
        [1.0 / Ixx * (q * r * (Iyy - Izz))],
        [1.0 / Iyy * (p * r * (Izz - Ixx))],
        [1.0 / Izz * (p * q * (Ixx - Iyy))]
    ])

    gx = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1.0 / Ixx, 0, 0],
        [0, 1.0 / Iyy, 0],
        [0, 0, 1.0 / Izz]
    ])


    if np.linalg.norm(state) < 0.1:  # switch to LQR when state is enough close to equilibrium.
        action = (- K @ state).reshape(-1)

    else:
        a_net = action_net(x)
        a_net = a_net.detach().numpy()[0] * np.array([u1_max, u2_max, u3_max])
        a_net = a_net.reshape((-1, 1))

        Ldot = dVdx @ (fx + gx @ a_net)
        A = dVdx @ gx
        b1 = Ldot
        b2 = -0.0001 * np.linalg.norm(A) * np.linalg.norm(state)

        if Ldot > 0:
            delta_u = A.T @ np.linalg.inv(A @ A.T) * (-b1 + b2)
            print("delta_u:", delta_u)
            print("##################################x:", np.linalg.norm(state))
        else:
            delta_u = 0

        action = (a_net + delta_u).reshape(-1)

    return action


def LQR_Controller(x):
    x.reshape((6, 1))
    action = (-K @ x).reshape(-1)
    action = np.clip(action, -tau_max, tau_max)
    return action



fig1 = plt.figure()
ax1_1 = fig1.add_subplot(411)
ax1_2 = fig1.add_subplot(412)
ax1_3 = fig1.add_subplot(413)
ax1_4 = fig1.add_subplot(414)
ax1_4.set_xticklabels([])
ax1_4.set_xlabel("test case")
ax1_4.set_ylabel("performance index")


def Plot(state, controller):
    theta0 = np.arctan2(state[3], state[0])
    V = 0
    t = 0
    dt = 0.005

    data = []
    while (t < 8):
        action = controller(state)

        phi, theta, psi, p, q, r = state
        tau_x, tau_y, tau_z = action

        phi_dot = p + np.tan(theta) * np.sin(phi) * q + np.tan(theta) * np.cos(phi) * r
        theta_dot = np.cos(phi) * q + (-np.sin(phi) * r)
        psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r
        p_dot = 1 / Ixx * (tau_x + q * r * (Iyy - Izz))
        q_dot = 1 / Iyy * (tau_y + p * r * (Izz - Ixx))
        r_dot = 1 / Izz * (tau_z + p * q * (Ixx - Iyy))

        state_dot = np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
        state = state + state_dot * dt

        t = t + dt
        V = V + (state.T @ Q @ state + action.T @ R @ action) * dt

        data.append(np.hstack([t, state, action, V]))

    # print(V)
    data = np.array(data)
    ax1_1.plot(data[:, 0], data[:, 1], label=controller.__name__)
    ax1_2.plot(data[:, 0], data[:, 4], label=controller.__name__)
    ax1_3.plot(data[:, 0], data[:, -4], label=controller.__name__)
    ax1_3.plot(data[:, 0], data[:, -3], label=controller.__name__)
    ax1_3.plot(data[:, 0], data[:, -2], label=controller.__name__)
    if controller.__name__ == "LQR_Controller":
        ax1_4.scatter(theta0, V, color="blue")
    else:
        ax1_4.scatter(theta0, V, color="red")

    return data


LQRData = []
for theta0 in np.linspace(0, 2*np.pi, 20):
    print(theta0)
    r0 = np.sqrt(3.9 ** 2 + 1.56 ** 2)
    x1 = r0 * np.cos(theta0)
    x1 = np.clip(x1, -1.56, 1.56)
    x4 = r0 * np.sin(theta0)
    x4 = np.clip(x4, -3.9, 3.9)
    state = np.array([x1, -0.74805067, -0.97876079, x4, 4.45423371, 0.40903922])# + np.random.uniform(-0.2, 0.2, (6,))
    data = Plot(state, controller=LQR_Controller)
    LQRData.append(data)
    plt.pause(0.1)
LQRData = np.array(LQRData)

NetData = []
for theta0 in np.linspace(0, 2*np.pi, 20):
    print(theta0)
    r0 = np.sqrt(3.9 ** 2 + 1.56 ** 2)
    x1 = r0 * np.cos(theta0)
    x1 = np.clip(x1, -1.56, 1.56)
    x4 = r0 * np.sin(theta0)
    x4 = np.clip(x4, -3.9, 3.9)
    state = np.array([x1, -0.74805067, -0.97876079, x4, 4.45423371, 0.40903922])# + np.random.uniform(-0.2, 0.2, (6,))
    data = Plot(state, controller=NetController)
    NetData.append(data)
    plt.pause(0.1)
NetData = np.array(NetData)

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

# 两轴居中
ax1.set_ylim([-3, 3])

ax2.set_ylim([-12, 12])

plt.tight_layout()

fig.savefig('image/comparison_ATT.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()