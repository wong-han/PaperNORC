import numpy as np
import torch
import matplotlib.pyplot as plt
from B_LNN_Learning_PaperPlot import ValueNet, ActionNet

from B_WingedConeHeightControl.config import Args
import tyro
import matplotlib as mpl
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

def NetController(x):
    x1, x2 = x

    x = (x - np.array([110000, 0])) / np.array([delta_x1_max, delta_x2_max])
    x = torch.Tensor([x]).requires_grad_(True)

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
        delta_u = (-0.00001*np.abs((dVdx @ gx)) * np.linalg.norm([x1-110000, x2]) - Ldot) / (dVdx @ gx)  # \Delta u=\frac{-k\lVert x \rVert _2-\frac{\text{d}V}{\text{d}x}\left( f\left( x \right) +g\left( x \right) a_{net} \right)}{\frac{\text{d}V}{\text{d}x}g\left( x \right)}
        delta_u = delta_u[0][0]*0
        # print("delta_u:", delta_u)
    else:
        delta_u = 0

    alpha = np.clip(a_net + delta_u, -alpha_max_rad, alpha_max_rad)
    return alpha


def LQR_Controller(x):
    alpha = - K @ (x - np.array([110000, 0])) + alpha0_rad
    alpha = alpha[0]
    alpha = np.clip(alpha, -alpha_max_rad, alpha_max_rad)
    return alpha


fig1_1 = plt.figure(figsize=(3.5, 3.5))
ax1_1 = fig1_1.add_subplot(211)
ax1_1.set_xlabel(r'$t(s)$', usetex=True)
ax1_1.set_ylabel(r'$h$(ft)', usetex=True)
ax1_1.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

fig1_2 = plt.figure(figsize=(3.5, 3.5))
ax1_2 = fig1_2.add_subplot(211)
ax1_2.set_xlabel(r'$t(s)$', usetex=True)
ax1_2.set_ylabel(r'$\dot h$(ft/s)', usetex=True)
ax1_2.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

fig1_3 = plt.figure(figsize=(3.5, 3.5))
ax1_3 = fig1_3.add_subplot(211)
ax1_3.set_xlabel(r'$t(s)$', usetex=True)
ax1_3.set_ylabel(r'$\alpha(^\circ)$', usetex=True)
ax1_3.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

# 定义颜色循环和线型（用于区分不同轨迹）
color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
line_styles = ['-', '--', '-.', ':']  # 不同线型


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


for count, theta in enumerate(np.linspace(0, 2*np.pi, 20)):
    print(theta)

    r0 = 1562
    delta_x1 = np.clip(r0 * np.cos(theta), -1534, 1534)
    delta_x2 = np.clip(r0 * np.sin(theta), -290, 290)

    x1 = 110000 + delta_x1
    x2 = delta_x2
    data = Plot(x1, x2, controller=NetController)

    ax1_1.plot(data[:, 0], data[:, 1],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')
    ax1_2.plot(data[:, 0], data[:, 2],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')
    ax1_3.plot(data[:, 0], data[:, -2]*57.3,
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')

    plt.pause(0.01)




fig1_1.savefig('image/x1_WCC.pdf', bbox_inches='tight', pad_inches=0.1)
fig1_2.savefig('image/x2_WCC.pdf', bbox_inches='tight', pad_inches=0.1)
fig1_3.savefig('image/u_WCC.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
