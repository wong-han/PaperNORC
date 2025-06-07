import numpy as np
import torch
import tyro
import matplotlib.pyplot as plt
import matplotlib as mpl
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



fig1_1 = plt.figure(figsize=(3.5, 3.5))
ax1_1 = fig1_1.add_subplot(211)
ax1_1.set_xlabel(r'$t(s)$', usetex=True)
ax1_1.set_ylabel(r'$\phi$(rad)', usetex=True)
ax1_1.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

fig1_2 = plt.figure(figsize=(3.5, 3.5))
ax1_2 = fig1_2.add_subplot(211)
ax1_2.set_xlabel(r'$t(s)$', usetex=True)
ax1_2.set_ylabel(r'$\dot\phi$(rad/s)', usetex=True)
ax1_2.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

fig1_3 = plt.figure(figsize=(3.5, 5.2))
ax3_1 = fig1_3.add_subplot(311)
ax3_1.set_ylabel(r'$\tau_x$(Nm)', usetex=True)
ax3_1.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
# ax3_1.set_xticklabels([])
ax3_2 = fig1_3.add_subplot(312)
ax3_2.set_ylabel(r'$\tau_y$(Nm)', usetex=True)
ax3_2.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
# ax3_2.set_xticklabels([])
ax3_3 = fig1_3.add_subplot(313)
ax3_3.set_xlabel(r'$t(s)$', usetex=True)
ax3_3.set_ylabel(r'$\tau_z$(Nm)', usetex=True)
ax3_3.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)



# 定义颜色循环和线型（用于区分不同轨迹）
color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
line_styles = ['-', '--', '-.', ':']  # 不同线型

def Plot(state, controller):
    theta0 = np.arctan2(state[3], state[0])
    V = 0
    t = 0
    dt = 0.005

    data = []
    while (t < 7):
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
    return data


for count, theta0 in enumerate(np.linspace(0, 2*np.pi, 20)):
    print(theta0)
    r0 = np.sqrt(3.9 ** 2 + 1.56 ** 2)
    x1 = r0 * np.cos(theta0)
    x1 = np.clip(x1, -1.56, 1.56)
    x4 = r0 * np.sin(theta0)
    x4 = np.clip(x4, -3.9, 3.9)
    state = np.array([x1, -0.74805067, -0.97876079, x4, 4.45423371, 0.40903922])# + np.random.uniform(-0.2, 0.2, (6,))
    data = Plot(state, controller=NetController)

    ax1_1.plot(data[:, 0], data[:, 1],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')
    ax1_2.plot(data[:, 0], data[:, 4],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')
    ax3_1.plot(data[:, 0], data[:, -4],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')
    ax3_2.plot(data[:, 0], data[:, -3],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')
    ax3_3.plot(data[:, 0], data[:, -2],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')

    plt.pause(0.01)


fig1_1.savefig('image/phi_ATT.pdf', bbox_inches='tight', pad_inches=0.1)
fig1_2.savefig('image/phi_dot_ATT.pdf', bbox_inches='tight', pad_inches=0.1)
fig1_3.savefig('image/u_ATT.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
