'''
Generate Data
'''
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
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

# print(mpl.rcParams.keys())  # 查看画图时有哪些可以设置的默认参数


dt = 0.001


def system(t, state):
    x1, x2, lam1, lam2 = state[0:4]
    phi = state[4:20].reshape(4, 4)

    u = -0.5 * lam2 * (np.cos(2 * x1) + 2)

    x1_dot = -x1 + x2
    x2_dot = -0.5 * x1 - 0.5 * x2 * (1 - (np.cos(2 * x1) + 2) ** 2) + (np.cos(2 * x1) + 2) * u
    lam1_dot = lam1 - 2*x1 + lam2*(2*u*np.sin(2*x1) + 2*x2*np.sin(2*x1)*(np.cos(2*x1) + 2) + 1/2)
    lam2_dot = - lam1 - 2*x2 - lam2*((np.cos(2*x1) + 2)**2/2 - 1/2)

    F = np.array([
        [-1, 1, 0, 0],
        [2*lam2*np.sin(2*x1)*(np.cos(2*x1) + 2) - 2*x2*np.sin(2*x1)*(np.cos(2*x1) + 2) - 1/2,        (np.cos(2*x1) + 2)**2/2 - 1/2,  0,     -(np.cos(2*x1) + 2)**2/2],
        [lam2*(2*lam2*np.sin(2*x1)**2 - 4*x2*np.sin(2*x1)**2 - 2*lam2*np.cos(2*x1)*(np.cos(2*x1) + 2) + 4*x2*np.cos(2*x1)*(np.cos(2*x1) + 2)) - 2, 2*lam2*np.sin(2*x1)*(np.cos(2*x1) + 2),  1, 2*x2*np.sin(2*x1)*(np.cos(2*x1) + 2) - 2*lam2*np.sin(2*x1)*(np.cos(2*x1) + 2) + 1/2],
        [2*lam2*np.sin(2*x1)*(np.cos(2*x1) + 2),  -2, -1,   1/2 - (np.cos(2*x1) + 2)**2/2]
    ])

    phi_dot = F @ phi

    dstatedt = -np.array([x1_dot, x2_dot, lam1_dot, lam2_dot])
    dphidt = -phi_dot

    return np.hstack((dstatedt, dphidt.flatten()))


def inverse_integ(x10, x20, T):

    t = 0
    x1 = x10
    x2 = x20
    lambda1 = x1
    lambda2 = 2 * x2
    phi = np.eye(4)

    action = -0.5 * lambda2 * (np.cos(2 * x1) + 2)
    V = 0.5*x1**2 + x2**2

    data = [np.hstack([t, x1, x2, lambda1, lambda2, phi.flatten(), action, V])]

    while t < T:

        state = np.hstack(([x1, x2, lambda1, lambda2], phi.flatten()))
        dstatedt = system(t, state)

        V = V + (x1**2 + x2**2 + action**2) * dt
        x1 = x1 + dstatedt[0] * dt
        x2 = x2 + dstatedt[1] * dt
        lambda1 = lambda1 + dstatedt[2] * dt
        lambda2 = lambda2 + dstatedt[3] * dt
        phi = phi + dstatedt[4:20].reshape(4, 4) * dt
        t = t + dt
        action = -0.5 * lambda2 * (np.cos(2 * x1) + 2)

        data.append(np.hstack([t, x1, x2, lambda1, lambda2, phi.flatten(), action, V]))

    data = np.array(data)
    return data


# <editor-fold desc="STM Guided">

fig2 = plt.figure(figsize=(4, 3))
ax2_1 = fig2.add_subplot(111)
ax2_1.axis('equal')

# 设置坐标轴范围和比例
ax2_1.set_xlim(-4, 4)
ax2_1.set_ylim(-4, 4)
ax2_1.set_aspect('equal', adjustable='box')

# 设置坐标轴标签和标题
ax2_1.set_xlabel(r'$x_1$', usetex=True)
ax2_1.set_ylabel(r'$x_2$', usetex=True)

# 设置网格样式
ax2_1.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

# 定义颜色循环和线型（用于区分不同轨迹）
color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
line_styles = ['-', '--', '-.', ':']  # 不同线型



x10 = 0.01
x20 = 0.0
T = 1.8

data = inverse_integ(x10, x20, T)
ax2_1.plot(data[:, 1], data[:, 2])
state = data[-1, 1:5]
r0 = np.sqrt(state[0]**2 + state[1]**2)
theta0 = np.arctan2(state[1], state[0])
x1d = r0 * np.cos(theta0)
x2d = r0 * np.sin(theta0)
DATA = [data]
delta_theta = 0.01
total_delta_theta = 0
count = 0
traj_counter = 0
# 修改散点样式
ax2_1.scatter(0, 0, marker='o', s=15, c='r', zorder=3, label='equilibrium point')
ax2_1.scatter(x1d, x2d,
              color='k',  # '#d62728',  # 使用醒目的红色
              marker='+',  # 改用x标记
              s=15,  # 增大标记尺寸
              zorder=3,  # 确保标记在前景
              label='desired point'
              )
fig2.legend(loc=(0.095, 0.91), ncols=2, frameon=False)

fig2.savefig('image/optimal_data_SOP.pdf', bbox_inches='tight', pad_inches=0.1)
while True:
    phi = data[-1][5:21].reshape(4, 4)
    LU = phi[0:2, 0:2]
    RU = phi[0:2, 2:4]
    S = np.array([[0.5, 0], [0, 1]])
    STM = LU + 2 * RU @ S
    print(STM)

    # delta_theta = np.clip(1 / np.linalg.norm(STM, ord=np.inf), -0.01, 0.01)

    theta_d = theta0 + delta_theta
    theta0 = theta_d
    total_delta_theta += delta_theta
    print(total_delta_theta)
    if abs(total_delta_theta) > np.pi*2:
        break
    x1d = r0 * np.cos(theta_d)
    x2d = r0 * np.sin(theta_d)
    delta_xd = np.array([x1d, x2d]) - data[-1, 1:3]
    delta_x = np.linalg.inv(STM) @ delta_xd
    x10 = x10 + delta_x[0]
    x20 = x20 + delta_x[1]

    # Adjust time and terminal state
    flag = np.linalg.norm([x10, x20]) > 0.2
    flag = flag * 2 - 1
    while flag * np.linalg.norm([x10, x20]) > flag * 0.2:
        lambda1, lambda2 = 2 * S @ np.array([x10, x20])
        u = -0.5 * lambda2 * (np.cos(2 * x10) + 2)

        x1_dot = -x10 + x20
        x2_dot = -0.5 * x10 - 0.5 * x20 * (1 - (np.cos(2 * x10) + 2) ** 2) + (np.cos(2 * x10) + 2) * u

        x10 = x10 + x1_dot * dt * flag
        x20 = x20 + x2_dot * dt * flag

        T = T + dt * flag

    # LQR Part
    state_in = np.array([x10, x20])
    in_traj = []
    while np.linalg.norm(state_in) > 0.01:
        V = state_in.T @ S @ state_in
        lambda1, lambda2 = 2 * S @ state_in
        in_traj.append(np.hstack([0, state_in[0], state_in[1], lambda1, lambda2, np.zeros(16), u, V]))

        u = -0.5 * lambda2 * (np.cos(2 * state_in[0]) + 2)

        x1_dot = -state_in[0] + state_in[1]
        x2_dot = -0.5 * state_in[0] - 0.5 * state_in[1] * (1 - (np.cos(2 * state_in[0]) + 2) ** 2) + (np.cos(2 * state_in[0]) + 2) * u

        state_in = state_in + np.array([x1_dot, x2_dot]) * dt

    data = inverse_integ(x10, x20, T)

    if np.linalg.norm(np.array([x1d, x2d]) - data[-1, 1:3]) < 0.02:
        count += 1
        delta_theta = 0.01
        if count % 10 == 0:
            # 修改轨迹绘制部分
            ax2_1.plot(data[:, 1], data[:, 2],
                       color=color_cycle[traj_counter % len(color_cycle)],
                       # linestyle=line_styles[traj_counter % len(line_styles)],
                       linewidth=1.5,
                       alpha=0.8,
                       label=f'Trajectory {traj_counter + 1}')

            in_traj_array = np.array(in_traj)
            # 修改轨迹绘制部分
            ax2_1.plot(in_traj_array[:, 1], in_traj_array[:, 2],
                       color=color_cycle[traj_counter % len(color_cycle)],
                       # linestyle=line_styles[traj_counter % len(line_styles)],
                       linewidth=1.5,
                       alpha=0.8,
                       label=f'Trajectory {traj_counter + 1}')
            # 修改散点样式
            ax2_1.scatter(x1d, x2d,
                          color='k',  # '#d62728',  # 使用醒目的红色
                          marker='+',  # 改用x标记
                          s=15,  # 增大标记尺寸
                          zorder=3)  # 确保标记在前景

            traj_counter += 1
            DATA.append(data)
            DATA.append(in_traj_array)
    else:
        delta_theta = 0.0

    plt.pause(0.01)
    print(x10, x20, T)

temp_data = DATA
length = len(temp_data)
while length != 1:
    temp_temp_data = []
    for i in range(0, len(temp_data), 2):
        if i == len(temp_data) - 1:
            temp_temp_data.append(temp_data[i])
        else:
            temp_temp_data.append(np.vstack((temp_data[i], temp_data[i + 1])))
    temp_data = temp_temp_data
    length = len(temp_data)

dataset = temp_data[0]

dataset = np.array(dataset)
index = [1, 2, 3, 4, -2, -1]
dataset = dataset[:, index]
np.savez("../data/DATA.npz", dataset=dataset)

fig2.savefig('image/optimal_data_SOP.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
# </editor-fold>