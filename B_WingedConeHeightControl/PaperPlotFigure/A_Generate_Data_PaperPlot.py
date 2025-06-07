'''
LQR + BGOE
'''
import tyro
import numpy as np

from B_WingedConeHeightControl.config import Args
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

fig = plt.figure(figsize=[3.5, 3.5])
ax1_0 = fig.add_subplot(111)
# 设置网格样式
ax1_0.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

# 定义颜色循环和线型（用于区分不同轨迹）
color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
line_styles = ['-', '--', '-.', ':']  # 不同线型

# 设置坐标轴标签和标题
ax1_0.set_xlabel(r'$h$(ft)', usetex=True)
ax1_0.set_ylabel(r'$\dot h$(ft/s)', usetex=True)

args = tyro.cli(Args)
dt = args.dt
R = args.R
Q = args.Q
q1 = args.Q[0, 0]
q2 = args.Q[1, 1]
S = args.S_matrix

mu = args.mu
Re = args.Re
hd = args.hd
v = args.vd

alpha0_rad = args.alpha0_rad


def system(Z):
    z = Z[0:4]
    phi = Z[4:20].reshape(4, 4)

    x1, x2, lam1, lam2 = z[0:4]

    alpha_star = -1.0 / (2 * R) * 64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * lam2 + alpha0_rad
    flag = abs(alpha_star) < args.alpha_max_rad
    alpha_star = np.clip(alpha_star, -args.alpha_max_rad, args.alpha_max_rad)

    x1_dot = x2
    x2_dot = 64345.28 * np.exp(-x1/24000) * np.sqrt(1-(x2/15060)**2) * alpha_star - 20.685555 * (1 - (x2/15060)**2)
    lam1_dot = -2*q1*(x1-hd) + 2.6811 * lam2 * np.exp(-x1/24000) * np.sqrt(1-(x2/15060)**2) * alpha_star
    lam2_dot = -2*q2*x2 - lam1 - 1.8241*10**(-7)*lam2*x2 + 2.837*10**(-4) * lam2 * np.exp(-x1/24000) * x2 / np.sqrt(1 - (x2/15060)**2) * alpha_star

    z_dot = np.array([x1_dot, x2_dot, lam1_dot, lam2_dot])
    C1 = 64345.28
    C2 = 20.685555
    C3 = 2.6811
    C4 = 1.8241 * 10 ** (-7)
    C5 = 2.837 * 10 ** (-4)
    C6 = 64345.28

    F_array = np.zeros((4, 4), dtype=float)
    F_array[0, 1] = 1
    F_array[1, 0] = -(C1*alpha_star*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2))/24000
    F_array[1, 1] = (C2*x2)/113401800 - (C1*alpha_star*x2*np.exp(-x1/24000))/(226803600*(1 - x2**2/226803600)**(1/2))
    F_array[2, 0] = - 2*q1 - (C3*alpha_star*lam2*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2))/24000
    F_array[2, 1] = -(C3*alpha_star*lam2*x2*np.exp(-x1/24000))/(226803600*(1 - x2**2/226803600)**(1/2))
    F_array[2, 3] = C3*alpha_star*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2)
    F_array[3, 0] = -(C5*alpha_star*lam2*x2*np.exp(-x1/24000))/(24000*(1 - x2**2/226803600)**(1/2))
    F_array[3, 1] = (C5*alpha_star*lam2*np.exp(-x1/24000))/(1 - x2**2/226803600)**(1/2) - C4*lam2 - 2*q2 + (C5*alpha_star*lam2*x2**2*np.exp(-x1/24000))/(226803600*(1 - x2**2/226803600)**(3/2))
    F_array[3, 2] = -1
    F_array[3, 3] = (C5*alpha_star*x2*np.exp(-x1/24000))/(1 - x2**2/226803600)**(1/2) - C4*x2

    pzdotpu = np.zeros((4, 1), dtype=float)
    pupz = np.zeros((1, 4), dtype=float)

    pzdotpu[1, 0] = C1*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2)
    pzdotpu[2, 0] = C3*lam2*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2)
    pzdotpu[3, 0] = (C5*lam2*x2*np.exp(-x1/24000))/(1 - x2**2/226803600)**(1/2)

    pupz[0, 0] = (C6*lam2*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2))/(48000*R)
    pupz[0, 1] = (C6*lam2*x2*np.exp(-x1/24000))/(453607200*R*(1 - x2**2/226803600)**(1/2))
    pupz[0, 3] = -(C6*np.exp(-x1/24000)*(1 - x2**2/226803600)**(1/2))/(2*R)
    pupz = pupz * flag

    F_array = F_array + pzdotpu @ pupz
    phi_dot = F_array @ phi

    return np.hstack([z_dot, phi_dot.flatten()])


def reverse_integ(x1, x2, T):

    t = 0
    phi = np.eye(4)
    S = args.S_matrix
    lam1, lam2 = 2 * S @ np.array([x1-args.hd, x2])
    V = np.array([x1-args.hd, x2]).T @ S @ np.array([x1-args.hd, x2])

    alpha_star = -1.0 / (2 * R) * 64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * lam2 + alpha0_rad
    alpha_star = np.clip(alpha_star, -args.alpha_max_rad, args.alpha_max_rad)

    Z = np.hstack([x1, x2, lam1, lam2, phi.flatten()])
    DATA = [np.hstack([Z, alpha_star, V])]
    while t < T:
        dZdt = system(Z)

        x1, x2, lam1, lam2 = Z[0:4]
        delta_x = np.array([x1-args.hd, x2])
        alpha_star = -1.0 / (2 * R) * 64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * lam2 + alpha0_rad
        alpha_star = np.clip(alpha_star, -args.alpha_max_rad, args.alpha_max_rad)
        delta_u = alpha_star - alpha0_rad

        V = V + (delta_x.T@ Q @ delta_x + delta_u * R * delta_u) * dt
        dZdt = -dZdt
        Z = Z + dZdt * dt
        t = t + dt
        DATA.append(np.hstack([Z, alpha_star, V]))

    data_array = np.array(DATA)
    return data_array






x1 = 109980.09513730012
x2 = 12.833070470974931
T = 15

threshold = np.linalg.norm([x1-hd, x2])

Z_array = reverse_integ(x1, x2, T)
x_old0 = Z_array[-1, 0:2]
x_d = x_old0
r0 = np.linalg.norm(x_old0 - np.array([hd, 0]))
theta0 = np.arctan2(x_old0[1], x_old0[0]-hd)
x1_limit = np.abs(r0 * np.cos(theta0))
x2_limit = np.abs(r0 * np.sin(theta0))

ax1_0.scatter(x_d[0], x_d[1], c='k', marker='+', s=15, zorder=3)
# ax1_0.plot([args.hd+x1_limit, args.hd+x1_limit, args.hd-x1_limit, args.hd-x1_limit, args.hd+x1_limit],
#            [-x2_limit, x2_limit, x2_limit, -x2_limit, -x2_limit], 'k', linewidth=2)

delta_theta = 0.01
total_delta_theta = 0
count = -1  # 收集轨迹计数
traj_counter = 0
DATA = [Z_array]
for i in range(10000):
    # print(x1, x2)
    Z_array = reverse_integ(x1, x2, T)
    x_old = Z_array[-1, 0:2]

    PHI = Z_array[-1, 4:20].reshape(4, 4)
    LU = PHI[0:2, 0:2]
    RU = PHI[0:2, 2:4]
    STM = LU + 2 * RU @ S
    # print(STM)
    print("i:", i)
    if np.linalg.norm(x_old - x_d) < 5:
        count += 1

        hit_flag = True
        theta0 += delta_theta
        total_delta_theta += delta_theta
        print(theta0, total_delta_theta, count)
        if abs(total_delta_theta) > np.pi * 2:
            break
    else:
        hit_flag = False

    x1d = np.clip(r0*np.cos(theta0), -x1_limit, x1_limit) + hd
    x2d = np.clip(r0*np.sin(theta0), -x2_limit, x2_limit)
    x_d = np.array([x1d, x2d])
    delta_x = np.linalg.inv(STM) @ (x_d - x_old)
    x1 = x1 + delta_x[0]
    x2 = x2 + delta_x[1]

    # Adjust time and terminal state
    flag = np.linalg.norm([x1-hd, x2]) > threshold
    flag = flag * 2 -1
    while flag * np.linalg.norm([x1-hd, x2]) > flag * threshold:
        lam1, lam2 = 2 * S @ np.array([x1 - args.hd, x2])
        alpha_star = -1.0 / (2 * R) * 64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * lam2 + alpha0_rad
        alpha_star = np.clip(alpha_star, -args.alpha_max_rad, args.alpha_max_rad)

        x1_dot = x2
        x2_dot = 64345.28 * np.exp(-x1 / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * alpha_star - 20.685555 * (1 - (x2 / 15060) ** 2)

        x1 = x1 + x1_dot * dt * flag
        x2 = x2 + x2_dot * dt * flag
        T = T + dt * flag

    # LQR Part
    state_in = np.array([x1, x2])
    in_traj = []
    while np.linalg.norm(state_in-np.array([hd, 0])) > 0.1:
        V = (state_in-np.array([hd, 0])).T @ S @ (state_in-np.array([hd, 0]))
        lambda1, lambda2 = 2 * S @ (state_in-np.array([hd, 0]))

        alpha_star = -1.0 / (2 * R) * 64345.28 * np.exp(-state_in[0] / 24000) * np.sqrt(1 - (state_in[1] / 15060) ** 2) * lambda2 + alpha0_rad
        alpha_star = np.clip(alpha_star, -args.alpha_max_rad, args.alpha_max_rad)

        in_traj.append(np.hstack([state_in[0], state_in[1], lambda1, lambda2, np.zeros(16), alpha_star, V]))

        x1_dot = state_in[1]
        x2_dot = 64345.28 * np.exp(-state_in[0] / 24000) * np.sqrt(1 - (state_in[1] / 15060) ** 2) * alpha_star - 20.685555 * (1 - (state_in[1] / 15060) ** 2)

        state_in = state_in + np.array([x1_dot, x2_dot]) * dt

    if abs(x2d) < x2_limit:
        interval = 1
    else:
        interval =  10
    if count % interval == 0 and hit_flag:

        in_traj = np.array(in_traj)
        ax1_0.plot(in_traj[:, 0], in_traj[:, 1],
                   color=color_cycle[traj_counter % len(color_cycle)],
                   # linestyle=line_styles[traj_counter % len(line_styles)],
                   linewidth=1.5,
                   alpha=0.8,
                   label=f'Trajectory {traj_counter + 1}')

        ax1_0.scatter(x_d[0], x_d[1], c='k', marker='+', s=15, zorder=3)
        ax1_0.plot(Z_array[:, 0], Z_array[:, 1],
                   color=color_cycle[traj_counter % len(color_cycle)],
                   # linestyle=line_styles[traj_counter % len(line_styles)],
                   linewidth=1.5,
                   alpha=0.8,
                   label=f'Trajectory {traj_counter + 1}')
        plt.pause(0.01)

        DATA.append(Z_array)
        DATA.append(in_traj)
        traj_counter += 1

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
index = [0, 1, 2, 3, -2, -1]
dataset = dataset[:, index]
np.savez("../data/DATA.npz", dataset=dataset)

fig.savefig('image/optimal_data_WCC.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
