'''
Generate Data
'''

import numpy as np
import tyro
from copy import deepcopy

from config import Args
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

z_dim = args.z_dim
x_dim = args.x_dim

def system(Z):

    z = Z[0:z_dim]
    PHI = Z[z_dim:z_dim+z_dim**2].reshape(z_dim, z_dim)

    phi, theta, psi, p, q, r, lam1, lam2, lam3, lam4, lam5, lam6 = z

    tau_x = -lam4 / (2.0*r1*Ixx)
    tau_y = -lam5 / (2.0*r2*Iyy)
    tau_z = -lam6 / (2.0*r3*Izz)
    flag1 = abs(tau_x) < args.tau_max
    flag2 = abs(tau_y) < args.tau_max
    flag3 = abs(tau_z) < args.tau_max
    tau_x = np.clip(tau_x, -args.tau_max, args.tau_max)
    tau_y = np.clip(tau_y, -args.tau_max, args.tau_max)
    tau_z = np.clip(tau_z, -args.tau_max, args.tau_max)

    phi_dot = p + np.tan(theta) * np.sin(phi) * q + np.tan(theta) * np.cos(phi) * r
    theta_dot = np.cos(phi) * q + (-np.sin(phi) * r)
    psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r
    p_dot = 1 / Ixx * (tau_x + q * r * (Iyy - Izz))
    q_dot = 1 / Iyy * (tau_y + p * r * (Izz - Ixx))
    r_dot = 1 / Izz * (tau_z + p * q * (Ixx - Iyy))

    lam1_dot = lam2*(r*np.cos(phi) + q*np.sin(phi)) - lam1*(q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta)) - 2*phi*q1 - lam3*((q*np.cos(phi))/np.cos(theta) - (r*np.sin(phi))/np.cos(theta))
    lam2_dot = - 2*q2*theta - lam3*((r*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 + (q*np.sin(phi)*np.sin(theta))/np.cos(theta)**2) - lam1*(r*np.cos(phi)*(np.tan(theta)**2 + 1) + q*np.sin(phi)*(np.tan(theta)**2 + 1))
    lam3_dot = -2*psi*q3
    lam4_dot = (lam5*r*(Ixx - Izz))/Iyy - 2*p*q4 - (lam6*q*(Ixx - Iyy))/Izz - lam1
    lam5_dot = - 2*q*q5 - lam2*np.cos(phi) - lam1*np.sin(phi)*np.tan(theta) - (lam3*np.sin(phi))/np.cos(theta) - (lam6*p*(Ixx - Iyy))/Izz - (lam4*r*(Iyy - Izz))/Ixx
    lam6_dot = lam2*np.sin(phi) - 2*q6*r - lam1*np.cos(phi)*np.tan(theta) - (lam3*np.cos(phi))/np.cos(theta) + (lam5*p*(Ixx - Izz))/Iyy - (lam4*q*(Iyy - Izz))/Ixx

    z_dot = np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, lam1_dot, lam2_dot, lam3_dot, lam4_dot, lam5_dot, lam6_dot])

    F_array = np.zeros((z_dim, z_dim))  # TO DO
    F_array[0, 0] = q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta)
    F_array[0, 1] = r*np.cos(phi)*(np.tan(theta)**2 + 1) + q*np.sin(phi)*(np.tan(theta)**2 + 1)
    F_array[0, 3] = 1
    F_array[0, 4] = np.sin(phi)*np.tan(theta)
    F_array[0, 5] = np.cos(phi)*np.tan(theta)
    F_array[1, 0] = - r*np.cos(phi) - q*np.sin(phi)
    F_array[1, 4] = np.cos(phi)
    F_array[1, 5] = -np.sin(phi)
    F_array[2, 0] = (q*np.cos(phi))/np.cos(theta) - (r*np.sin(phi))/np.cos(theta)
    F_array[2, 1] = (r*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 + (q*np.sin(phi)*np.sin(theta))/np.cos(theta)**2
    F_array[2, 4] = np.sin(phi)/np.cos(theta)
    F_array[2, 5] = np.cos(phi)/np.cos(theta)
    F_array[3, 4] = (r*(Iyy - Izz))/Ixx
    F_array[3, 5] = (q*(Iyy - Izz))/Ixx
    F_array[4, 3] = -(r*(Ixx - Izz))/Iyy
    F_array[4, 5] = -(p*(Ixx - Izz))/Iyy
    F_array[5, 3] = (q*(Ixx - Iyy))/Izz
    F_array[5, 4] = (p*(Ixx - Iyy))/Izz
    F_array[6, 0] = lam1*(r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta)) - 2*q1 + lam2*(q*np.cos(phi) - r*np.sin(phi)) + lam3*((r*np.cos(phi))/np.cos(theta) + (q*np.sin(phi))/np.cos(theta))
    F_array[6, 1] = - lam3*((q*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 - (r*np.sin(phi)*np.sin(theta))/np.cos(theta)**2) - lam1*(q*np.cos(phi)*(np.tan(theta)**2 + 1) - r*np.sin(phi)*(np.tan(theta)**2 + 1))
    F_array[6, 4] = lam2*np.sin(phi) - lam1*np.cos(phi)*np.tan(theta) - (lam3*np.cos(phi))/np.cos(theta)
    F_array[6, 5] = lam2*np.cos(phi) + lam1*np.sin(phi)*np.tan(theta) + (lam3*np.sin(phi))/np.cos(theta)
    F_array[6, 6] = r*np.sin(phi)*np.tan(theta) - q*np.cos(phi)*np.tan(theta)
    F_array[6, 7] = r*np.cos(phi) + q*np.sin(phi)
    F_array[6, 8] = (r*np.sin(phi))/np.cos(theta) - (q*np.cos(phi))/np.cos(theta)
    F_array[7, 0] = - lam3*((q*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 - (r*np.sin(phi)*np.sin(theta))/np.cos(theta)**2) - lam1*(q*np.cos(phi)*(np.tan(theta)**2 + 1) - r*np.sin(phi)*(np.tan(theta)**2 + 1))
    F_array[7, 1] = - 2*q2 - lam3*((r*np.cos(phi))/np.cos(theta) + (q*np.sin(phi))/np.cos(theta) + (2*r*np.cos(phi)*np.sin(theta)**2)/np.cos(theta)**3 + (2*q*np.sin(phi)*np.sin(theta)**2)/np.cos(theta)**3) - lam1*(2*r*np.cos(phi)*np.tan(theta)*(np.tan(theta)**2 + 1) + 2*q*np.sin(phi)*np.tan(theta)*(np.tan(theta)**2 + 1))
    F_array[7, 4] = - lam1*np.sin(phi)*(np.tan(theta)**2 + 1) - (lam3*np.sin(phi)*np.sin(theta))/np.cos(theta)**2
    F_array[7, 5] = - lam1*np.cos(phi)*(np.tan(theta)**2 + 1) - (lam3*np.cos(phi)*np.sin(theta))/np.cos(theta)**2
    F_array[7, 6] = - r*np.cos(phi)*(np.tan(theta)**2 + 1) - q*np.sin(phi)*(np.tan(theta)**2 + 1)
    F_array[7, 8] = - (r*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 - (q*np.sin(phi)*np.sin(theta))/np.cos(theta)**2
    F_array[8, 2] = -2*q3
    F_array[9, 3] = -2*q4
    F_array[9, 4] = -(lam6*(Ixx - Iyy))/Izz
    F_array[9, 5] = (lam5*(Ixx - Izz))/Iyy
    F_array[9, 6] = -1
    F_array[9, 10] = (r*(Ixx - Izz))/Iyy
    F_array[9, 11] = -(q*(Ixx - Iyy))/Izz
    F_array[10, 0] = lam2*np.sin(phi) - lam1*np.cos(phi)*np.tan(theta) - (lam3*np.cos(phi))/np.cos(theta)
    F_array[10, 1] = - lam1*np.sin(phi)*(np.tan(theta)**2 + 1) - (lam3*np.sin(phi)*np.sin(theta))/np.cos(theta)**2
    F_array[10, 3] = -(lam6*(Ixx - Iyy))/Izz
    F_array[10, 4] = -2*q5
    F_array[10, 5] = -(lam4*(Iyy - Izz))/Ixx
    F_array[10, 6] = -np.sin(phi)*np.tan(theta)
    F_array[10, 7] = -np.cos(phi)
    F_array[10, 8] = -np.sin(phi)/np.cos(theta)
    F_array[10, 9] = -(r*(Iyy - Izz))/Ixx
    F_array[10, 11] = -(p*(Ixx - Iyy))/Izz
    F_array[11, 0] = lam2*np.cos(phi) + lam1*np.sin(phi)*np.tan(theta) + (lam3*np.sin(phi))/np.cos(theta)
    F_array[11, 1] = - lam1*np.cos(phi)*(np.tan(theta)**2 + 1) - (lam3*np.cos(phi)*np.sin(theta))/np.cos(theta)**2
    F_array[11, 3] = (lam5*(Ixx - Izz))/Iyy
    F_array[11, 4] = -(lam4*(Iyy - Izz))/Ixx
    F_array[11, 5] = -2*q6
    F_array[11, 6] = -np.cos(phi)*np.tan(theta)
    F_array[11, 7] = np.sin(phi)
    F_array[11, 8] = -np.cos(phi)/np.cos(theta)
    F_array[11, 9] = -(q*(Iyy - Izz))/Ixx
    F_array[11, 10] = (p*(Ixx - Izz))/Iyy

    pzdotpu = np.zeros((12, 3), dtype=np.float64)
    pzdotpu[3, 0] = 1.0 / Ixx
    pzdotpu[4, 1] = 1.0 / Iyy
    pzdotpu[5, 2] = 1.0 / Izz

    pupz = np.zeros((3, 12), dtype=np.float64)
    pupz[0, 9] = -1.0/(2*Ixx*r1)
    pupz[1, 10] = -1.0/(2*Iyy*r2)
    pupz[2, 11] = -1.0/(2*Izz*r3)

    pupz[0, 9] = pupz[0, 9] * flag1
    pupz[1, 10] = pupz[1, 10] * flag2
    pupz[2, 11] = pupz[2, 11] * flag3

    F_array = F_array + pzdotpu @ pupz

    PHI_dot = F_array @ PHI

    return np.hstack((z_dot, PHI_dot.flatten()))


def reverse_integ(state, T):

    t = 0
    PHI = np.eye(z_dim)

    Lambda = 2 * S @ state
    Z = np.hstack([state, Lambda, PHI.flatten()])

    lam1, lam2, lam3, lam4, lam5, lam6 = Lambda
    tau_x = -lam4 / (2.0 * r1 * Ixx)
    tau_y = -lam5 / (2.0 * r2 * Iyy)
    tau_z = -lam6 / (2.0 * r3 * Izz)
    tau_x = np.clip(tau_x, -args.tau_max, args.tau_max)
    tau_y = np.clip(tau_y, -args.tau_max, args.tau_max)
    tau_z = np.clip(tau_z, -args.tau_max, args.tau_max)
    action = np.array([tau_x, tau_y, tau_z])

    V = state.T @ S @ state
    DATA = [np.hstack([Z, action, V])]

    while t < T:
        dZdt = system(Z)

        state = Z[0:6]
        Lambda = Z[6:12]
        lam1, lam2, lam3, lam4, lam5, lam6 = Lambda
        tau_x = -lam4 / (2.0 * r1 * Ixx)
        tau_y = -lam5 / (2.0 * r2 * Iyy)
        tau_z = -lam6 / (2.0 * r3 * Izz)
        tau_x = np.clip(tau_x, -args.tau_max, args.tau_max)
        tau_y = np.clip(tau_y, -args.tau_max, args.tau_max)
        tau_z = np.clip(tau_z, -args.tau_max, args.tau_max)
        action = np.array([tau_x, tau_y, tau_z])

        V = V + (state.T @ Q @ state + action.T @ R @ action) * dt
        dZdt = -dZdt
        Z = Z + dZdt * dt
        t = t + dt
        DATA.append(np.hstack([Z, action, V]))

    data_array = np.array(DATA)
    return data_array

fig = plt.figure(figsize=[3.5, 3.5])
ax1_0 = fig.add_subplot(111)
# 设置网格样式
ax1_0.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

# 定义颜色循环和线型（用于区分不同轨迹）
color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
line_styles = ['-', '--', '-.', ':']  # 不同线型

# 设置坐标轴标签和标题
ax1_0.set_xlabel(r'$\phi$(rad)', usetex=True)
ax1_0.set_ylabel(r'$\dot \phi$(rad/s)', usetex=True)

# state = np.random.uniform(-0.1, 0.1, 6)
# state = np.array([ 0.39819346,  0.10714945, -0.2563235,  -0.40016018, -0.06973792,  0.24682883])
state = np.array([ 0.00430169, -0.00491039, -0.00514428, -0.00911669,  0.00990358,  0.00828207])
T = 0.6*2.8
Z_array = reverse_integ(state, T)
x_d = Z_array[-1, 0:x_dim]
phi_d0 = abs(x_d[0])
phi_dot_d0 = abs(x_d[3])
r0 = np.linalg.norm([x_d[0], x_d[3]])
degree0 = np.arctan2(x_d[3], x_d[0])

delta_theta = 0.01
total_delta_theta = 0

DATA = []
count = -1
traj_counter = 0
interval = 10
while abs(total_delta_theta) <= np.pi * 2:

    Z_array = reverse_integ(state, T)
    x_old = Z_array[-1, 0:x_dim]

    PHI = Z_array[-1, z_dim:z_dim+z_dim**2].reshape(z_dim, z_dim)
    LU = PHI[0:x_dim, 0:x_dim]
    RU = PHI[0:x_dim, x_dim:x_dim+x_dim]
    STM = LU + 2 * RU @ S
    # print(STM)

    if np.linalg.norm(x_old-x_d) < 0.01:
        hit_flag = True
        count += 1
        if count % interval == 0:
            ax1_0.scatter(x_d[0], x_d[3], c='k', marker='+', s=15, zorder=3)
        degree0 += delta_theta
        total_delta_theta += delta_theta
        phi_d = r0 * np.cos(degree0)
        phi_dot_d = r0 * np.sin(degree0)
        phi_d = np.clip(r0 * np.cos(degree0), -phi_d0, phi_d0)
        phi_dot_d = np.clip(r0 * np.sin(degree0), -phi_dot_d0, phi_dot_d0)
        x_d[0] = phi_d
        x_d[3] = phi_dot_d
        if abs(phi_d) < phi_d0:
            interval = 5
        else:
            interval = 10
    else:
        hit_flag = False

    if np.linalg.norm(state) > 2:
        refine_traj = [state]
        for _ in range(1):
            phi, theta, psi, p, q, r = state
            Lambda = 2 * S @ state
            lam1, lam2, lam3, lam4, lam5, lam6 = Lambda
            tau_x = -lam4 / (2.0 * r1 * Ixx)
            tau_y = -lam5 / (2.0 * r2 * Iyy)
            tau_z = -lam6 / (2.0 * r3 * Izz)
            tau_x = np.clip(tau_x, -args.tau_max, args.tau_max)
            tau_y = np.clip(tau_y, -args.tau_max, args.tau_max)
            tau_z = np.clip(tau_z, -args.tau_max, args.tau_max)

            phi_dot = p + np.tan(theta) * np.sin(phi) * q + np.tan(theta) * np.cos(phi) * r
            theta_dot = np.cos(phi) * q + (-np.sin(phi) * r)
            psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r
            p_dot = 1 / Ixx * (tau_x + q * r * (Iyy - Izz))
            q_dot = 1 / Iyy * (tau_y + p * r * (Izz - Ixx))
            r_dot = 1 / Izz * (tau_z + p * q * (Ixx - Iyy))

            lam1_dot = lam2*(r*np.cos(phi) + q*np.sin(phi)) - lam1*(q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta)) - 2*phi*q1 - lam3*((q*np.cos(phi))/np.cos(theta) - (r*np.sin(phi))/np.cos(theta))
            lam2_dot = - 2*q2*theta - lam3*((r*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 + (q*np.sin(phi)*np.sin(theta))/np.cos(theta)**2) - lam1*(r*np.cos(phi)*(np.tan(theta)**2 + 1) + q*np.sin(phi)*(np.tan(theta)**2 + 1))
            lam3_dot = -2*psi*q3
            lam4_dot = (lam5*r*(Ixx - Izz))/Iyy - 2*p*q4 - (lam6*q*(Ixx - Iyy))/Izz - lam1
            lam5_dot = - 2*q*q5 - lam2*np.cos(phi) - lam1*np.sin(phi)*np.tan(theta) - (lam3*np.sin(phi))/np.cos(theta) - (lam6*p*(Ixx - Iyy))/Izz - (lam4*r*(Iyy - Izz))/Ixx
            lam6_dot = lam2*np.sin(phi) - 2*q6*r - lam1*np.cos(phi)*np.tan(theta) - (lam3*np.cos(phi))/np.cos(theta) + (lam5*p*(Ixx - Izz))/Iyy - (lam4*q*(Iyy - Izz))/Ixx

            state_dot = np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
            lambda_dot = np.array([lam1_dot, lam2_dot, lam3_dot, lam4_dot, lam5_dot, lam6_dot])
            state = state + state_dot * dt
            Lambda = Lambda + lambda_dot * dt
            T = T + dt
            refine_traj.append(state)
        refine_array = np.array(refine_traj)
        print("refining:", state, T)
        ax1_0.plot(refine_array[:, 0], refine_array[:, 3], c='k')


    if count % interval == 0 and hit_flag:
        state_in = state
        in_traj = []
        while np.linalg.norm(state_in) > 0.001:
            phi, theta, psi, p, q, r = state_in
            Lambda = 2 * S @ state_in
            lam1, lam2, lam3, lam4, lam5, lam6 = Lambda
            tau_x = -lam4 / (2.0 * r1 * Ixx)
            tau_y = -lam5 / (2.0 * r2 * Iyy)
            tau_z = -lam6 / (2.0 * r3 * Izz)
            tau_x = np.clip(tau_x, -args.tau_max, args.tau_max)
            tau_y = np.clip(tau_y, -args.tau_max, args.tau_max)
            tau_z = np.clip(tau_z, -args.tau_max, args.tau_max)

            phi_dot = p + np.tan(theta) * np.sin(phi) * q + np.tan(theta) * np.cos(phi) * r
            theta_dot = np.cos(phi) * q + (-np.sin(phi) * r)
            psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r
            p_dot = 1 / Ixx * (tau_x + q * r * (Iyy - Izz))
            q_dot = 1 / Iyy * (tau_y + p * r * (Izz - Ixx))
            r_dot = 1 / Izz * (tau_z + p * q * (Ixx - Iyy))

            lam1_dot = lam2 * (r * np.cos(phi) + q * np.sin(phi)) - lam1 * (q * np.cos(phi) * np.tan(theta) - r * np.sin(phi) * np.tan(theta)) - 2 * phi * q1 - lam3 * (
                    (q * np.cos(phi)) / np.cos(theta) - (r * np.sin(phi)) / np.cos(theta))
            lam2_dot = - 2 * q2 * theta - lam3 * ((r * np.cos(phi) * np.sin(theta)) / np.cos(theta) ** 2 + (q * np.sin(phi) * np.sin(theta)) / np.cos(theta) ** 2) - lam1 * (
                    r * np.cos(phi) * (np.tan(theta) ** 2 + 1) + q * np.sin(phi) * (np.tan(theta) ** 2 + 1))
            lam3_dot = -2 * psi * q3
            lam4_dot = (lam5 * r * (Ixx - Izz)) / Iyy - 2 * p * q4 - (lam6 * q * (Ixx - Iyy)) / Izz - lam1
            lam5_dot = - 2 * q * q5 - lam2 * np.cos(phi) - lam1 * np.sin(phi) * np.tan(theta) - (lam3 * np.sin(phi)) / np.cos(theta) - (lam6 * p * (Ixx - Iyy)) / Izz - (lam4 * r * (Iyy - Izz)) / Ixx
            lam6_dot = lam2 * np.sin(phi) - 2 * q6 * r - lam1 * np.cos(phi) * np.tan(theta) - (lam3 * np.cos(phi)) / np.cos(theta) + (lam5 * p * (Ixx - Izz)) / Iyy - (lam4 * q * (Iyy - Izz)) / Ixx

            state_dot = np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
            lambda_dot = np.array([lam1_dot, lam2_dot, lam3_dot, lam4_dot, lam5_dot, lam6_dot])
            state_in = state_in + state_dot * dt
            Lambda = Lambda + lambda_dot * dt

            action = np.array([tau_x, tau_y, tau_z])
            V = state_in.T @ S @ state_in
            in_traj.append(np.hstack([state_in, np.zeros(6), np.zeros(144), action, V]))


        ax1_0.plot(Z_array[:, 0], Z_array[:, 3],
                   color=color_cycle[traj_counter % len(color_cycle)],
                   # linestyle=line_styles[traj_counter % len(line_styles)],
                   linewidth=1.5,
                   alpha=0.8,
                   label=f'Trajectory {traj_counter + 1}')
        in_traj = np.array(in_traj)
        ax1_0.plot(in_traj[:, 0], in_traj[:, 3],
                   color=color_cycle[traj_counter % len(color_cycle)],
                   # linestyle=line_styles[traj_counter % len(line_styles)],
                   linewidth=1.5,
                   alpha=0.8,
                   label=f'Trajectory {traj_counter + 1}')

        DATA.append(Z_array)
        DATA.append(in_traj)
        traj_counter += 1



    delta_state = np.linalg.inv(STM) @ (x_d - x_old)
    state = state + delta_state
    print(state, T)

    plt.pause(0.01)


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
index = [0, 1, 2, 3, 4, 5, -4, -3, -2, -1]
dataset = dataset[:, index]
np.savez("../data/DATA.npz", dataset=dataset)

fig.savefig('image/optimal_data_ATT.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()



