import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from B_LNN_Learning_PaperPlot import ValueNet, ActionNet

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

action_net = ActionNet()
action_net.load_checkpoint()
value_net = ValueNet()
value_net.load_checkpoint()

def NetController(x):
    x1, x2 = x
    x = torch.Tensor([x]).requires_grad_(True)

    value = value_net(x)
    dVdx = torch.autograd.grad(value, x, torch.ones_like(value))
    dVdx = dVdx[0].detach().numpy()

    fx = np.array([
        [-x1 + x2],
        [-0.5*x1 - 0.5*x2*(1 - (np.cos(2*x1)+2)**2)]
    ])
    gx = np.array([
        [0],
        [(np.cos(2*x1)+2)]
    ])

    a_net = action_net(x)
    a_net = a_net.detach().numpy()[0][0]

    Ldot = dVdx @ (fx + gx * a_net)
    if Ldot >= 0:
        delta_u = (-np.abs((dVdx @ gx)) * np.linalg.norm([x1, x2]) - Ldot) / (dVdx @ gx)  # \Delta u=\frac{-k\lVert x \rVert _2-\frac{\text{d}V}{\text{d}x}\left( f\left( x \right) +g\left( x \right) a_{net} \right)}{\frac{\text{d}V}{\text{d}x}g\left( x \right)}
        delta_u = delta_u[0][0]
        print("delta_u:", delta_u)
    else:
        delta_u = 0

    return a_net + delta_u


fig1_1 = plt.figure(figsize=(3.5, 3.5))
ax1_1 = fig1_1.add_subplot(211)
ax1_1.set_xlabel(r'$t(s)$', usetex=True)
ax1_1.set_ylabel(r'$x_1$', usetex=True)
ax1_1.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

fig1_2 = plt.figure(figsize=(3.5, 3.5))
ax1_2 = fig1_2.add_subplot(211)
ax1_2.set_xlabel(r'$t(s)$', usetex=True)
ax1_2.set_ylabel(r'$x_2$', usetex=True)
ax1_2.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

fig1_3 = plt.figure(figsize=(3.5, 3.5))
ax1_3 = fig1_3.add_subplot(211)
ax1_3.set_xlabel(r'$t(s)$', usetex=True)
ax1_3.set_ylabel(r'$u$', usetex=True)
ax1_3.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

# 定义颜色循环和线型（用于区分不同轨迹）
color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
line_styles = ['-', '--', '-.', ':']  # 不同线型


def Plot(x1, x2, controller):
    # x1 = -0.5
    # x2 = 0.5

    V = 0
    t = 0
    dt = 0.005

    data = []
    while (t < 6):
        u = controller(np.array([x1, x2]))

        dx1dt = -x1 + x2
        dx2dt = -0.5*x1 - 0.5*x2*(1 - (np.cos(2*x1)+2)**2) + (np.cos(2*x1)+2)*u

        x1 = x1 + dx1dt * dt
        x2 = x2 + dx2dt * dt
        t = t + dt
        V = V + (x1**2+x2**2+u**2)*dt

        data.append([t, x1, x2, u, V])

        # if abs(x1) < 0.5 and abs(x2) < 0.5:
        #     break

    # print(V)
    data = np.array(data)
    return data

r0 = 4
for count, theta in enumerate(np.linspace(0, 2*np.pi, 20)):
    print(theta)
    x1 = r0 * np.cos(theta)
    x2 = r0 * np.sin(theta)
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
    ax1_3.plot(data[:, 0], data[:, 3],
               color=color_cycle[count % len(color_cycle)],
               # linestyle=line_styles[traj_counter % len(line_styles)],
               linewidth=1.5,
               alpha=1,
               label=f'Trajectory {count + 1}')

fig1_1.savefig('image/x1_SOP.pdf', bbox_inches='tight', pad_inches=0.1)
fig1_2.savefig('image/x2_SOP.pdf', bbox_inches='tight', pad_inches=0.1)
fig1_3.savefig('image/u_SOP.pdf', bbox_inches='tight', pad_inches=0.1)
# plt.legend()
plt.show()
