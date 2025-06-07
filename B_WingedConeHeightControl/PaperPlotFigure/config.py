import numpy as np
from dataclasses import dataclass
import tyro
from scipy.linalg import solve_continuous_are

@dataclass
class Args:
    # ================ 模型相关 ============
    mass: float = 9375  # 飞行器质量 slugs
    Re: float = 20903500 # 地球半径 ft
    S: float = 3603 # 参考面积 ft^2
    mu: float = 1.39 * 10 ** 16 # 地球引力常数
    hd: float = 110000 # 标称高度
    vd: float = 15060 # 标称速度

    def cal_CL_a(self, h):
        # CL对alpha的系数
        a = self.cal_a(h)
        M = vd / a
        CL = 0.493+1.91 / M
        return CL

    def cal_a(self, h):
        a = 8.99 * 10 ** (-9) * h ** 2 - 9.16 * 10 ** (-4) * h + 996
        return a

    # ================= 仿真相关 ===========
    dt: float = 0.001

    # ================ 控制相关 ============
    alpha0_rad: float = 0.0314533
    alpha_max_rad: float = np.deg2rad(5)
    R = 1000
    Q = np.eye(2) * 0.0001
    A = np.array([[0, 1],
                  [-0.0009, 0]])
    B = np.array([[0], [657.6583]])
    S_matrix = solve_continuous_are(A, B, Q, R)
    K = 1 / R * B.T @ S_matrix



if __name__ == '__main__':
    args = tyro.cli(Args)

    mass = args.mass
    mu = args.mu
    Re = args.Re
    hd = args.hd
    vd = args.vd
    S = args.S

    print("第一项 C1 * exp(-h/24000) * alpha * cos(gamma) 的系数:", 0.5 * 0.00238 * vd**2 * S * args.cal_CL_a(hd) / mass)
    print("第二项 cos(gamma)**2 的系数：", (mu - vd**2*(hd+Re))/((hd+Re)**2))

    P = solve_continuous_are(args.A, args.B, args.Q, args.R)
    print("P", P)

    K = 1 / args.R * args.B.T @ args.S_matrix
    print("K", K)
