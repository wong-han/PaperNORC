import numpy as np
from dataclasses import dataclass
import tyro
from scipy.linalg import solve_continuous_are

@dataclass
class Args:
    # ================ 模型相关 ============
    tau_max = 0.6
    Ixx = 0.0025
    Iyy = 0.0025
    Izz = 0.0035
    x_dim = 6
    z_dim = x_dim * 2


    # ================= 仿真相关 ===========
    dt: float = 0.001

    # ================ 控制相关 ============
    X0 = np.zeros(6)
    U0 = np.zeros(3)
    R = np.eye(3) * 10000  # 仿真时性能指标太小，Q和R同时乘以10000，不影响最优性结果
    Q = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])*0.1 * 10000

    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1 / Ixx, 0, 0],
        [0, 1 / Iyy, 0],
        [0, 0, 1 / Izz]
    ])
    S_matrix = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ S_matrix



if __name__ == '__main__':
    args = tyro.cli(Args)

    P = solve_continuous_are(args.A, args.B, args.Q, args.R)
    print("P", P)

    K = np.linalg.inv(args.R) @ args.B.T @ args.S_matrix
    print("K", K)
