clear; clc

syms x1 x2 hd
syms alpha alpha0
syms lam1 lam2
syms q1 q2
syms R

x1_dot = x2;
x2_dot = 64345.28 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * alpha - 20.685555 * (1 - (x2/15060)^2);
X_dot = [x1_dot; x2_dot];

Q = diag([q1 q2]);

X = [x1; x2];
Lambda = [lam1; lam2];
U = [alpha];

Delta_X = [x1-hd; x2]
Delta_U = [alpha-alpha0]


H = Delta_X.' * Q * Delta_X + Delta_U.' * R * Delta_U + Lambda.' * X_dot;

Lambda_dot = - jacobian(H, X).'
pHpU = jacobian(H, U).'

% lam1_dot = -2*q1*(x1-hd) + 2.6811 * lam2 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * alpha;
% lam2_dot = -2*q2*x2 - lam1 - 1.8241*10^(-7)*lam2*x2 + 2.837*10^(-4) * lam2 * exp(-x1/24000) * x2 / sqrt(1 - (x2/15060)^2) * alpha;
% alpha = -1.0/(2*R) * 64345.28 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * lam2 + alpha0;