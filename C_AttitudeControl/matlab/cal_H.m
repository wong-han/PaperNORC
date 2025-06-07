clear; clc

syms phi theta psi p q r
syms Ixx Iyy Izz
syms tau_x tau_y tau_z
syms lam1 lam2 lam3 lam4 lam5 lam6
syms q1 q2 q3 q4 q5 q6
syms r1 r2 r3

phi_dot = p + tan(theta) * sin(phi) * q + tan(theta) * cos(phi) * r;
theta_dot = cos(phi) * q + (-sin(phi) * r);
psi_dot = sin(phi)/cos(theta)*q + cos(phi)/cos(theta)*r;

p_dot = 1/Ixx * (tau_x + q*r*(Iyy-Izz));
q_dot = 1/Iyy * (tau_y + p*r*(Izz-Ixx));
r_dot = 1/Izz * (tau_z + p*q*(Ixx-Iyy));

Q = diag([q1 q2 q3 q4 q5 q6]);
R = diag([r1, r2, r3]);

X = [phi; theta; psi; p; q; r];
X_dot = [phi_dot; theta_dot; psi_dot; p_dot; q_dot; r_dot];
Lambda = [lam1; lam2; lam3; lam4; lam5; lam6];
U = [tau_x; tau_y; tau_z];

H = X.' * Q * X + U.' * R * U + Lambda.' * X_dot;

Lambda_dot = - jacobian(H, X).'
pHpU = jacobian(H, U).'