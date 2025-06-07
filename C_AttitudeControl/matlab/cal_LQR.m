clear; clc

syms phi theta psi p q r
syms Ixx Iyy Izz
syms tau_x tau_y tau_z
phi_dot = p + tan(theta) * sin(phi) * q + tan(theta) * cos(phi) * r;
theta_dot = cos(phi) * q + (-sin(phi) * r);
psi_dot = sin(phi)/cos(theta)*q + cos(phi)/cos(theta)*r;

p_dot = 1/Ixx * (tau_x + q*r*(Iyy-Izz));
q_dot = 1/Iyy * (tau_y + p*r*(Izz-Ixx));
r_dot = 1/Izz * (tau_z + p*q*(Ixx-Iyy));


Z_dot = [phi_dot; theta_dot; psi_dot; p_dot; q_dot; r_dot];
Z = [phi; theta; psi; p; q; r];
U = [tau_x; tau_y; tau_z];


A = jacobian(Z_dot, Z)
B = jacobian(Z_dot, U)

phi = 0;
theta = 0;
psi = 0;
p = 0;
q = 0;
r = 0;

A_eval = eval(A)
B_eval = eval(B)

