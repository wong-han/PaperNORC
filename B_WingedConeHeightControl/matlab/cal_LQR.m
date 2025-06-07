clear; clc
syms x1 x2 alpha

x1_dot = x2;
x2_dot = 64345.28 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * alpha - 20.685555 * (1 - (x2/15060)^2);

fxu = [x1_dot; x2_dot]
x = [x1; x2];
u = [alpha];

A = jacobian(fxu, x)
B = jacobian(fxu, u)

x1 = 110000;
x2 = 0;
alpha = 0.0314533;

A_eval = eval(A)
B_eval = eval(B)