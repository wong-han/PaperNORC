clear; clc

syms x1 x2 hd
syms alpha alpha0
syms lam1 lam2
syms q1 q2
syms R
syms C1 C2 C3 C4 C5 C6

% C1 = 64345.28;
% C2 = 20.685555;
% C3 = 2.6811;
% C4 = 1.8241*10^(-7);
% C5 = 2.837*10^(-4);
% C6 = 64345.28;

X = [x1; x2];
Lambda = [lam1; lam2];
U = [alpha];


x1_dot = x2;
x2_dot = C1 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * alpha - C2 * (1 - (x2/15060)^2);
X_dot = [x1_dot; x2_dot];

lam1_dot = -2*q1*(x1-hd) + C3 * lam2 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * alpha;
lam2_dot = -2*q2*x2 - lam1 - C4*lam2*x2 + C5 * lam2 * exp(-x1/24000) * x2 / sqrt(1 - (x2/15060)^2) * alpha;
Lambda_dot = [lam1_dot; lam2_dot];


z = [X;Lambda];
z_dot = [X_dot; Lambda_dot];

phpz = jacobian(z_dot, z)
for i = 1:4
    for j = 1:4
        if eval(phpz(i,j)) == 0
            continue
        end
        disp([i-1, j-1,  eval(phpz(i, j))])
    end
end

phpu = jacobian(z_dot, U)


Uz = -1.0/(2*R) * C6 * exp(-x1/24000) * sqrt(1-(x2/15060)^2) * lam2 + alpha0;

pUzpz = jacobian(Uz, z)
