clear; clc

syms phi theta psi p q r
syms Ixx Iyy Izz
syms tau_x tau_y tau_z
syms lam1 lam2 lam3 lam4 lam5 lam6
syms q1 q2 q3 q4 q5 q6
syms r1 r2 r3

X = [phi; theta; psi; p; q; r];
Lambda = [lam1; lam2; lam3; lam4; lam5; lam6];
U = [tau_x; tau_y; tau_z];

phi_dot = p + tan(theta) * sin(phi) * q + tan(theta) * cos(phi) * r;
theta_dot = cos(phi) * q + (-sin(phi) * r);
psi_dot = sin(phi)/cos(theta)*q + cos(phi)/cos(theta)*r;
p_dot = 1/Ixx * (tau_x + q*r*(Iyy-Izz));
q_dot = 1/Iyy * (tau_y + p*r*(Izz-Ixx));
r_dot = 1/Izz * (tau_z + p*q*(Ixx-Iyy));
X_dot = [phi_dot; theta_dot; psi_dot; p_dot; q_dot; r_dot];

lam1_dot = lam2*(r*cos(phi) + q*sin(phi)) - lam1*(q*cos(phi)*tan(theta) - r*sin(phi)*tan(theta)) - 2*phi*q1 - lam3*((q*cos(phi))/cos(theta) - (r*sin(phi))/cos(theta));
lam2_dot = - 2*q2*theta - lam3*((r*cos(phi)*sin(theta))/cos(theta)^2 + (q*sin(phi)*sin(theta))/cos(theta)^2) - lam1*(r*cos(phi)*(tan(theta)^2 + 1) + q*sin(phi)*(tan(theta)^2 + 1));
lam3_dot = -2*psi*q3;
lam4_dot = (lam5*r*(Ixx - Izz))/Iyy - 2*p*q4 - (lam6*q*(Ixx - Iyy))/Izz - lam1;
lam5_dot = - 2*q*q5 - lam2*cos(phi) - lam1*sin(phi)*tan(theta) - (lam3*sin(phi))/cos(theta) - (lam6*p*(Ixx - Iyy))/Izz - (lam4*r*(Iyy - Izz))/Ixx;
lam6_dot = lam2*sin(phi) - 2*q6*r - lam1*cos(phi)*tan(theta) - (lam3*cos(phi))/cos(theta) + (lam5*p*(Ixx - Izz))/Iyy - (lam4*q*(Iyy - Izz))/Ixx;
Lambda_dot = [lam1_dot; lam2_dot; lam3_dot; lam4_dot; lam5_dot; lam6_dot];


z = [X;Lambda];
z_dot = [X_dot; Lambda_dot];

phpz = jacobian(z_dot, z)
for i = 1:12
    for j = 1:12
        if eval(phpz(i,j)) == 0
            continue
        end
        disp([i-1, j-1,  eval(phpz(i, j))])
    end
end

phpu = jacobian(z_dot, U)

Uz = [
-lam4 / (2.0*r1*Ixx);
-lam5 / (2.0*r2*Iyy);
-lam6 / (2.0*r3*Izz);
    ];

pUzpz = jacobian(Uz, z)
