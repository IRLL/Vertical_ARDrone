function [der,der2] = DlogPiDThetaREINFORCE(policy,x,u,param)

% Programmed by Jan Peters (jrpeters@usc.edu).
N = param.param.N;
M = param.param.M;

sigma = max(policy.sigma,0.00001);
k = policy.theta;
xx = x;
der = [];
for i=1:M
    der = [der;(u(i)-k(1+N*(i-1):N*(i-1)+N)'*xx)*xx/(sigma(i)^2)];
end
