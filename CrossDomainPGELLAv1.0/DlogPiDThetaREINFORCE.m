function [der,der2] = DlogPiDThetaREINFORCE(policy,x,u,param)
% Based on Jan Peters code. This code estimates the gradient of the policy
% by using episodic REINFORCE

%Importing the number of states and inputs of the current task.
N = param.param.N;
M = param.param.M;

% Importing policy parameters
sigma = max(policy.sigma,0.00001);
k = policy.theta;

der = [];
for i=1:M
    der = [der;(u(i)-k(1+N*(i-1):N*(i-1)+N)'*x)*x/(sigma(i)^2)];
end
