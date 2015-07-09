function X0 = drawStartState(param);
% Based on Jan Peters' codes 

N = param.param.N;

mu0 = param.param.mu0;
S0 = param.param.S0;
X0 = mvnrnd(mu0, S0, 1);