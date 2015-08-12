function X0 = drawStartState(param);
% Based on Jan Peters' codes. This  function generates a random initial
% state based on the original parameters of the task.

mu0 = param.param.mu0;
S0 = param.param.S0;
X0 = mvnrnd(mu0, S0, 1);