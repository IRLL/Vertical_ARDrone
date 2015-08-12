function u = drawAction(policy,x,param);
% Based on Jan Peters' code. This function calculates an action given the
% parameters of the current policy.

u = mvnrnd((reshape(policy.theta,param.param.N,param.param.M)'*x)', policy.sigma);