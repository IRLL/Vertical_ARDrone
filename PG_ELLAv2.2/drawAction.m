function u = drawAction(policy,x,param);
% Based on Jan Peters' code.

u = mvnrnd((reshape(policy.theta,param.param.N,param.param.M)'*x)', policy.sigma);