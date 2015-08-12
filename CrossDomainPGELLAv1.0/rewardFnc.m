function rew = rewardFnc(x, u)
% This function calculates the reward function. This simple reward
% function maximizes regulation accuracy and minimizes control cost.

rew = -2*sqrt(x'*x)-1e-2*sqrt(u'*u);