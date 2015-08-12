function dJdtheta = episodicREINFORCE(policy, data,param)
% Based on Jan Peter's code. This code carries out the calculation of the
% gradient of the objective function J with respect to the parameter vector 
% \theta

% Importing number of states and inputs of the current task
N = param.param.N;
M = param.param.M;

% Importing the discount factor \gamma
gamma = param.param.gamma;
   
% Initializing the gradient of the logarithm of the policy
dJdtheta = zeros(size(DlogPiDThetaREINFORCE(policy,ones(N,1),ones(M,1),param)));

% Estimating the gradient based on the generated data from the task.
for Trials = 1 : max(size(data))
    dSumPi = zeros(size(DlogPiDThetaREINFORCE(policy,ones(N,1),ones(M,1),param)));
    sumR   = 0;   
    for Steps = 1 : max(size(data(Trials).u))
        dSumPi = dSumPi + DlogPiDThetaREINFORCE(policy, data(Trials).x(:,Steps), data(Trials).u(:,Steps),param);
        sumR   = sumR   + gamma^(Steps-1)*data(Trials).r(Steps);
    end
    dJdtheta = dJdtheta + dSumPi * sumR;
end
     
dJdtheta = (1-gamma)*dJdtheta / max(size(data));