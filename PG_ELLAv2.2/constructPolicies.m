function [PGPol]=constructPolicies(Tasks)

for i=1:size(Tasks,2)
    N = Tasks(i).param.N; % Number of states
    M = Tasks(i).param.M; % Number of inputs
    
    poliType = Tasks(i).param.poliType; % Structure of policy
    
    if strcmpi(poliType,'Gauss')
        policy.theta = rand(N*M,1);
        policy.sigma = rand(1,M);
    else
        error('Undefined Policy Type')
        break;
    end

    PGPol(i).policy = policy;
end