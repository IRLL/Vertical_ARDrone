function [Tasks]=createSys(nSystems,poliType,baseLearner,gamma)
%% Fixed Parameters
Tasks.nSystems = nSystems; % Number of tasks

% Range of parameters for dynamical systems
Tasks.MassMin = 3;
Tasks.MassMax = 5; 
Tasks.Mink = 1; 
Tasks.Maxk = 7; 
Tasks.Maxd = 0.1;
Tasks.Mind = 0.01; 

counter = 1;
for i=1:nSystems
    
    param.N = 2; % Number of states
    param.M = 1; % Number of inputs

    % Differential for integration of dynamical system
    param.dt = .01;

    % Calculation of random parameters
    param.Mass = (Tasks(1).MassMax - Tasks(1).MassMin)*rand() + Tasks(1).MassMin;
    param.k = (Tasks(1).Maxk - Tasks(1).Mink)*rand() + Tasks(1).Mink;
    param.d = (Tasks(1).Maxd - Tasks(1).Mind)*rand() + Tasks(1).Mind;
    
    % Initial and reference (final) states
    param.mu0 = 2*rand(2,1);
    param.Xref = zeros(2,1);
    
    % Parameters for policy
    param.poliType = poliType;
    param.baseLearner = baseLearner;
    param.gamma = gamma;
    
    % Covariance matrix for Gaussian Policy
    param.S0 = 0.0001*eye(param.N); % Covariance matrix for Gaussian Policy
        
    % Assignation of parameters to task i
    param.TaskId = i;
    Tasks(i).param = param; 
    
    counter = counter + 1;
end