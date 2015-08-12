function [modelCrossDomainPGELLA,modelPGELLA] = learningTheta(Tasks,Policies,learningRate,trajLength,numRollouts,modelCrossDomainPGELLA,modelPGELLA)
% This function carries out the Cross-Domain-PG-ELLA algorithm based on the
% parameters provided as arguments.

% Initialization of the tasks counter
counter = 1;

% Initialization of the observed tasks indicator
ObservedTasks = zeros(size(Tasks,2),1);

% Defining the eange values for random indexes of observed tasks
limitOne = 1;
limitTwo = size(Tasks,2);

while ~all(ObservedTasks) % Repeat until all tasks are observed 

    [taskId] = randi([limitOne limitTwo],1);      % get a random task
          
    if ObservedTasks(taskId) == 0 % This condition is satisfied the
        ObservedTasks(taskId) = 1; % first time taskId is observed
    end
    
    
    disp(['Iteration # ',num2str(counter), ', Observed task ',num2str(taskId),': ',Tasks(taskId).param.taskLabel])

    % Carrying out PG on the observed task,

    % Generating data from the task by exiting its input with actions
    % defined by the Gaussian policy
    [data]=obtainData(Policies(taskId).policy,trajLength...
            ,numRollouts,Tasks(taskId));
    
    % Calculating the gradient of the objective function using episodic
    % REINFORCE
    dJdTheta = episodicREINFORCE(Policies(taskId).policy,data...
        ,Tasks(taskId));
    
    % Updating Policy Parameters
    Policies(taskId).policy.theta = Policies(taskId).policy.theta ...
        +learningRate*dJdTheta;

    trajLengthHessian = 10;
    numRolloutsHessian = 10;
    % Computing the Hessian for episodic REINFORCE
    HessianArray(taskId).D = computeHessian(Policies,trajLengthHessian,numRolloutsHessian,Tasks,taskId);
    ParameterArray(taskId).alpha = Policies(taskId).policy.theta;

    % Implementing the update rules for Cross-Domain PG-ELLA
    [modelCrossDomainPGELLA] = updateCrossDomainPGELLA(modelCrossDomainPGELLA,taskId,ObservedTasks,HessianArray,ParameterArray,Tasks);
    
    % Implementing the update rules for PG-ELLA
    [modelPGELLA] = updatePGELLA(modelPGELLA,taskId,ObservedTasks,HessianArray,ParameterArray,Tasks);

    counter = counter + 1;
end

disp(['All Tasks observed at iteration: ',num2str(counter-1)]);
end

