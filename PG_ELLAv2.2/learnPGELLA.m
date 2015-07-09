function [modelPGELLA] = learnPGELLA(Tasks,Policies,learningRate,trajLength,numRollouts,modelPGELLA)

clc
counter = 1;
ObservedTasks = zeros(size(Tasks,2),1);
limitOne = 1;
limitTwo = size(Tasks,2);



while ~all(ObservedTasks) % Repeat until all tasks are observed
    
    [taskId] = randi([limitOne limitTwo],1); % Pick a random task
          
    if ObservedTasks(taskId) == 0 % Entry is set to 1 when corresponding task is observed
        ObservedTasks(taskId) = 1;
    end

%--------------------------------------------------------------------------
% Policy Gradients on taskId
%--------------------------------------------------------------------------
    [data]=obtainData(Policies(taskId).policy,trajLength...
        ,numRollouts,Tasks(taskId));
    
    if strcmp(Tasks(taskId).param.baseLearner,'REINFORCE')
        dJdTheta = episodicREINFORCE(Policies(taskId).policy,data...
        ,Tasks(taskId));
    else
        dJdTheta = episodicNaturalActorCritic(Policies(taskId).policy,data...
        ,Tasks(taskId));
    end
    
    % Updating theta*
    Policies(taskId).policy.theta = Policies(taskId).policy.theta ...
        +learningRate*dJdTheta;

%--------------------------------------------------------------------------                

    % Computing Hessian 
    [data]=obtainData(Policies(taskId).policy,trajLength,numRollouts,Tasks(taskId));
    
    D=computeHessian(data,Policies(taskId).policy.sigma);
    HessianArray(taskId).D=D;
    
    ParameterArray(taskId).alpha=Policies(taskId).policy.theta;
%--------------------------------------------------------------------------
    % Perform Updating L and S
    [modelPGELLA]=updatePGELLA(modelPGELLA,taskId,ObservedTasks,HessianArray,ParameterArray); % Perform PGELLA for that Group 
%--------------------------------------------------------------------------
     
    disp(['Iterating @: ',num2str(counter)])
    counter=counter+1;  
end

disp(['All Tasks observed @: ',num2str(counter-1)]);