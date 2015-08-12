function [model,allowedTaskIndexG] = updateCrossDomainPGELLA(model,taskId,ObservedTasks,HessianArray,ParameterArray,Tasks)
% This function updates the Cross-Domain-PG-ELLA model, i.e., Psi, L and S

% Updating L
[AllTasks] = find(ObservedTasks);
% From the observed tasks, determine current observed groups
for l = 1:length(AllTasks)
    GroupsPresentD(l,:) = Tasks(AllTasks(l)).param.Group; 
end
GroupsPresent = unique(GroupsPresentD);
for z = 1:length(GroupsPresent) % sum over the goals ... 
    % Determining from observed tasks which ones are allowed 
    [allowedTaskIndexD] = getAllowedTasks(GroupsPresent(z,:),ObservedTasks,Tasks);
    allowedTaskIndexG{z} = allowedTaskIndexD(find(allowedTaskIndexD)); 
end

% Having determined the allowed tasks, compute the derivative
sumAllOne = 0;
for m = 1:length(GroupsPresent) % Outer Summation over Groups
    TasksforGroupz = allowedTaskIndexG{m};
    sumG = 0; % Resetting SumG
    for b = 1:size(TasksforGroupz,1) % number of rows determines how many tasks are in group z (i.e., inner summation)
        prodOne = -2*model(GroupsPresent(m)).Proj.Psi'*HessianArray(TasksforGroupz(b)).D*ParameterArray(TasksforGroupz(b)).alpha*model(1).S(:,TasksforGroupz(b))';
        prodTwo = 2*model(GroupsPresent(m)).Proj.Psi'*HessianArray(TasksforGroupz(b)).D*model(GroupsPresent(m)).Proj.Psi*model(1).L*model(1).S(:,TasksforGroupz(b))*(model(1).S(:,TasksforGroupz(b)))';
        sumG = sumG+prodOne+prodTwo;
    end
    sumAllOne = sumAllOne + 1./size(TasksforGroupz,1) + 2*model(1).mu3*model(1).L;
    
end

model(1).L = (model(1).L - model(1).learningRateL*sumAllOne); 


sum=0; 

% Update Psi_{g} for group g
[IndexAllowedTasksDum] = getAllowedTasks(Tasks(taskId).param.Group,ObservedTasks,Tasks); % This will get the observed tasks corresponding to group g

IndexAllowedTasks = IndexAllowedTasksDum(find(IndexAllowedTasksDum)); % This is a hack .. the problem is that in the first place i can add a zero if I didn't observe the task and that is not a valid index, so what I do is find the non zero entries and use them 
IndexAllowedTasks = unique(IndexAllowedTasks);
% Determine the correct Psi to use 
for k = 1:size(model,2)
    if model(k).Proj.Group == Tasks(taskId).param.Group
        Psi = model(k).Proj.Psi;
        indPsi = k;
    end
end
sum = zeros(size(Psi));

% Create a zero sum matrix of size Psi_{g} 
Tg = length(IndexAllowedTasks); % Number of tasks so far in Group G_{g} 
for i = 1:Tg
    sum = sum-2*HessianArray(IndexAllowedTasks(i)).D*ParameterArray(IndexAllowedTasks(i)).alpha*model(1).S(:,IndexAllowedTasks(i))'*model(1).L' ...
            +2*HessianArray(IndexAllowedTasks(i)).D*Psi*model(1).L*model(1).S(:,IndexAllowedTasks(i))*(model(1).L*model(1).S(:,IndexAllowedTasks(i)))'+2*model(1).mu2*Psi;
end
Psi = (Psi - model(1).learningRatePsi*1./Tg*sum);
model(indPsi).Proj.Psi = Psi;

% Update s_{taskId} using LASSO

% Determine which group taskId belongs to 
G = Tasks(taskId).param.Group;
Dsqrt = real(HessianArray(taskId).D^.5);
target = Dsqrt*ParameterArray(taskId).alpha;
dictTransformed = Dsqrt*model(G).Proj.Psi*model(1).L;

s = full(mexLasso(target,dictTransformed,struct('lambda',model(1).mu1/2)));
model(1).S(:,taskId)=s;