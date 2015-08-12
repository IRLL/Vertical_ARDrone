function [modelPGELLA]=updatePGELLA(modelPGELLA,taskId,ObservedTasks,HessianArray,ParameterArray,Tasks)
% This function updates the PG-ELLA model, i.e., L and S

% Updating L
[AllTasks]=find(ObservedTasks);
% From the observed tasks, determine which groups we have .. 
for l=1:length(AllTasks)
    GroupsPresentD(l,:)=Tasks(AllTasks(l)).param.Group; 
end
GroupsPresent=unique(GroupsPresentD);
for z=1:length(GroupsPresent) % sum over the goals ... 
    % Determining from observed tasks which ones are allowed 
    [allowedTaskIndexD]=getAllowedTasks(GroupsPresent(z,:),ObservedTasks,Tasks);
    allowedTaskIndexG{z}=allowedTaskIndexD(find(allowedTaskIndexD)); 
end

% Having determined the allowed tasks, compute the derivative 
sumAllOne=0;
TasksforGroupz=allowedTaskIndexG{find(GroupsPresent==Tasks(taskId).param.Group)};
sum=zeros(size(modelPGELLA(Tasks(taskId).param.Group).L));
Tg=length(TasksforGroupz);

for i=1:Tg
    sum=sum-2*HessianArray(TasksforGroupz(i)).D*ParameterArray(TasksforGroupz(i)).alpha*modelPGELLA(1).S(:,i)' ...
            +2*HessianArray(TasksforGroupz(i)).D*modelPGELLA(Tasks(taskId).param.Group).L*modelPGELLA(1).S(:,i)*modelPGELLA(1).S(:,i)' ...
            +2*modelPGELLA(1).mu_two*modelPGELLA(Tasks(taskId).param.Group).L; 
end

modelPGELLA(Tasks(taskId).param.Group).L = (modelPGELLA(Tasks(taskId).param.Group).L+modelPGELLA(1).learningRateL*1./Tg*sum);

% Update s_{taskId} using LASSO

% Determine which group taskId belongs to 
Dsqrt = real(HessianArray(taskId).D^.5);
target = Dsqrt*ParameterArray(taskId).alpha;
dictTransformed = Dsqrt*modelPGELLA(Tasks(taskId).param.Group).L;

s = full(mexLasso(target,dictTransformed,struct('lambda',modelPGELLA(1).mu_one/2)));
modelPGELLA(1).S(:,taskId)=s; 
