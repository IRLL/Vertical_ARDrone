function [ELLAmodel]=updatePGELLA(ELLAmodel,taskId,ObservedTasks,HessianArray,ParameterArray)

%--------------------------------------------------------------------------
% Update L -- Tasks(taskId).param.Group --> Know which Group .. 
%--------------------------------------------------------------------------

summ=zeros(size(ELLAmodel.L));

allowedTask = find(ObservedTasks==1);
Tg=sum(ObservedTasks);

for i=1:Tg
    summ=summ-2*HessianArray(allowedTask(i)).D*ParameterArray(allowedTask(i)).alpha*ELLAmodel.S(:,allowedTask(i))' ...
            +2*HessianArray(allowedTask(i)).D*ELLAmodel.L*ELLAmodel.S(:,allowedTask(i))*ELLAmodel.S(:,allowedTask(i))' ...
            +2*ELLAmodel.mu_two*ELLAmodel.L; 
end

ELLAmodel.L=(ELLAmodel.L-ELLAmodel.learningRate*1./Tg*summ);

%--------------------------------------------------------------------------
% Update s_{taskId} using LASSO
%--------------------------------------------------------------------------
% Determine which group taskId belongs to 
Dsqrt = HessianArray(taskId).D^.5;
%Dsqrt = sqrtm(HessianArray(idxTask).D);
target = Dsqrt*ParameterArray(taskId).alpha;
dictTransformed = Dsqrt*ELLAmodel.L;

s = full(mexLasso(target,dictTransformed,struct('lambda',ELLAmodel(1).mu_one/2)));
ELLAmodel.S(:,taskId)=s;