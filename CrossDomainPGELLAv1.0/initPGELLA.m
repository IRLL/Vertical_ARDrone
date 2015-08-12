function [model]=initPGELLA(Tasks,k,mu1,mu2,learningRateL)

model.T = 0;
model.S = zeros(k,size(Tasks,2));
model.mu_one = mu1;
model.mu_two = mu2; 
model.learningRateL = learningRateL;

% Determine the number of Groups
for i = 1:size(Tasks,2)
    GroupsArray(i) = Tasks(i).param.Group;
end

[Value,Idx]=max(GroupsArray);

for l = 1:Value
    for z = 1:size(Tasks,2)
        if Tasks(z).param.Group == l
            model(l).L = rand(Tasks(z).param.N*Tasks(z).param.M,k); % Initial L for each group
            break
        end
    end
end