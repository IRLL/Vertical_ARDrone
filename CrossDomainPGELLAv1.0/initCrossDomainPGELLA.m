function [model] = initCrossDomainPGELLA(Tasks,d,k,mu1,mu2,mu3,learningRateL,learningRatePsi)
% This function initializes the parameters of the MTL objective function
% based on the input parameters.


% Initiliazing model parameters
model.L = rand(d,k); % Initial matrix L (knowledge base matrix)
model.S = zeros(k,size(Tasks,2)); % Initial matrix S (sparse task coeffcient matrix)
model.mu1 = mu1; % Coefficient for sparsity of s
model.mu2 = mu2; % Coefficient for regularization of Psi
model.mu3 = mu3; % Coefficient for regularization of L
model.learningRateL = learningRateL; % Learning rate of L
model.learningRatePsi = learningRatePsi; % Learning rate of Psi

for i=1:size(Tasks,2) % To initialize Psi with correct dimensions 
    GroupsArray(i)=Tasks(i).param.Group;
end

[Value]=max(GroupsArray);

for k = 1:Value
    for z = 1:size(Tasks,2)
        if Tasks(z).param.Group == k
            % Initial Psi_{g} per groups
            model(k).Proj.Psi = rand(Tasks(z).param.N*Tasks(z).param.M,d); 
            model(k).Proj.Group = k;
            break % Observing only one task in Group k is enough to init Psi 
        end
    end
end
