%% Codes for Cross Domain PG Efficient Lifelong Learning
% This sample code implements the Cross-Domain-PG-ELLA algorithm through
% a very simple example. The lifelong learning algorithm considers two
% different task domains, namely, Single Mass (SM) systems and Double Mass
% (DM) systems.
%
% The calculation of theta* is not included in this sample code, it has
% been already precalculated and it is contained in the Data.mat file
% attached.
%
% The code shows the plots of the regulated states of all the tasks after
% the Cross-Domain-PG-ELLA have been executed.
%
% Reference paper:
% Haitham Bou Ammar, Eric Eaton, Jose Marcio Luna & Paul Ruvolo. Autonomous
% Cross-Domain Knowledge Transfer in Lifelong Policy Gradient 
% Reinforcement Learning. In Proceedings of the International Joint 
% Conference on Artificial Intelligence (IJCAI), 2015. Buenos Aires, Argentina, July.


close all
clear all
clc

rng(2)
%% Loading predefined tasks and values of theta*
addpath(genpath('spams-matlab'))

% Run compile of spams-matlab 

disp(['Welcome to your sample code of Cross-Domain-PG-ELLA!'])
disp(' ');
disp(['Loading precalculated theta* for all tasks...']);
load Data
disp(['All ',num2str(size(Tasks,2)),' tasks and theta* have been loaded!']);
disp(' ');
disp('Theta has been calculated using the following parameters:')
disp(' ');

% Displaying the parameters of the experiment:
disp(['Number of Simple Mass (SM) systems: ',num2str(nSMSystems)]);
disp(['Number of Double Mass (DM) systems: ',num2str(nDMSystems)]);
disp(['Learning rate used to learn theta*: ',num2str(learningRate)]);
disp(['Base learner for theta*: ', baseLearner]);
disp(['Policy type: ', poliType]);
disp(['Discount factor gamma: ',num2str(gamma)]);
disp(['Trajectory length to calculate theta*: ', num2str(trajLengthTheta)]);
disp(['Horizon to calculate theta*: ', num2str(numRolloutsTheta)]);
disp([' ']);
disp(['Press any key to continue...']);

pause
clc

%% Learning the Cross-Domain PG-ELLA

% Parameters for Cross-Domain Lifelong Learning
trajLength = 2000;%2000
numRollouts = 200;

mu1 = exp(-5); % Sparsity coefficient
mu2 = exp(-5); % Regularization coefficient for L
mu3 = exp(-5); % Regularization coefficient for \Psi

% Calculating the number of rows of matrix L which corresponds to the 
% maximum number of policy parameters among all tasks (systems).
% here we assume theta*phi -- same number of features is same dim of theta
for i = 1:size(Tasks,2)
    dl(i) = Tasks(i).param.N; % dimensionality of the state space 
end
d = max(dl);

k = 1; % Number of latent dimensions (i.e., L \in \mathbb{R}^{d x k}

% Determining the learning reates for the update rules of L and Psi
learningRateL = 0.0000001; % 0.000001 Learinig rate for L
learningRatePsi = 0.0000001; % 0.0000001 Learning rate for Psi

% Initializing the model

% Model for Cross-Domain-PG-ELLA
[modelCrossDomainPGELLA] = initCrossDomainPGELLA(Tasks,d,k,mu1,mu2,mu3,learningRateL,learningRatePsi);


% Determining the learning rate for the update rules of L
learningRateL = 0.001; % 0.000001 Learinig rate for L
% Model for PG-ELLA
[modelPGELLA] = initPGELLA(Tasks,k,mu1,mu2,learningRateL);

disp(['Executing learning process using Cross-Domain-PG-ELLA and PG-ELLA...'])
disp(['Please be patient, depending on the trajectory length and the horizon chosen, this can take several minutes...'])
disp(['Trajectory length to update Psi, L and S: ', num2str(trajLength)]);
disp(['Horizon to to update Psi, L and S: ', num2str(numRollouts)]);
% Executing the learning process
[modelCrossDomainPGELLA,modelPGELLA] = learningTheta(Tasks,Policies,learningRate,trajLength,numRollouts,modelCrossDomainPGELLA,modelPGELLA);

disp(['The model has been learned!'])

waitfor(msgbox('The model has been learned!'));
clc
%% Testing Phase

disp(['Testing trajectory regulation for all ',num2str(size(Tasks,2)),' tasks']);
disp(['Please be patient, this can take several minutes...'])

% Parameters for testing the algorithms
trajLengthTest = 5000;

testingTheta(Tasks,Policies,trajLengthTest,modelCrossDomainPGELLA)
% testingTheta(Tasks,Policies,trajLengthTest,modelPGELLA)

disp(['Testing of trajectories finalized!'])

waitfor(msgbox('Click OK to close all plots'));

close all
clc


% Parameters for testing the algorithms
trajLengthComp = 50;
numRolloutsComp = 50;
numIterationsComp = 500;

disp(['Finally, testing online performance for all ',num2str(size(Tasks,2)),' tasks']);
disp(['Calculating the average reward for each domain']);
disp([' ']);
disp(['Trajectory length to test performance: ', num2str(trajLengthComp)]);
disp(['Horizon to test performance: ', num2str(numRolloutsComp)]);
disp(['Number of iterations to test performance: ', num2str(numIterationsComp)]);

% Learning rates for gradient descent
learningRateComp = .1;

% Creating Normal PG policy 
[PGPol]=constructPolicies(Tasks);

% Vector ccontaining the number of tasks per domain
nGroupDomain = [nSMSystems,nDMSystems];

compPerformance(Tasks,PGPol,learningRateComp,trajLengthComp,numRolloutsComp,numIterationsComp,modelCrossDomainPGELLA,modelPGELLA,nGroupDomain)
