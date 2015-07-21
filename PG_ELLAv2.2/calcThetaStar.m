function [Policies] = calcThetaStar(Params,Policies,rates,trajlength,rollouts,numIterations)

nSystems = size(Params,2); 

for i = 1:nSystems
    clc;
%     R = rand;
%     G = rand;
%     B = rand;
    policy = Policies(i).policy; % Resetting policy IMP  
    for k = 1:numIterations
        disp(['@ Iteration: ', num2str(k)]);    
        [data] = obtainData(policy, trajlength,rollouts,Params(i));
        if strcmp(Params(i).param.baseLearner,'REINFORCE')
            dJdtheta = episodicREINFORCE(policy,data,Params(i));
        else
            dJdtheta = episodicNaturalActorCritic(policy,data,Params(i));
        end
        policy.theta = policy.theta + rates*dJdtheta;
%         %% Uncomment this section to plot the reward function
%         for z = 1:size(data,2) % Loop over rollouts 
%             r(z,:) = sum(data(z).r);
%         end
%         figure(i)
%         plot(k,mean(r),'*','Color',[R,G,B])
%         hold on 
%         drawnow;            
%         %%
    end            
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       

    Policies(i).policy = policy; % Calculating theta*
end