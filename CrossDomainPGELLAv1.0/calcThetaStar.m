function [Policies] = calcThetaStar(Params,Policies,rates,trajlength,rollouts,numIterations)

nSystems = size(Params,2); 

for i = 1:nSystems
    disp(['Calculating theta* for task ',num2str(i),': ',Params(i).param.ProblemID,'...',])

    policy = Policies(i).policy; % Resetting policy IMP  
    h = waitbar(0,['Please wait until theta* is calculated for task ',' ',num2str(i),': ',Params(i).param.ProblemID]);
    for k = 1:numIterations
        [data] = obtainData(policy, trajlength,rollouts,Params(i));
        dJdtheta = episodicNaturalActorCritic(policy,data,Params(i));
        policy.theta = policy.theta + rates*dJdtheta;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Uncomment this section to plot the reward function
        R = rand;
        G = rand;
        B = rand;
        for z = 1:size(data,2) % Loop over rollouts 
            r(z,:) = sum(data(z).r);
        end
        figure(i)
        plot(k,mean(r),'*','Color',[R,G,B])
        hold on 
        drawnow;            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        waitbar(k / numIterations)
    end       
    close(h) 

    Policies(i).policy = policy; % Calculating theta*
end