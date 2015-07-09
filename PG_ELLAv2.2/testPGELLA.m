function testPGELLA(Tasks,PGPol,learningRate,trajLength,numRollouts,numIterations,modelPGELLA)

for i=1:size(Tasks,2) % loop over all tasks
    theta_PG_ELLA = modelPGELLA.L*modelPGELLA.S(:,i);
    policyPGELLA.theta=theta_PG_ELLA;
    policyPGELLA.sigma=PGPol(i).policy.sigma;
    PolicyPGELLAGroup(i).policy=policyPGELLA;
end

for k=1:size(Tasks,2) % Test over all tasks
    for m=1:numIterations % Loop over Iterations
    % PG
    [data]=obtainData(PGPol(k).policy,trajLength,numRollouts,Tasks(k));
    
    if strcmp(Tasks(k).param.baseLearner,'REINFORCE')
        dJdTheta = episodicREINFORCE(PGPol(k).policy,data,Tasks(k));
    else
        dJdTheta = episodicNaturalActorCritic(PGPol(k).policy,data,Tasks(k));
    end
    
    % Update Policy Parameters
    PGPol(k).policy.theta=PGPol(k).policy.theta ...
                                            + learningRate*dJdTheta;

    % PG-ELLA
    [dataPGELLA]=obtainData(PolicyPGELLAGroup(k).policy,trajLength,numRollouts,Tasks(k));

    if strcmp(Tasks(k).param.baseLearner,'REINFORCE')
        dJdThetaPGELLA = episodicREINFORCE(PolicyPGELLAGroup(k).policy,dataPGELLA,Tasks(k));
    else
        dJdThetaPGELLA = episodicNaturalActorCritic(PolicyPGELLAGroup(k).policy,dataPGELLA,Tasks(k));
    end
    % Update Policy Parameters
    PolicyPGELLAGroup(k).policy.theta=PolicyPGELLAGroup(k).policy.theta ...
                                            + learningRate*dJdThetaPGELLA;

    % Computing Average in one System per iteration
    for z=1:size(data,2)
        Sum_rPG(z,:)=sum(data(z).r);
    end
    
    for z=1:size(dataPGELLA,2)
        Sum_rPGELLA(z,:)=sum(dataPGELLA(z).r);
    end
                
    Avg_rPGELLA(m,k)=mean(Sum_rPGELLA);
    Avg_rPG(m,k)=mean(Sum_rPG);
            
    figure(k)
    plot(m,mean(Sum_rPGELLA),'r*')
    hold on 
    plot(m,mean(Sum_rPG),'b*')
    drawnow
    end
end

% Plotting avverage reward
for i=1:size(Avg_rPGELLA,1) % Loop over iterations
    Toplot_rPGELLA(i,:)=mean(Avg_rPGELLA(i,:)); % Average over each column
end
for i=1:size(Avg_rPG,1)
    Toplot_rPG(i,:)=mean(Avg_rPG(i,:));
end

figure
[Y20]=fastsmooth(Toplot_rPG,10,1,1);
plot(Y20,'b--','Linewidth',2)
hold on 

[Y1]=fastsmooth(Toplot_rPGELLA,10,1,1);
plot(Y1,'-.g','Linewidth',2)
grid on 
legend('PG','PG-ELLA')