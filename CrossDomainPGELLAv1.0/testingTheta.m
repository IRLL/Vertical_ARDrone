function testingTheta(Tasks,PGPol,trajLengthTest,modelCrossDomainPGELLA)
% This function tests that the implemented policy regulates the states of
% the system to the desired reference value. For this example Xref = 0

for i = 1:size(Tasks,2) % loop over all tasks
    % Take the task and determine to which group it belongs 
    G_Task = Tasks(i).param.Group;
    Groups(i) = G_Task;
    % Building testing policies for Cross-Domain PG-ELLA
    Psi = modelCrossDomainPGELLA(G_Task).Proj.Psi;
    theta_Task = Psi*modelCrossDomainPGELLA(1).L*modelCrossDomainPGELLA(1).S(:,i);
%     theta_Task = modelCrossDomainPGELLA(G_Task).L*modelCrossDomainPGELLA(1).S(:,i);
    policy.theta = theta_Task;
    policy.sigma = PGPol(i).policy.sigma;
    modelCrossDomainPGELLA(i).policy = policy;
end

% Test Cross-Domain PG-ELLA
%--------------------------------------------------------------------------

for k = 1:size(Tasks,2) % Test over all tasks 

    disp(['Testing task ',num2str(k),': ',Tasks(k).param.taskLabel,'...',])
    h = waitbar(0,['Testing task ',num2str(k),': ',Tasks(k).param.taskLabel]);
    
    clear x u
    
    % Draw initial state
    x(:,1) = drawStartState(Tasks(k));
    % Perform a trial of length L
    for steps = 1:trajLengthTest 
        % Draw an action from the policy
        u(:,steps) = modelCrossDomainPGELLA(k).policy.theta'*x(:,steps);
        % Draw next state from environment      
        x(:,steps + 1) = drawNextState(x(:,steps), u(:,steps),Tasks(k));
        waitbar(steps / trajLengthTest)
    end
    close(h)
    figure1 = figure;
    axes1 = axes('Parent',figure1);
    plot1 = plot([0:size(x,2)-1]*Tasks(k).param.dt,x','LineWidth',3);
    title(['Trajectories of task ',num2str(k),' (',Tasks(k).param.taskLabel,')'])
    ylabel('State Trajectories','FontSize',14)
    xlabel('time(s)','FontSize',14)
    legend1 = legend(axes1,'show');
    set(legend1,'FontSize',14);
    for n = 1:size(plot1,1)
        set(plot1(n),'DisplayName',['x',num2str(n)]);
    end
end