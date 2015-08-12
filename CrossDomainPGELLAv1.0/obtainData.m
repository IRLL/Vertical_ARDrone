function data = obtainData(policy, L, H,param)
% Based on Jan Peters; codes. This code exites the inputs of the current
% task (system) and obtain the outputs to create trajectories. The observed
% trajectories are used to evaluate the reward function and to update new
% actions to excite the input of the current task.


% Perform H episodes
for trials=1:H
    % Save associated policy
    data(trials).policy = policy;
      
    % Draw the initial state
    data(trials).x(:,1) = drawStartState(param);

    % Perform a trial of length L
    for steps=1:L 
        % Draw an action from the policy
        data(trials).u(:,steps)   = drawAction(policy, data(trials).x(:,steps),param);
        % Calculate the reward
        data(trials).r(steps)   = rewardFnc(data(trials).x(:,steps), data(trials).u(:,steps));
        % Draw the next state given the action
        data(trials).x(:,steps+1) = drawNextState(data(trials).x(:,steps), data(trials).u(:,steps),param);
    end
end