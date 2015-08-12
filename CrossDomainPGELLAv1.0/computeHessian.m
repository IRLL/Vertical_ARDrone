function D = computeHessian(Policies,trajLength,numRollouts,Tasks,taskId)
    % This function selects the estimator of the Hessian for episodic
    % REINFORCE based on the current domain.
    % Can be done using point estimated ToDo

    [data]=obtainData(Policies(taskId).policy,trajLength...
        ,numRollouts,Tasks(taskId)); % Get data using new policy for D
    
    switch Tasks(taskId).param.ProblemID
        case 'SM'
            D = computeHessianSM(data,Policies(taskId).policy.sigma);
        case 'DM'
            D = computeHessianDM(data,Policies(taskId).policy.sigma);            
    end
end

