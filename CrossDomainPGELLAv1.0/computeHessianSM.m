function [Hessian] = computeHessianSM(data,sigma)
% This function estimates the Hessian for the Simple Mass task assuming
% episodic REINFORCE

nRollouts = size(data,2); 
Hes = zeros(2,2);

for i = 1:nRollouts
    
    Pos = data(i).x(1,:); 
    Reward = data(i).r; 
    PosSquare = sum(Pos.^2);
    Vel = data(i).x(2,:); 
    VelSquare = sum(Vel.^2);
    PosVel = sum(Pos.*Vel); 
    RewardDum = sum(Reward); 
    Matrix = [PosSquare PosVel; PosVel VelSquare]*RewardDum;
    Hes = Hes + Matrix; 
end

Hessian = -Hes./nRollouts;


