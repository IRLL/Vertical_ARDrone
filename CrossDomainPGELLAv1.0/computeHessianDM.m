function [Hessian] = computeHessianDM(data,sigma)
% This function estimates the Hessian for the Double Mass task assuming
% episodic REINFORCE

nRollouts = size(data,2); 
Hes = zeros(4,4);

for i = 1:nRollouts
    
    Pos = data(i).x(1,:); 
    PosSquare = sum(Pos.^2);
    Vel = data(i).x(2,:); 
    VelSquare = sum(Vel.^2);
    theta = data(i).x(3,:); 
    thetaSquare = sum(theta.^2);
    thetadot = data(i).x(4,:); 
    thetadotSquare = sum(thetadot.^2);
    PosVel = sum(Pos.*Vel);
    PosTheta = sum(Pos.*theta);
    PosThetadot = sum(Pos.*thetadot);
    VelTheta = sum(Vel.*theta);
    VelThetadot = sum(Vel.*thetadot);
    ThetaThetadot = sum(theta.*thetadot);
    
    Reward = data(i).r; 
    RewardDum = sum(Reward);
    
    Matrix = [PosSquare PosVel  PosTheta PosThetadot; PosVel VelSquare VelTheta VelThetadot; PosTheta VelTheta thetaSquare ThetaThetadot; PosThetadot VelThetadot ThetaThetadot thetadotSquare]*RewardDum;
%     Matrix = [PosSquare 0 0 0; 0 VelSquare 0 0; 0 0 thetaSquare 0; 0 0 0 thetadotSquare]*RewardDum;
    
    Hes = Hes + Matrix; 
end

Hessian = -Hes./nRollouts;