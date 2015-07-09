function xn = drawNextState(x,u,param,i);
% Based on Jan Peters' code. 

A = [0 1; -param.param.k/param.param.Mass -param.param.d/param.param.Mass];
b = [0; 1/param.param.Mass];
xd = A*x + b*u; 
xn = x + param.param.dt*xd;

% % fixing the limits 
% 
% if xn(1) > 20
%     xn(1) = 20;
% end
% if xn(1) < -20 
%    xn(1) = -20; 
% end