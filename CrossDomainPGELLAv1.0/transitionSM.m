function xn = transitionSM(x,u,param)


xd = param.param.A*x + param.param.b*u; 
xn = x + param.param.dt*xd;

% fixing the limits 

% xn(1) = min(50, max(-50, xn(1)));
% xn(2) = min(50, max(-50, xn(2)));