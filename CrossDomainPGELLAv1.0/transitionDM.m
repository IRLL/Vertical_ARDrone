function  [xn]=transitionDM(x,u,param)
 
 xdot = param.param.A*x + param.param.b*u; 
 xn = x + param.param.dt*xdot;

% xn(1) = min(50, max(-50, xn(1)));
% xn(2) = min(50, max(-50, xn(2)));
% xn(3) = min(50, max(-50, xn(3)));
% xn(4) = min(50, max(-50, xn(4)));