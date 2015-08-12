function xn = drawNextState(x,u,Tasks)
% This code carries out the integration of the dynamics of the task

xd = Tasks.param.A*x + Tasks.param.b*u; 
% xn = x + .01*xd;
xn = x + Tasks.param.dt*xd;