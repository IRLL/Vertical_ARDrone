function [model]=initPGELLA(Tasks,k,mu_one,mu_two,learningRate)

model.S=zeros(k,size(Tasks,2));
model.mu_one=mu_one;
model.mu_two=mu_two; 
model.learningRate=learningRate;

model.L=rand(Tasks(1).param.N*Tasks(1).param.M,k);
