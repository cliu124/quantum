clear all;
close all;
clc;

n=3;
N=2^n;
h=1/(N+1);
D2=-2/h^2*diag(ones(1,N))+1/h^2*diag(ones(1,N-1),1)+1/h^2*diag(ones(1,N-1),-1);
results=eig(D2);
