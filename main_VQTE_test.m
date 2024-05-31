clear all;
close all;
clc
n=2;
N=2^n;% #matrix size
h=1/(N+1);%#step size
ini=randn(4,1);
H = diag(-2/h^2*ones(1,N)) + diag(1/h^2*ones(1,N-1),1) + diag(1/h^2*ones(1,N-1),-1);
stop_time=1;

sol=expm(H*stop_time)*ini;
