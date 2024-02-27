clear all;
close all;
clc;
n=8;
N=2^n;
kx=1; kz=1;
Re=358;
omega=0;

[xi, DM] = chebdif(N+2, 4);
D1_bc=DM(2:end-1,2:end-1,1);
D2_bc=DM(2:end-1,2:end-1,2);
[~,D4_bc]=cheb4c(N+2);
I_bc=eye(N,N);
zero_bc=zeros(N,N);
U_bar=diag(xi(2:end-1));
d_U_bar=I_bc;
dd_U_bar=zero_bc;

K2= kx^2+kz^2; % Total wave number, in convienent for calculation
zi=sqrt(-1);
A11=(D4_bc-2*K2*D2_bc+K2^2*I_bc)/Re+dd_U_bar*zi*kx*I_bc-zi*kx*U_bar*(D2_bc-K2*I_bc); %%Orr-Sommerfeld operator
A21= -zi*kz*d_U_bar*I_bc; %Coulping operator
A22= -zi*kx*U_bar*I_bc+1/Re*(D2_bc-K2*I_bc); %Squire operator  
inv_lap=inv([D2_bc-K2*I_bc, zero_bc; zero_bc, I_bc]);

A= inv_lap*[A11, zero_bc; A21, A22];
B=inv_lap*[-zi*kx*D1_bc, -K2*I_bc, -zi*kz*D1_bc; zi*kz*I_bc, zero_bc, -zi*kx*I_bc];

C=[zi*kx*D1_bc, -zi*kz*I_bc;
                    K2*I_bc, zero_bc; 
                   zi*kz*D1_bc, zi*kx*I_bc]/K2;
Bx=inv_lap*[-zi*kx*D1_bc;
    zi*kz*I_bc];
Cu=[zi*kx*D1_bc, -zi*kz*I_bc]/K2;

Gradient=[zi*kx*I_bc;
    D1_bc;
    zi*kz*I_bc];

H_unweight=C*inv(1i*omega-A)*B;
H_unweight_ux=Cu*inv(1i*omega-A)*Bx;
sigma_bar_unweight=max(svd(H_unweight));
sigma_bar_unweight_ux=max(svd(H_unweight_ux));
num=round(abs(N+1));

%create D0
D0=[];
vec=(0:1:num)';
for j=0:1:num
    D0=[D0 cos(j*pi*vec/num)];
end

inte=zeros(N+2,N+2);
for i =1:2:N+2
  inte(:,i)=2/(1-(i-1)^2)*ones(N+2,1);
end
weight_full=inte*inv(D0);
weight_full=weight_full(1,:)';
weight_bc=weight_full(2:end-1);
Iw_root_bc=sqrtm(diag(weight_bc));
H=blkdiag(Iw_root_bc,Iw_root_bc,Iw_root_bc)*C*inv(zi*omega-A)*B*blkdiag(inv(Iw_root_bc),inv(Iw_root_bc),inv(Iw_root_bc));
sigma_bar=max(svd(H));
sigma_bar_eig=sqrt(max(eig(H*H')));

H_ux=Iw_root_bc*H_unweight_ux*inv(Iw_root_bc);
sigma_bar_ux=max(svd(H_ux));

H_grad_ux=blkdiag(Iw_root_bc,Iw_root_bc,Iw_root_bc)*Gradient*H_unweight_ux*inv(Iw_root_bc);
sigma_bar_ux_grad=max(svd(H_grad_ux));
sigma_bar_ux_grad_eig=max(sqrt(eig(H_grad_ux'*H_grad_ux)));
