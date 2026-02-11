clear all;
close all;
clc;

n=8; 
N=2^n;
Nx=N;
Lx=1;
Ny=N;
Ly=1;
Nz=N;
Lz=1;
x=linspace(0,Lx,Nx+2)';
y=linspace(0,Ly,Ny+2)';
z=linspace(0,Lz,Nz+2)';

%test 1D laplacian matrix free
u_1D=sin(pi*x);
u_inner=u_1D(2:Nx+1);
ddu_1D=ddu_finite_diff_1D(u_inner,Nx,Lx);
ddu_ana_1D=-pi^2*sin(pi*x);
ddu_ana_1D=ddu_ana_1D(2:Nx+1);

res_1D=norm(ddu_1D-ddu_ana_1D);
res_1D_max=max(abs(ddu_1D-ddu_ana_1D));
%test 2D laplacian matrix-free
[X,Y]=meshgrid(x,y);
X=X'; Y=Y';
u_2D=sin(pi*X).*sin(pi*Y);
u_inner_2D=u_2D(2:Nx+1,2:Ny+1);
ddu_2D=ddu_finite_diff_2D(reshape(u_inner_2D,Nx*Ny,1),Nx,Lx,Ny,Ly);
ddu_matrix_2D=reshape(ddu_2D,Nx,Ny);
ddu_ana_2D=-2*pi^2*sin(pi*X).*sin(pi*Y);
ddu_ana_2D=ddu_ana_2D(2:Nx+1,2:Ny+1);
res_2D=norm(ddu_matrix_2D-ddu_ana_2D);
res_2D_max=max(max(abs(ddu_1D-ddu_ana_1D)));

%test 3D Laplacian matrix-free
[X,Y,Z]=meshgrid(x,y,z);
X=permute(X,[2,1,3]);
Y=permute(Y,[2,1,3]);
Z=permute(Z,[2,1,3]);
u_3D=sin(pi*X).*sin(pi*Y).*sin(pi*Z);
u_inner_3D=u_3D(2:Nx+1,2:Ny+1,2:Nz+1);
ddu_3D=ddu_finite_diff_3D(u_inner_3D,Nx,Lx,Ny,Ly,Nz,Lz);
ddu_matrix_3D=reshape(ddu_3D,Nx,Ny,Nz);
ddu_ana_3D=-3*pi^2*sin(pi*X).*sin(pi*Y).*sin(pi*Z);
ddu_ana_3D=ddu_ana_3D(2:Nx+1,2:Ny+1,2:Nz+1);
res_3D=norm(ddu_ana_3D-ddu_matrix_3D,'fro');
res_3D_max=max(max(max(abs(ddu_1D-ddu_ana_1D))));


%test eigenvalue computation is correct
opts.tol = 1e-6;
opts.maxit = 1000;
Afun_1D=@(u)ddu_finite_diff_1D(u,Nx,Lx);
eig_val_1D=eigs(Afun_1D,Nx,1,'largestreal',opts);

Afun_2D=@(u)ddu_finite_diff_2D(u,Nx,Lx,Ny,Ly);
eig_val_2D=eigs(Afun_2D,Nx*Ny,1,'largestreal',opts);

Afun_3D=@(u)ddu_finite_diff_3D(u,Nx,Lx,Ny,Ly,Nz,Lz);
eig_val_3D=eigs(Afun_3D,Nx*Ny*Nz,1,'lr',opts);

%full matrix method:

%lap_1D=lap_finite_diff_1D(Nx,Lx);
%lap_2D=lap_finite_diff_2D(Nx,Lx,Ny,Ly);
%lap_3D=lap_finite_diff_3D(Nx,Lx,Ny,Ly,Nz,Lz);

function lap_u=ddu_finite_diff_3D(u_vec,Nx,Lx,Ny,Ly,Nz,Lz)
    %u is a (NxNyNz)*1 vector. 
    % This can be reshape into the matrix, where different column is for different x and
    %different row is for different x. Thus, the total size is Nx*Ny. 
    
    %make each column as each z grid. Then each row will be equivalent to
    %the 2D vectorized grid. 
    u_z_vec=reshape(u_vec,Nx*Ny,Nz);

    % partial_xx + partial_yy. 
    ddu_ddx_plus_ddu_ddy=zeros(size(u_z_vec));
    for z_ind=1:Nz
        ddu_ddx_plus_ddu_ddy(:,z_ind)=ddu_finite_diff_2D(u_z_vec(:,z_ind),Nx,Lx,Ny,Ly);
    end

    % partial_zz
    ddu_ddz=zeros(size(u_z_vec));
    for xy_ind=1:Nx*Ny
        ddu_ddz(xy_ind,:)=ddu_finite_diff_1D(u_z_vec(xy_ind,:)',Nz,Lz)';
    end

    lap_u=reshape(ddu_ddx_plus_ddu_ddy+ddu_ddz,Nx*Ny*Nz,1);

end

function lap_u=ddu_finite_diff_2D(u_vec,Nx,Lx,Ny,Ly)
    %u is a (NxNy)*1 vector, a vectorized data 
    % This can be reshape into the matrix, where different column is for different x and
    %different row is for different x. Thus, the total size is Nx*Ny. 
%    delta_x=Lx/(1+Nx);
%    delta_y=Ly/(1+Ny);

    u_matrix=reshape(u_vec,Nx,Ny);

    ddu_ddx=zeros(size(u_matrix)); %second order derivative of u in x, result is in Nx*Ny matrix form
    for y_ind=1:Ny
        ddu_ddx(:,y_ind)=ddu_finite_diff_1D(u_matrix(:,y_ind),Nx,Lx);
    end

    ddu_ddy=zeros(size(u_matrix)); %seonnd order derivative of u in y, result is in Nx*Ny matrix form
    for x_ind=1:Nx
        ddu_ddy(x_ind,:)=ddu_finite_diff_1D(u_matrix(x_ind,:)',Ny,Ly)';
        % ddu_ddy=ddu_ddy_tranpose';
    end

    lap_u_matrix=ddu_ddx+ddu_ddy;
    lap_u=reshape(lap_u_matrix,Nx*Ny,1);

end

function lap_u=ddu_finite_diff_1D(u,Nx,Lx)
    %Input: u(x), as a column vector of Nx by 1, the value at the grid points
    %excluding boundary (x1, x2, ..., x_{Nx}). This assume
    %u(x0)=u(x_{Nx+1})=0
    %Nx: the gird point number excluding the boundary points
    %Lx: the domain size in x 
    
    lap_u=zeros(Nx,1);

    delta_x=Lx/(Nx+1); %The grid size. 
    lap_u(1)=-2/delta_x^2*u(1)+u(2)/delta_x^2;

    for ind=2:Nx-1
        lap_u(ind)=(u(ind-1)+u(ind+1))/delta_x^2-2*u(ind)/delta_x^2;
    end
    lap_u(Nx)=u(Nx-1)/delta_x^2-2/delta_x^2*u(Nx);

end


function lap=lap_finite_diff_1D(Nx,Lx)
    delta_x=Lx/(Nx+1);
    lap=-2/delta_x^2*diag(ones(1,Nx))+1/delta_x^2*diag(ones(1,Nx-1),1)+1/delta_x^2*diag(ones(1,Nx-1),-1);
end


function lap=lap_finite_diff_2D(Nx,Lx,Ny,Ly)
    lap_xx=lap_finite_diff_1D(Nx,Lx);
    lap_yy=lap_finite_diff_1D(Ny,Ly);
    Ix=eye(Nx,Nx);
    Iy=eye(Ny,Ny);
    lap=kron(Iy,lap_xx)+kron(lap_yy,Ix);

end


function lap=lap_finite_diff_3D(Nx,Lx,Ny,Ly,Nz,Lz)
    lap_xx=lap_finite_diff_1D(Nx,Lx);
    lap_yy=lap_finite_diff_1D(Ny,Ly);
    lap_zz=lap_finite_diff_1D(Ny,Ly);

    Ix=eye(Nx,Nx);
    Iy=eye(Ny,Ny);
    Iz=eye(Nz,Nz);
    
    lap=kron(Iz,kron(Iy,lap_xx))+kron(Iz,kron(lap_yy,Ix))+kron(lap_zz,kron(Iy,Ix));
end