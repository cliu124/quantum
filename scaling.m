function [eta,c0,y_fit] = scaling(x,y)

%%This is performing the least squares to the input data x, y
%%Input: x, y data you want to obtain the scaling exponent as;
%% y=c_0 x^{eta}. This is firstly perform the loglog transform and then least squares
%% y_fit=c_0 x^{eta} to check how does this fitting match the original data.
%%Author: Chang Liu
%%Date: 2021/03/16

%%check whether I have zero.. get rid of these points
x(find(x<=0))=[];
y(find(y<=0))=[];

%%check whether I am in the column vector.... 
if size(x,1)<size(x,2)
    x=x';
end
if size(x,2)~=1
    error('x should be a column vector');
end

if size(y,1)<size(y,2)
    y=y';
end
if size(y,2)~=1
    error('y should be a column vector');
end

exponent=[log10(x),ones(size(x))]\log10(y);

eta=exponent(1);
c0=10^(exponent(2));
y_fit=c0*x.^(exponent(1));

end

