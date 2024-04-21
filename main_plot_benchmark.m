clear all;
close all;
clc;
% 
% %simulator data 2024/3
% qubit=[1
% 2
% 3
% 4
% 5
% 6
% 7
% 8
% 9];
% 
% 
% sigma_bar_scipy=[1.893585978
% 11.5821648
% 84.98562899
% 146.8777524
% 147.1069074
% 147.1295323
% 147.1353964
% 147.1368886
% 147.1372789];
% 
% sigma_bar_VQE=[1.893585978
% 11.5821648
% 84.98562899
% 146.8365827
% 147.0139456
% 145.8517681
% 143.1231949
% 142.6087605
% 129.1690728];
% 
% clock_time=[0.05706858635
% 0.8374683857
% 4.61222434
% 67.3436799
% 220.219125
% 825.87133
% 1455.018733
% 7119.688637
% 34493.70703];

%hardware data, 04/03/2024
qubit=[1
2
3
4
5
6
7];

clock_time=[137.1528854
179.0752852
555.944205
679.2846708
2676.528051
9392.734721
31577.56458];

sigma_bar_VQE=[1.866380997
8.165907368
28.74087495
31.82437233
45.21308269
16.99907224
14.66416434];

sigma_bar_scipy=[1.893585978
11.5821648
84.98562899
146.8777524
147.1069074
147.1295323
147.1353964];

output=[qubit,ones(size(qubit))]\log10(clock_time);

y_fit=10^output(2)*(10^output(1)).^qubit;
N_scaling=log2(10^output(1));

data{1}.x=qubit;
data{1}.y=clock_time;
data{2}.x=qubit;
data{2}.y=y_fit;
plot_config.ytick_list=[1,10,100,10^3,10^4,10^5];
plot_config.legend_list={0,'HPC testing','$y=33.31*2.51^n=0.0382*N^{1.33}$'};
plot_config.label_list={1,'Qubit numbers','Clock time (s)'};
plot_config.name='clock_time.png';
plot_config.loglog=[0,1];
plot_line(data,plot_config);


data{1}.x=qubit;
data{1}.y=sigma_bar_VQE;
data{2}.x=qubit;
data{2}.y=sigma_bar_scipy;
plot_config.ytick_list=0;
plot_config.loglog=[0,0];
plot_config.name='sigma_bar_comparison';
plot_config.label_list={1,'Qubit numbers','$\bar{\sigma}$'};
plot_config.legend_list={1,'VQE','scipy'};
plot_line(data,plot_config);