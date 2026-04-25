%% Visualization for RMPC - single run

clc
clear
close all

% load iterative data
data_each_iter = load("data_each_Step.csv");
x1_ub = data_each_iter(:,1:10);
x1_lb = data_each_iter(:,11:20);
x2_ub = data_each_iter(:,21:30);
x2_lb = data_each_iter(:,31:40);
x1_pred = data_each_iter(:,41:50);
x1_true = data_each_iter(:,51:60);
x2_pred = data_each_iter(:,61:70);
x2_true = data_each_iter(:,71:80);

% load time trajectory
data_traj = load("whole_traj.csv");
ref = data_traj(:,1);
x1_output_save = data_traj(:,2);
x2_output_save = data_traj(:,3);
u_nominal = data_traj(:,4);
u_applied = data_traj(:,5);

% Plot
figure;
time_stamp = linspace(0,189,190);
ax1 = subplot(3,1,1);
hold on
h1 = plot(time_stamp, ones([190,1])*2.5,"--",LineWidth=1.5,Color="#EDB120"); % x1_ub
plot(time_stamp, ones([190,1])*-2,"--",LineWidth=1.5,Color="#EDB120"); % x1_lb
h2 = plot(time_stamp, ref,":",LineWidth=2,Color=[0.5, 0.5, 0.5]); % ref
h3 = plot(time_stamp,x1_output_save,"b",LineWidth=2); % x1_traj
h4 = plot(nan,nan,"r",linewidth=2);
h5 = plot(nan,nan,"g",linewidth=2);
h6 = fill(nan,nan, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Fill area
axis([0,189,-2.5,3])
legend([h1,h2,h3,h4,h5,h6],{"Constraints","Reference","State output","Prediction","Ground truth","Tube"}, "NumColumns", 3, "location","southeast","Fontsize",13.5)
box on
set(gca, "FontSize",14)
set(gca, "XTickLabel",[])
set(gca,"Position",[0.13,0.709264705882353,0.775,0.215735294117647])

ax2 = subplot(3,1,2);
hold on
h1 = plot(time_stamp, ones([190,1])*3.5,"--",LineWidth=1.5,Color="#EDB120"); % x1_ub
plot(time_stamp, ones([190,1])*-3.5,"--",LineWidth=1.5,Color="#EDB120"); % x1_lb
h2 = plot(time_stamp,x2_output_save,"b",LineWidth=2); % x2_traj
h4 = plot(nan,nan,"r",linewidth=2);
h5 = plot(nan,nan,"g",linewidth=2);
h6 = fill(nan,nan, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Fill area
axis([0,189,-4,4])
box on
legend([h1,h2,h4,h5,h6],{"Constraints","State output","Prediction","Ground truth","Tube"}, "NumColumns", 3, "location","southeast","Fontsize",13.5)
set(gca, "FontSize",14)
set(gca, "XTickLabel",[])
set(gca, "Position",[0.13,0.468065287868135,0.775,0.215735294117647])   

ax3 = subplot(3,1,3);
hold on
h1 = plot(time_stamp, ones([190,1])*5,"--",LineWidth=1.5,Color="#EDB120"); % x1_ub
plot(time_stamp, ones([190,1])*-5,"--",LineWidth=1.5,Color="#EDB120"); % x1_lb
h2 = plot(time_stamp,u_nominal,"g",LineWidth=2); % nominal_u
h3 = plot(time_stamp,u_applied,"r:",LineWidth=2);
axis([0,189,-5.5,5.5])
legend([h1,h2,h3],{"Constraints","Nominal input","Applied input"},"NumColumns", 3, "location","southeast","Fontsize",13.5)
set(gca, "FontSize",14)
set(gca,"Position",[0.13,0.225537848605578,0.775,0.215735294117647])
box on
xlabel("Timestep")

set(gcf,"Position",[854.5999999999999,210.6,862.4000000000001,778.3999999999999])

% plot highlights
figure
start_time = 78;
hold on
time_range = linspace(start_time-1, start_time + 9, 11);
lower_bound = [x1_output_save(start_time),x1_pred(start_time+1, :) + x1_lb(start_time+1, :)];
upper_bound = [x1_output_save(start_time),x1_pred(start_time+1, :) + x1_ub(start_time+1, :)];
fill([time_range, fliplr(time_range)], [lower_bound, fliplr(upper_bound)], 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Fill area
h1 = plot(time_stamp, ones([190,1])*2.5,"--",LineWidth=1.5,Color="#EDB120"); % x1_ub
plot(time_stamp, ones([190,1])*-2,"--",LineWidth=1.5,Color="#EDB120"); % x1_lb
h2 = plot(time_stamp, ref,":",LineWidth=2,Color=[0.5, 0.5, 0.5]); % ref
h3 = plot(linspace(0,start_time-1,start_time),x1_output_save(1:start_time),"b",LineWidth=2); % x1_traj
h4 = plot(start_time-1,x1_output_save(start_time),"k.",MarkerSize=20)
h5 = plot(linspace(start_time-1,start_time+9,11),[x1_output_save(start_time),x1_pred(start_time+1,:)],"r",linewidth=2)
h6 = plot(linspace(start_time-1,start_time+9,11),[x1_output_save(start_time),x1_true(start_time+1,:)],"g",linewidth=2)
box on
xlabel("Timestep")
axis([0,189,-2.5,3])
set(gca,"FontSize",10)
set(gcf,"Position",[458.6,293.8,213.6,152.8])

% plot highlights
figure
start_time = 78;
hold on
time_range = linspace(start_time-1, start_time + 9, 11);
lower_bound = [x2_output_save(start_time),x2_pred(start_time+1, :) + x2_lb(start_time+1, :)];
upper_bound = [x2_output_save(start_time),x2_pred(start_time+1, :) + x2_ub(start_time+1, :)];
fill([time_range, fliplr(time_range)], [lower_bound, fliplr(upper_bound)], 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Fill area
h1 = plot(time_stamp, ones([190,1])*3.5,"--",LineWidth=1.5,Color="#EDB120"); % x1_ub
plot(time_stamp, ones([190,1])*-3.5,"--",LineWidth=1.5,Color="#EDB120"); % x1_lb
h3 = plot(linspace(0,start_time-1,start_time),x2_output_save(1:start_time),"b",LineWidth=2); % x1_traj
h4 = plot(start_time-1,x2_output_save(start_time),"k.",MarkerSize=20)
h5 = plot(linspace(start_time-1,start_time+9,11),[x2_output_save(start_time),x2_pred(start_time+1,:)],"r",linewidth=2)
h6 = plot(linspace(start_time-1,start_time+9,11),[x2_output_save(start_time),x2_true(start_time+1,:)],"g",linewidth=2)
box on
xlabel("Timestep")
axis([0,189,-3.5,3.5])
set(gca,"FontSize",10)
set(gcf,"Position",[458.6,293.8,213.6,152.8])

