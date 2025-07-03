% visualization for model evaluation

clc;
clear;
close all

loss_all = load("loss_history.csv");
val_loss = loss_all(1,:);
train_loss = loss_all(2,:);

response = load("save_response_evaluation.csv");
data_one_step = load("save_one_step.csv");

% subplot 1
subplot(2,2,1)
plot(linspace(1,1500,1500),train_loss,"b","LineWidth",1.5)
hold on
plot(linspace(1,1500,1500),val_loss,"r","LineWidth",1.5)
xlabel("Epoch","FontSize",14);
ylabel("Loss","FontSize",14);
legend("Training loss","Validation loss","Fontsize",12)
title("(a) Loss curve")
grid on 
box on
set(gca,"FontSize",14)

% subplot 2
subplot(2,2,2)
x = linspace(1, 10, 10);
ref = data_one_step(1,:);
true_x1 = data_one_step(2,:);
pred_x1 = data_one_step(3,:);
hold on
h1 = plot(x,ref,"linewidth",2,"Color",[0.4,0.4,0.4]);
h2 = plot(x,true_x1,"r-o","linewidth",2);
h3 = plot(x,pred_x1,"b","linewidth",2);

xlabel("Timestep","FontSize",14);
ylabel("$x_1$","Interpreter","latex","FontSize",14);
legend([h1,h2,h3],{"Reference","Ground Truth","Prediction"},"Fontsize",12,"location","southeast")
title("(b) MPC in one step")
grid on 
box on
axis([-inf,inf,0,2.4])
set(gca,"FontSize",14)


% subplot 3
subplot(2,2,3)
true_x1 = response(1,:)*5;
median_x1 = response(2,:)*5;
lb_x1 = response(3,:)*5;
ub_x1 = response(4,:)*5;
x = linspace(1, 10, 10);

hold on
h3 = fill([x, fliplr(x)], [lb_x1, fliplr(ub_x1)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h1 = plot(linspace(1,10,10),median_x1,"r",LineWidth=2);
h2 = plot(linspace(1,10,10),true_x1,"b:",LineWidth=1.8);
box on 
grid on
xlabel("Timestep","FontSize",14);
ylabel("$x_1$","Interpreter","latex","FontSize",14);
legend([h1,h2,h3],{"Median","Val. data","Tube"},"Fontsize",10,"Numcolumns",2,"position",[0.137406724500612,0.132755025019276,0.307650277236772,0.066515364527036])
title("(c) x_1 prediction")
grid on 
box on
axis([-inf,inf,-5.2,5])
set(gca,"FontSize",14)


% subplot 4 
subplot(2,2,4)
true_x2 = response(5,:)*4.2;
median_x2 = response(6,:)*4.2;
lb_x2 = response(7,:)*4.2;
ub_x2 = response(8,:)*4.2;
x = linspace(1, 10, 10);

hold on
h3 = fill([x, fliplr(x)], [lb_x2, fliplr(ub_x2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h1 = plot(linspace(1,10,10),median_x2,"r",LineWidth=2);
h2 = plot(linspace(1,10,10),true_x2,"b:",LineWidth=1.8);
box on 
grid on
xlabel("Timestep","FontSize",14);
ylabel("$x_2$","Interpreter","latex","FontSize",14);
legend([h1,h2,h3],{"Median","Val. data","Tube"},"Fontsize",10,"Numcolumns",2,"position",[0.579209630034202,0.13415118557774,0.307650277236772,0.066515364527036])
title("(d) x_2 prediction")
grid on 
box on
axis([-inf,inf,-5.2,5])
set(gca,"FontSize",14)







set(gcf,"Position",[744,377,732.2,573]);