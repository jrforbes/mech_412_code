% James Richard Forbes
% 2012/09/06

% This is a piece of sample code written by me, Prof. Forbes. 
% It's purpose is to show students how to do some basic things in matlab. 

clear all % Clears all variables.
clc % Clears the command window.
close all % Closes all figures/plots.
format long 

% If you're unsure about what a function does, just type ``help XXX'' in
% the command windown. For instance, some useful functions used for control 
% systems design can be found by typing ``help control''.

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear algebra stuff

A = [1 2 3;
     10 11 12;
     20 21 29];

% eigenvalues; help eig
[eig_vec_A,eig_val_A] = eig(A)

% Rank of a matrix; help rank
rank_A = rank(A)

% Null space of a matrix; help null
null_A = null(A)

% Another example
Q = [1 2 1;
     2 4 2;
     1 2 3];

rank_Q = rank(Q)

null_Q = null(Q)

det_Q = det(Q)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loops and if/ifelse stuff

% For loop
div_by_2 = [];
div_by_5 = [];
for lv1 = 1:10
    
    if rem(lv1,2) == 0
        div_by_2 = [div_by_2; lv1];
        
    elseif rem(lv1,5) == 0        
        div_by_5 = [div_by_5; lv1];
        
    else
       % Do nothing.
       
    end   
end
div_by_2
div_by_5

% break % This stops the code here. If you want to run the code below, just 
      % comment out the break using a `` % ''.

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Control stuff (Not needed for AER 540, but good to know none the less.)

% Transfer function; help tf
a = 1;
b = 2;
c = 5;
tf_1 = tf([1 a],[1 b c])

% lsim example; help lsim
n = 500;
t_span = linspace(0,10,n); % time span of 10 seconds, 500 points within the time span. 
[y,t,x] = lsim(tf_1,0.1*sin(t_span*5),t_span);

% tf to ss; help tf2ss
[A,B,C,D] = tf2ss(cell2mat(tf_1.num), cell2mat(tf_1.den))

% plot; help plot
% First, create font size, line size, and line width variables. 
% Do not hand in plots without clear (large) labels.
font_size = 15;
line_size = 15;
line_width = 2;

% Now, plot y vs time.
figure
plot(t,y,'Linewidth',line_width);
hold on
xlabel('Time (s)','fontsize',font_size,'Interpreter','latex');
ylabel('Output y (units)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on

