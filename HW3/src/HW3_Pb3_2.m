close all; clear all; clc;

%% Solve the dual form of the hard-margin SVM using quadratic programming

% training data from 2 classes & their label
% Assume label of class w1 is 1, and label of class w2 is -1
X = [1,1; 2,2; 2,0; 0,0; 1,0; 0,1];
Y = [1; 1; 1; -1; -1; -1];

% construct H matrix for quadprog() : it is a symmetric matrix
H = zeros(length(Y));
for i = 1:length(Y)
    for j= 1:length(Y)
        H(i,j) = Y(i)*Y(j)*X(i,:)*X(j,:)';
    end
end

% define f vector for the 2nd part of the minimization problem
f = - ones(length(Y), 1);

% declare A,b as empty since we do not use this constraint
A = [];
b = [];

% define Aeq, beq for the constraint sum(alpha_i * y_i) = 0
Aeq = [Y, Y, Y, Y, Y, Y]';
beq = zeros(length(Y), 1);

% define lb, for the constraint alpha_i >= 0 for i=1..n
lb = zeros(length(Y), 1);

% find the alpha's : we should have 2 null coefficients (for the
% non-support vectors)

disp('Solving the dual problem parameterized by the alpha coefs :');
alpha = quadprog(H,f,A,b,Aeq,beq,lb)

% Check the results are coherent : we compute w & b
w = (alpha .* Y)' * X;
fprintf('The weight vector w is = (%d, %d)\n', w)

% We need to select only the support vectors to compute b
Xsv = [1,1; 2,0; 1,0; 0,1];
Ysv = [1; 1; -1; -1];
b = mean(Ysv' - w * Xsv');
fprintf('The bias b is = %d\n', b)

