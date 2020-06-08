load('assign2_data2.mat');
L_R = 250;
 
% D = 8;
% Layer2_Size = 64;
 
% D = 16;
% Layer2_Size = 128;
 
D = 32;
Layer2_Size = 256;
 
 
Epoch_Number = 50;
neta = 0.15;
mini_batch = 200;
alpha = 0.85;
 
R = 0.01*rand(L_R,D);
 
W_L = length(words); 
V_L = length(vald);
 
trainx = double(trainx); 
traind = double(traind);
testx = double(testx);   
testd = double(testd);
valx = double(valx);     
vald = double(vald);
 
X = trainx';
Y = traind';
 
X_valid = valx';
Y_valid = vald';
 
X_test = testx';
Y_test = testd';
 
N = length(X);
Layer1_Size  = size(X, 2)*D;  % Input Layer
Labels_NO = 250;   % Output classes   
 
%% Initializing the Parameters
m = size(trainx, 1);
 
%% Randomly Initialize Weights
 
% Ranging between -epsilon to +epsilon
Weight1 = 0.01*randn(Layer2_Size,Layer1_Size+1);
Weight2 = 0.01*randn(Labels_NO,Layer2_Size+1);
 
%% Training
% Gradient Array
grad = 0;
% Storage of performance and Cost
pp=zeros(1,Epoch_Number);
J=zeros(1,Epoch_Number);
C=zeros(1,Epoch_Number);
 
% Run for Epochs
for epochs=1:Epoch_Number
    
% Compute the gradient and cost via function
[R,J(epochs), Weight1,Weight2] = Gradient_Cost2(Weight1,Weight2,...
    X, Y,neta,mini_batch,alpha,L_R,R,D);
 
% Predict the validation
[C(epochs),pred,~] = Estimate_Class2(Weight1, Weight2, X_valid,R,Y_valid);
 
fprintf('\nEpoch %d, \tError: %.4f, \tError: %.4f', ...
    epochs,J(epochs),C(epochs));
 
if(epochs>1)
if(abs(C(epochs)-C(epochs-1))<=0.005)
    break;
end
end
 
 
end
 
figure, 
plot(J(1:epochs)); hold on; plot(C(1:epochs));
xlabel('Epochs');ylabel('Cross Entropy');
legend('Training Within Epoch(s)', 'Validation After Epoch(s)')
 
 
%%
[embedX]=nlin_dim(R);
figure,
scatter(embedX(:,1),embedX(:,2))
text(embedX(:,1)+0.2,embedX(:,2)+0.2, cellstr(words'))
 
%%
T = randperm(46500);
Sample_Trigrams = X_test(T(1:5),:);
Output_Trigrams = Y_test(T(1:5),:);
 
 
for i=1:5
    disp(words(Sample_Trigrams(i,:)))
    [~,pred_test,H] = Estimate_Class2(Weight1, Weight2, Sample_Trigrams(i,:),...
        R,Output_Trigrams);
    [P,Ind] = sort(H);
    disp(words(Ind(1:10)))
end
 
 
 
%% Functions
function Weights = WeightInitialization(Limit_in, Limit_out,e_init)
 
Weights = ones(Limit_out, 1 + Limit_in) + e_init;
 
end
 
function g = sigmoidGradient(z)
g = sigmoid(z).*(1-sigmoid(z));
end
 
function g = sigmoid(z)
g = 1  ./ (1+exp(-z));
end
 
 
function [J, Weight1,Weight2] = Gradient_Cost(Weight1,Weight2,...
                                   num_labels, X, y,neta,smpl,alpha)
 
 
% Length of Data
m = size(X, 1);
         
% Cost function and Weight Gradients 
J = 0;
JJ = 0;
Weight1_grad = zeros(size(Weight1));
Weight2_grad = zeros(size(Weight2));
 
mm = smpl;          % Mini Batch
ind = randperm(m);
X = X(ind,:);
y = y(ind);
 
y=y';
 
count = 0;
% Loop to calculate Old weights and 
for i=1:m
    
        % Feed forward
        % Input layer and hidden layer
        a1 = [1 X(i,:)];
        z2 = Weight1*a1';
        a2 = tanh(z2);
        a2 = a2';
        a2 = [1 a2];
        
        % Hidden layer and output layer
        z3 = Weight2*a2';
        a3 = tanh(z3);
        hx = a3;
 
        % True label
        yk = y(i);
        
        % Back propagation
        delta3_grad = 1/2*(hx - yk);
        
        % going backward
        delta2_grad = Weight2'*delta3_grad;
        delta2_grad = delta2_grad(2:end).*(1-tanh(z2).^2);
        
        % Update the weight gradients
        Weight2_grad = Weight2_grad + delta3_grad*a2;
        Weight1_grad = Weight1_grad + delta2_grad*a1;
        
        % Calculate the cost
        JJ = JJ + (yk - (hx)).^2;
        
        % Update the weights in case of stochastic or minibatch
        if(rem(i,mm)==0)
        J = J + JJ/mm;% + sum(sum(Weight2(:,2:end).^2))/(2*mm);
        count = count + 1;
        JJ = 0;
        
        Weight11 = Weight1;
        Weight11(:,1) = 0;
 
        Weight22 = Weight2;
        Weight22(:,1) = 0;
 
        % Normalization and regularization
        Weight1_grad = Weight1_grad/mm;
        Weight2_grad = Weight2_grad/mm;
 
        Weight1 = alpha*Weight1 - neta*Weight1_grad;
        Weight2 = alpha*Weight2 - neta*Weight2_grad;
        JJ=0;
        Weight1_grad = zeros(size(Weight1));
        Weight2_grad = zeros(size(Weight2));
        
 
        end
        
        % Update the weights in case of stochastic or minibatch
                    
        
end
 
J = J / count;
end
 
 
function [ output_class ] = Estimate_Class( Weight1, Weight2, X )
% Length of Data
m = size(X, 1);
         
output_class = X(:,1).*0;
% Loop to calculate Old weights and 
for i=1:m
    
        a1 = [1 X(i,:)];
        z2 = Weight1*a1';
        a2 = tanh(z2);
        a2 = a2';
        a2 = [1 a2];
        
        z3 = Weight2*a2';
        a3 = tanh(z3);
        hx = a3';
 
        output_class(i) = (hx);
end
 
 
end
 
 
function [ output_class ] = Estimate_Class_2layers( Weight1, Weight2,Weight3, X )
% Length of Data
m = size(X, 1);
         
output_class = X(:,1).*0;
% Loop to calculate Old weights and 
for i=1:m
    
        a1 = [1 X(i,:)];
        z2 = Weight1*a1';
        a2 = tanh(z2);
        a2 = a2';
        a2 = [1 a2];
       
        z3 = Weight2*a2';
        a3 = tanh(z3);
        a3 = a3';
        a3 = [1 a3];
       
        z4 = Weight3*a3';
        a4 = tanh(z4);
        hx = a4;
 
        output_class(i) = hx;
end
 
 
end
 
 
 
function [J, Weight1,Weight2,Weight3] = Gradient_Cost_2layers(Weight1,...
                                        Weight2,Weight3,...
                                   num_labels, X, y,neta,smpl,alpha)
% Length of Data
m = size(X, 1);
         
% Cost function and Weight Gradients 
J = 0;
JJ = 0;
JJ1 = 0;
JJ2 = 0;
JJ3 = 0;
Weight1_grad = zeros(size(Weight1));
Weight2_grad = zeros(size(Weight2));
Weight3_grad = zeros(size(Weight3));
 
mm = smpl;
ind = randperm(m);
X = X(ind,:);
y = y(ind);
y=y';
 
count = 0;
% Loop to calculate Old weights and 
for i=1:m
    
 
        a1 = [1 X(i,:)];
        z2 = Weight1*a1';
        a2 = tanh(z2);
        a2 = a2';
        a2 = [1 a2];
       
        z3 = Weight2*a2';
        a3 = tanh(z3);
        a3 = a3';
        a3 = [1 a3];
       
        z4 = Weight3*a3';
        a4 = tanh(z4);
        hx = a4;
 
        yk = y(i);
     
        delta4_grad = 1/2*(hx - yk);
           
        delta3_grad = Weight3'*delta4_grad;
        delta3_grad = delta3_grad(2:end).*(1-tanh(z3).^2);
        
        delta2_grad = Weight2'*delta3_grad;
        delta2_grad = delta2_grad(2:end).*(1-tanh(z2).^2);  
        
        Weight3_grad = Weight3_grad + delta4_grad*a3;
        Weight2_grad = Weight2_grad + delta3_grad*a2;
        Weight1_grad = Weight1_grad + delta2_grad*a1;
        
        
        JJ = JJ + (yk - (hx)).^2;
 
        if((rem(i,mm)==0))
        J = J + JJ/mm;
        count = count + 1;
        JJ = 0;
        
        Weight11 =Weight1;
        Weight11(:,1) = 0;
 
        Weight22 =Weight2;
        Weight22(:,1) = 0;
 
        Weight33 =Weight3;
        Weight33(:,1) = 0;
 
        Weight1_grad = Weight1_grad/mm;
 
        Weight2_grad = Weight2_grad/mm;
 
        Weight3_grad = Weight3_grad/mm;
        
       
 
        Weight1 = Weight1 - neta*(Weight1_grad + alpha.*JJ1);
        Weight2 = Weight2 - neta*(Weight2_grad + alpha.*JJ2);
        Weight3 = Weight3 - neta*(Weight3_grad + alpha.*JJ3);
        
        JJ=0;
        JJ1 = Weight1_grad;
        JJ2 = Weight2_grad;
        JJ3 = Weight3_grad;
        
        Weight1_grad = zeros(size(Weight1));
        Weight2_grad = zeros(size(Weight2));
        Weight3_grad = zeros(size(Weight3));
        
 
        end
        
        
end
 
J = J / count;
end
 
 
 
function [J, grad] = CEcost(W,Layer1_Size,Layer2_Size,...
                                   num_labels, X, y,lambda)
% Cost Calculations                            
                               
                               
Weight1 = reshape(W(1:Layer2_Size * (Layer1_Size + 1)), ...
                 Layer2_Size, (Layer1_Size + 1));
 
Weight2 = reshape(W((1 + (Layer2_Size * (Layer1_Size + 1))):end), ...
                 num_labels, (Layer2_Size + 1));
 
% Length of Data
m = size(X, 1);
         
% Cost function and Weight Gradients 
J = 0;
JJ = 0;
Weight1_grad = zeros(size(Weight1));
Weight2_grad = zeros(size(Weight2));
 
y=y';
% Loop to calculate Old weights and 
for i=1:m
    
        % Feed forward
        % Input layer and hidden layer
        a1 = [1 X(i,:)];
        z2 = Weight1*a1';
        a2 = sigmoid(z2);
        a2 = a2';
        a2 = [1 a2];
        
        % Hidden layer and output layer
        z3 = Weight2*a2';
        a3 = sigmoid(z3);
        hx = a3;
 
        % True label
        yk = zeros(num_labels,1);
        yk(y(i))=1;
     
        % Back propagation
        delta3_grad = (hx - yk);
        
        % going backward
        delta2_grad = Weight2'*delta3_grad;
        delta2_grad = delta2_grad(2:end).*sigmoidGradient(z2);
        
        % Update the weight gradients
        Weight2_grad = Weight2_grad + delta3_grad*a2;
        Weight1_grad = Weight1_grad + delta2_grad*a1;
        
        % Calculate the cost
        JJ = JJ + (-log(hx')*yk-log(1-hx')*(1-yk));
                
end
 
        J = JJ/m + lambda*(sum(sum(Weight1(:,2:end).^2))...
        +sum(sum(Weight2(:,2:end).^2)))/(2*m);
 
        Weight11 =Weight1;
        Weight11(:,1) = 0;
 
        Weight22 =Weight2;
        Weight22(:,1) = 0;
 
        % Normalization and regularization
        Weight1_grad = Weight1_grad/m + (lambda*Weight11/m);
        Weight2_grad = Weight2_grad/m + (lambda*Weight22/m);
 
        grad = [Weight1_grad(:) ; Weight2_grad(:)];
 
end
 
 
 
function [R,J, Weight1,Weight2] = Gradient_Cost2(Weight1,Weight2,...
                                   X, y,neta,smpl,alpha,L_R,R,D)
% Calculates the gradient descend on batch algo and Cost function
 
% type=1 batch, 2 stochastic, 3 mini batch
% smpl is the length of mini batch
 
% Length of Data
m = size(X, 1);
         
% Cost function and Weight Gradients 
J = 0;
JJ = 0;
 
G_Weight1_grad = zeros(size(Weight1));
G_Weight2_grad = zeros(size(Weight2));
G_D_R = zeros(L_R,D);
 
Weight1_grad = zeros(size(Weight1));
Weight2_grad = zeros(size(Weight2));
D_R = zeros(L_R,D);
 
 
mm = smpl;          % Mini Batch
ind = randperm(m);
X = X(ind,:);
y = y(ind);
 
% y=y';
 
count = 0;
% Loop to calculate Old weights and 
for i=1:m
    
        X2 = X(i,:);
        
        X_1 = R(X2(1),:);
        X_2 = R(X2(2),:);
        X_3 = R(X2(3),:);
        
        XX = [X_1 X_2 X_3];
        
        % Feed forward
        % Input layer and hidden layer
        a1 = [1 XX];
        z2 = Weight1*a1';
        a2 = sigmoid(z2);
        a2 = a2';
        a2 = [1 a2];
        
        % Hidden layer and output layer
        z3 = Weight2*a2';
        a3 = softmax(z3);
        hx = a3;
 
        % True label
        yk = zeros(L_R,1);
        yk(y(i)) = 1;
        
        % Back propagation
        delta3_grad = (hx-yk);
        
        % going backward
        delta2_grad = Weight2'*delta3_grad;
        delta2_grad = delta2_grad(2:end).*sigmoidGradient(z2);
        
        
        % Update the weight gradients
        Weight2_grad = Weight2_grad + delta3_grad*a2;
        Weight1_grad = Weight1_grad + delta2_grad*a1;
        
        % Calculate the cost
        JJ = JJ - sum(yk.*log(hx));
        
        A1 = zeros(250,1);
        A2 = zeros(250,1);
        A3 = zeros(250,1);
        
        A1(X2(1)) = 1;
        A2(X2(2)) = 1;
        A3(X2(3)) = 1;
 
        D_R = D_R + A1*(Weight1(:,1+1:D+1)'*delta2_grad)'; 
        D_R = D_R + A2*(Weight1(:,D+1+1:2*D+1)'*delta2_grad)';
        D_R = D_R + A3*(Weight1(:,2*D+1+1:3*D+1)'*delta2_grad)';
        
        % Update the weights in case of stochastic or minibatch
        if(rem(i,mm)==0)
        J = J + JJ/mm;% + sum(sum(Weight2(:,2:end).^2))/(2*mm);
        count = count + 1;
        JJ = 0;
 
        % Normalization and regularization
        Weight1_grad = Weight1_grad/mm;
        Weight2_grad = Weight2_grad/mm;
        D_R = D_R/mm;
        
        G_Weight1_grad = alpha*G_Weight1_grad + Weight1_grad;
        G_Weight2_grad = alpha*G_Weight2_grad + Weight2_grad;
        G_D_R = alpha*G_D_R + D_R;
 
        Weight1 = Weight1 - neta*G_Weight1_grad;
        Weight2 = Weight2 - neta*G_Weight2_grad;
        R = R - neta*G_D_R;
 
        
        
        JJ=0;
        Weight1_grad = zeros(size(Weight1));
        Weight2_grad = zeros(size(Weight2));
        D_R = zeros(L_R,D);
 
 
        end
        
        % Update the weights in case of stochastic or minibatch
                    
        
end
 
J = J / count;
end
 
 
function [ JJ,output_class,hx ] = Estimate_Class2( Weight1, Weight2, X, R,Y )
%ESTIMATE_CLASS Summary of this function goes here
%   Detailed explanation goes here
 
% Length of Data
m = size(X, 1);
 
% xk = zeros(1,250); 
output_class = X(:,1).*0;
 
JJ = 0;
% Loop to calculate Old weights and 
for i=1:m
    
        X2 = X(i,:);
        
        yk = zeros(250,1); 
        yk(Y(i)) = 1;
        
        X_1 = R(X2(1),:);
        X_2 = R(X2(2),:);
        X_3 = R(X2(3),:);
        
        XX = [X_1 X_2 X_3];
        
        a1 = [1 XX];
        z2 = Weight1*a1';
        a2 = sigmoid(z2);
        a2 = a2';
        a2 = [1 a2];
        
        z3 = Weight2*a2';
        a3 = softmax(z3);
        hx = a3;
 
        [~,output_class(i)] = max(hx);
%         xk(output_class(i)) = 1;
        
        JJ = JJ - sum(yk.*log(hx));
        
end
 
JJ = JJ/m;
 
end
 
 


