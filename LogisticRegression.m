function [] = LogisticRegression();
clear all;
data = load('crabdata.txt');
trainIn = 1:150;
testIn = 151:200;

X = [data(:,2:8)];
Y = normalize(data(:,1),'range',[0 1]);

[m, n] = size(X);
X = [ones(m, 1) X];
Xtrain = X(trainIn,:);
Ytrain = Y(trainIn,:);
Xtest = X(testIn,:); Ytest = Y(testIn,:);


initial_theta = zeros(n+1, 1);
[~, grad] = costFunction(initial_theta, Xtrain, Ytrain);

options = optimset('GradObj', 'on','MaxIter',150);

[theta, cost] = fminunc(@(t)(costFunction(t, Xtrain, Ytrain)), initial_theta, options);

class = round(1./(1+exp(-Xtest*theta)));

conf = confusionmat(Ytest, class);
display(conf);

SVMModel = fitcsvm(Xtrain,Ytrain,'KernelFunction','linear',...
    'Standardize',true);

[classSVM,score,costSVM] = predict(SVMModel,Xtest);

confSVM = confusionmat(Ytest, classSVM);
display(confSVM);

function [J, grad] = costFunction(theta, X, y)
m = length(y);

grad = zeros(size(theta));

h =  1./(1+exp(-(X*theta)));
J = -(1/m)*sum( (y.*log(h))+((1-y).*log(1-h)) );

for i = 1:size(theta,1)
    grad(i) = (1/m)*sum((h-y).*X(:,i));
end
end
end