function [] = LassoRegularization();
clear all; close all;
testStruct = importdata('musictestdata.txt',' ');
train = load('musicdata.txt');
test = testStruct.data; %convert struct into array
Xtrain = train(:,2:end); Ytrain = train(:,1);
Xtest = test(:,2:end); Ytest = test(:,1);

Lambda = [ 0.001 : 0.001: 0.009, 0.01 :0.01 :0.09, 0.1 :0.1: 0.9, 1:1:10, 11 :89/53 :99, 100 :100:1000]; %range between 0.001 and 1000
for i = 1:100
coef = lasso(Lambda(i),Xtrain,Ytrain);
predsTest = round(glmval(coef,Xtest,'identity'));%round because the year is a whole number.
predsTrain = round(glmval(coef,Xtrain,'identity'));%round because the year is a whole number.
errorTrain(i) = sqrt(mean((Ytrain - predsTrain).^2));
errorTest(i) = sqrt(mean((Ytest - predsTest).^2));
numZero(i) = sum(coef ~= 0);
end
figure(1);
semilogx(Lambda,errorTrain);
title('log(Lambda) VS Training Error')
xlabel('log(Lambda)');
ylabel('Training Error');
figure(2);
semilogx(Lambda,errorTest)
title('log(Lambda) VS Testing Error')
xlabel('log(Lambda)');
ylabel('Testing Error');
figure(3);
plot(Lambda,numZero)
title('Lambda VS Number of non-zero Coefficients')
xlabel('Lambda');
ylabel('Number of non-zero Coefficients');
figure(4);
semilogx(Lambda,numZero)
title('log(Lambda) VS Number of non-zero Coefficients')
xlabel('log(Lambda)');
ylabel('Number of non-zero Coefficients');
end

function [coef] = lasso(Lambda,Xtrain,Ytrain) %lasso function
[B, FitInfo] = lassoglm(Xtrain,Ytrain,'normal','CV',10,'lambda',Lambda);

idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
end
