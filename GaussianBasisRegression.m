function [] = GaussianBasisRegression();
clear all; close all;

filename = 'auto-mpg.data';
formatSpec = '%4f%4f%8f%11f%11f%10f%5f%4f%s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string',  'ReturnOnError', false);
dataArray{9} = strtrim(dataArray{9});
fclose(fileID);

for i = 1:9
    V(:, i) = dataArray{:, i};
end

x = V(:,2:8);
x = reshape(zscore(x(:)),size(x,1),size(x,2));%normalize the data
y = (V(:,1));

sig = 2;

N = 100; %Testing data length

n = 5:10:95; %number of basis functions
l=1;
for n = n %1a Linear basis regression for gaussian (NOT REGULARIZED)

mu = randi([1 100],1,n); %random basis functions

Sig = 2*eye(7);
phit = zeros(N,n);
phi = zeros(length(V)-N,n);
for j = 1:n %Use training data to find the coefficients (1:100)
    for i = 1:N
        phit (i,j) = exp(-((x(i,:) - x(mu(j),:))*(inv(Sig))*(x(i,:) - x(mu(j),:))'));
    end
end

W = pinv(phit)*y(1:N);

for i = (N+1):length(V) %calculate phi for testing data (101:392)
    for j = 1:n
        phi (i-N,j) = exp(-((x(i,:) - x(mu(j),:))*(inv(Sig))*(x(i,:) - x(mu(j),:))'));
    end
end

yp =W'*phi' ; 
yt = W'*phit';

errorTrain(l) = sqrt(mean((y(1:100) - yt').^2));
errorTest(l) = sqrt(mean((y(101:end) - yp').^2));
l=l+1;

end
figure(1);
plot(5:10:95,errorTrain,5:10:95,errorTest,'r')
title('Number of Basis Functions VS the Training/Testing Error')
xlabel('Number of Basis Functions')
ylabel('Training/Testing Error')
legend('Training Error','Testing Error')


n = 90; %number of basis functions
lambda = [0.01,0.1,1,10,100,1000];% lambda = 0 omitted, as regularization simplifies to the above when so.
for k = 1:length(lambda) %1b Linear basis regression for gaussian (REGULARIZED)
indices = crossvalind('Kfold',y,10);

mu = randi([1 100],1,n); %random basis functions
for l = 1:10
  test = (indices == l);
  train = ~test;
  xtest = x(test,:);
  xtrain = x(train,:);
  N=length(test);
  Sig = 2*eye(7);
  phit = zeros(length(xtrain),n);
  phi = zeros(length(xtest),n);
  for j = 1:n %Use training data to find the coefficients (1:100)
      for i = 1:length(xtrain)
        phit (i,j) = exp(-((xtrain(i,:) - x(mu(j),:))*(inv(Sig))*(xtrain(i,:) - x(mu(j),:))'));
      end
  end

  W = (lambda(k)*eye(n) + phit'*phit)\phit'*y(train);

  for i = 1:length(xtest) %calculate phi for testing data (101:392)
      for j = 1:n
          phi (i,j) = exp(-((xtest(i,:) - x(mu(j),:))*(inv(Sig))*(xtest(i,:) - x(mu(j),:))'));
      end
  end
  ypr =W'*phi'; 
  e(k,l) = sqrt(mean((y(test) - ypr').^2));%K
  
end
CV(k) = 1/10*sum(e(k,:));

end
figure(2);
semilogx(lambda,CV);
title('log(Lambda) VS Cross Validation Error')
xlabel('log(Lambda)');
ylabel('CV Error');
end
 