# Regression-MATLAB

## Gaussian Basis Linear Regression

Linear basis function regression with Gaussian basis functions were used to classify the fuel efficiency
of 392 cars given 7 features describing the car. The model for linear basis function is as follows, where
∅(𝑥) is the matrix of gaussian basis functions on 𝑥. The following was performed for several basis
functions in the range of 5:10:95.

                          ![alt text](https://raw.githubusercontent.com/somogysm/Regression-MATLAB/master/imgs/1.PNG)

The basis functions for the training data are used to find the coefficients 𝑤.

                                           𝑤 = (∅𝑇 ∅)−1 ∅𝑇 𝑦𝑡𝑟𝑎𝑖𝑛
The coefficients are then used to calculate the predicted response variables using the above. The
training/testing error can be plotted against the number of basis functions used. In the plot, it is seen
that as the number of basis functions increased the testing error increases. The opposite in shown for
the training error, as it continuously decreases. This is most likely occurring due to over fitting, and thus
regularization is implemented.


## Lasso Regularization
Regularization is much like the above process, however the solution for the coefficients are changed,
where 𝜆 is the regularization coefficient. Regularization was performed for 𝜆 =
[0.01 0.1 0 1 10 100 1000]

                                        𝑤 = (𝜆𝐼 + ∅𝑇 ∅)−1 ∅𝑇 𝑦𝑡𝑟𝑎𝑖𝑛
10-fold cross-validation was used to decide the best lambda for regularization. A plot of log(𝜆) vs the
cross-validation error is below. As 𝜆 increases to 1000, the cross-validation error increases. Also, for a
𝜆=0, the regression simplifies to the non-regularization case. The optimal regularization coefficient for
this data set seems to be 0.001, having the smallest CV error.
2.

A ML estimation regularized with a Lasso regularizer was used to classify data consisting of 91 variables
regarding songs from the time frame between 1922 and 2011. A function was created with the input of
the X training data variable, and Y response data, as well as a regularizer coefficient Lambda. The
function outputs the desired ML estimation coefficients. A list of 100 Lambda values were created
between 0.001 and 1000. The function was used for each Lambda, and plots regarding the log(lambda)
VS training/test error were produced. As well as a plot of Lambda VS the number of non-zero
coefficients. The training error began to increase dramatically for lambdas greater than 0.1, meanwhile
the testing error was optimal for lambda in the range of 0.5. As lambda increased above 1, error reached
a maximum. The number of non-zero coefficients were highest for the smallest values of Lambda,
however has Lambda approached 1, the number of zero coefficients increased. For Lambda’s above 1,
majority of the coefficients were zero.

## Logistic Regression
Logistic regression was implemented in order to classify the species of crab given in the Australian
Crab dataset. The crab dataset included 8 columns of data with 200 samples. The columns included
were; Species, Sex, Index, Frontal Lobe Size, Rear Width of the Shell, Carapace Length, Carapace Width,
and Body Depth. The first 150 samples were used for training data, while the last 50 samples were used
for testing data. The error function for logistic regression is as follows, along with its gradient. Where h
is the sigmoid of x and theta.
                                          𝑚
                                     1
                                𝐽 = − ∑ 𝑦𝑙𝑜𝑔(ℎ) + (1 − 𝑦)log(1 − ℎ)
                                     𝑚
                                           1
                                                      𝑚
                                                1
                                          ∇𝐽 = − ∑(ℎ − 𝑦)𝑥
                                                𝑚
                                                      1

Minimizing the above using gradient descent, or MATLABs fminunc function. The appropriate
coefficients may be found. The classes can then me calculated using the following equation.
                                                         1
                                       ypred =
                                                 (1 + exp(−𝑋𝑡𝑒𝑠𝑡 𝜃))
The resulting confusion matrix generated shows that the logistic regression model classified correctly 26
species in class 1, and 23 species into class 2. However, one sample was misclassified into species 1,
while it truly belonged to species 2.
                                                         26 0
                                     𝑐𝑜𝑛𝑓𝑢𝑠𝑖𝑜𝑛 𝑚𝑎𝑡𝑟𝑖𝑥 = [      ]
                                                          1 23


b) The same process was performed using MATLABs built-in SVM function instead of logistic regression.
The confusion matrix classified 26 species as class 1, and 24 species as class 2. Resulting in zero
misclassifications.
                                                         26 0
                                     𝑐𝑜𝑛𝑓𝑢𝑠𝑖𝑜𝑛 𝑚𝑎𝑡𝑟𝑖𝑥 = [      ]
                                                          0 24
