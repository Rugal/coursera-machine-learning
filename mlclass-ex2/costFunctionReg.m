function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%X = [ones(20,1) (exp(1) * sin(1:1:20))' (exp(0.5) * cos(1:1:20))'];
%y = sin(X(:,1) + X(:,2)) > 0
% costFunctionReg([0.25 0.5 -0.5]', X, y, 0.1)


h = sigmoid(X*theta);

J = (-y'* log(h) - (1 - y)'* log(1-h) + lambda / 2 * (sum(theta.^2) - theta(1)^2 ) )/m;

grad = (X'*(h - y) + lambda*theta)/m;
grad(1)= X(:,1)' * (h-y) / m;
% =============================================================

end
