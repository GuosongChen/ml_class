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

% ====================== There are two way ===================			   
%h=sigmoid(X*theta);
%Tr=theta;
%Tr(1)=0;

%J=(1/m)*(-y'*log(h)-(1-y)'*log(1-h))+lambda/(2*m)*Tr'*Tr;
%grad=(1/m)*(X'*(h-y)+lambda*Tr);

% ====================== 2nd one ===========================
Htheta=sigmoid(X*theta);
theta_J=theta;
theta_J(1)=0;
grad=1/m*(X'*(Htheta-y)+theta_J*lambda);

J_val=-y.*log(Htheta)-(1-y).*log(1-Htheta);
J=(1/m)*sum(J_val)+(lambda/(2*m))*sum(theta_J.^2);

% =============================================================
end
