function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a_1 = X;
a_1 = [ones(m,1), a_1];
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m,1), a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
% y_temp = zeros(m,num_labels);

% for i= 1:m
	% y_temp(i,y(i)) =  1;
% end

% J = (1/m)*(-(y_temp')*log(a_3)-(1-y_temp')*log(1-a_3));

for i=1:m
	for k=1:num_labels
		if y(i,1) == k
			J = J + (1/m)*(-log(a_3(i,k)));
		else 
			J = J + (1/m)*(-log(1-a_3(i,k)));
		end
	end
end	
temp = 0;
[row, col] = size(Theta1);
for j=1:row
	for k=2:col
		temp =  temp + Theta1(j, k)*Theta1(j, k);
	end
end
[row, col] = size(Theta2);
for j=1:row
	for k=2:col
		temp = temp + Theta2(j, k)*Theta2(j, k);
	end
end
J = J + (lambda/(2*m))*(temp);
D2 = 0;
D1 = 0;
for t = 1:m
	a1 = X(t,:)';
	a1 = [1 ; a1];
	z2 = Theta1*a1;
	a2 = sigmoid(z2);
	a2 = [1 ; a2];
	z3 = Theta2*a2;
	a3 = sigmoid(z3);
	d3 = zeros(size(a3));
	for k=1:num_labels
		if y(t,1) == k
			d3(k,1) = a3(k,1) - 1;
		else
			d3(k,1) = a3(k,1);
		end
	end
	zd = sigmoidGradient(z2);
	zd = [0 ; zd];
	d2 = Theta2'*d3.*zd;
	d2 = d2(2:length(d2),:);
	D2 = D2 + d3*a2';
	D1 = D1 + d2*a1';
end
Theta2_grad = (1/m)*D2 + (lambda/m)*Theta2;
Theta1_grad = (1/m)*D1 + (lambda/m)*Theta1;
[row, col] = size(Theta2);
for i = 1:row
	Theta2_grad(i,1) = Theta2_grad(i,1) - (lambda/m)*Theta2(i,1);
	
end
[row, col] = size(Theta1);
for i= 1:row
	Theta1_grad(i,1) = Theta1_grad(i,1) - (lambda/m)*Theta1(i,1);
end
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
