function y = dLogisticSigmoid(x)
% dLogisticSigmoid Derivative of the logistic sigmoid.
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the derivative of the logistic sigmoid was
% applied element by element.
%
    y = logisticSigmoid(x).*(1 - logisticSigmoid(x));
end