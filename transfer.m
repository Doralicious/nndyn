function [ a ] = transfer( x, index, diff )
%TRANSFER Transfer function for neural networks
% Inputs:
%    x = input values (scalar, matrix, or vector)
%    index = which transfer function to use
%        0: tanh
%        1: ReLU
%        2: Leaky ReLU
%        3: sigmoid
%        4: Leaky tanh (experimental)
%        5: linear
%        6: linear piecewise (experimental)
%        7: random (gaussian, input controls std dev)
%    diff = whether to differentiate or not
%        0: do not differentiate
%        1: differentiate
% Outputs:
%    a = output values

if diff == 0
    if index == 0
        a = tanh(x);
    elseif index == 1
        a = max(0, x);
    elseif index == 2
        a = max(0.01*x, x);
    elseif index == 3
        a = 1./(1 + exp(-x));
    elseif index == 4
        a = transfer(x, 0, 0) + 0.1*x;
    elseif index == 5
        a = x;
    elseif index == 6
        a = min(max(0.1*x - 1, x), 0.1*x + 1);
    elseif index == 7
        a = random('Normal', 0, x);
    end
elseif diff == 1
    if index == 0
        a = 1 - tanh(x).^2;
    elseif index == 1
        a = (x >= 0);
    elseif index == 2
        a = (x >= 0) + 0.01*(x < 0);
    elseif index == 3
        a = exp(x)./((1 + exp(x)).^2);
    elseif index == 4
        a = transfer(x, 0, 1) + 0.1;
    elseif index == 5
        a = 1;
    elseif index == 6
        a = 0.1*((x < 0.1*x - 1) + (x > 0.1*x + 1)) + (x > 0.1*x - 1).*(x < 0.1*x + 1);
    elseif index == 7
        a = random('Normal', 0, x).*x*2;
    end
end

end