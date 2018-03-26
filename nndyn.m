classdef nndyn < handle
    %NNDYN Neural network with configurable layers & weights 
    %   Detailed explanation goes here

     properties
         L % number of layers (not including input layer); = length(widths) - 1
         widths % column vector of layer widths (including input layer)
         weights % column vector of connection weights
         biases % column vector of biases
         T % column vector of transfer function ids (one per layer, not including input layer)
         lam % regularization parameter (L2)
         batch % size of the batch for mini-batch gradient descent
     end

     methods
         function obj = nndyn(widths, T)
             obj.widths = widths;
             obj.T = T;
             
             if length(T) ~= length(widths) - 1
                 error("widths must be one element longer than T")
             end
             
             obj.L = length(T);
             
           % Old weight initialization scheme:
%              widths_shift = circshift(widths, 1);
%              widths_shift(1) = 0;
%              nw = widths'*widths_shift;
%              obj.weights = (rand([nw, 1]) - (widths(1)^-0.5))*2*widths(1)^-0.5;
             
           % Xavier weight initialization:
             weight_offset = 0;
             for l = 1:obj.L
                 Wl_size = [obj.widths(l + 1), obj.widths(l)];
                 Wl_length = prod(Wl_size); Wl_end = weight_offset + Wl_length;
                 
                 isReLU = sign((obj.T(l) == 1) + (obj.T(l) == 2));
                 
                 obj.weights(weight_offset + 1:Wl_end) = ...
                     random('Normal', 0, sqrt((1 + isReLU)/obj.widths(l)), ...
                     [obj.widths(l), obj.widths(l + 1)]);
                 
                 weight_offset = Wl_end;
             end
             obj.weights = obj.weights';
             
             obj.biases = zeros([sum(widths(2:end)), 1]);

             obj.lam = 0;
             obj.batch = 0;
         end
         
         function [C] = grad_desc(obj, X, Y, type, a, b, max_i, ex_w, ex_b)
             % Inputs:
             %    X = matrix of column vectors of input training data
             %    Y = matrix of column vectors of output training data
             %    type = 'normal', 'momentum', 'rms', 'adam' (then, b is a 2-vector)
             %    a = learning rate; if real number, static learning rate
             %        if 2-vector, second element is decay rate
             %    b = moment parameter; 2-vector for 'adam', real number otherwise
             %    max_i = max iterations; when to stop training
             %    ex_w = column vector containing weights to exclude from grad. desc.
             %    ex_b = column vector containing biases to exclude from grad. desc.
             % Outputs:
             %    C = cost function/training accuracy for each iteration
             
             momentum_bool = 0;
             rms_bool = 0;
             adam_bool = 0;
             if strcmp(type, 'normal')
                 b = 0;
             elseif strcmp(type, 'momentum')
                 momentum_bool = 1;
             elseif strcmp(type, 'rms')
                 rms_bool = 1;
             elseif strcmp(type, 'adam')
                 adam_bool = 1;
             end
             
             decay_bool = 0;
             if length(size(a)) == 2
                 decay_bool = 1;
             end
             
             C = zeros([max_i, 1]);
             
             szX = size(X);
             m = szX(2);
             
             if obj.batch == 0
                 batches = 1;
             else
                 batches = ceil(m/obj.batch);
                 
                 columnperm = randperm(m);
                 X = X(:, columnperm);
                 Y = Y(:, columnperm);
             end
             
             if nargin == 7
                 in_w = 1:numel(obj.weights);
                 in_b = 1:numel(obj.biases);
             elseif nargin == 9
                 in_w = setdiff(1:numel(obj.weights), ex_w);
                 in_b = setdiff(1:numel(obj.biases), ex_b);
             end
             
             dweights_av = zeros(size(obj.weights));
             dbiases_av = zeros(size(obj.biases));
             dweights_rms = dweights_av;
             dbiases_rms = dbiases_av;
             
             i_tot = 0; % total iteration number
             for i = 1:max_i
                 for t = 1:batches
                     i_tot = i_tot + 1;
                     
                     if t == batches
                         X_batch = X(:, (t - 1)*obj.batch + 1:end);
                         Y_batch = Y(:, (t - 1)*obj.batch + 1:end);
                     else
                         X_batch = X(:, (t - 1)*obj.batch + 1:t*obj.batch);
                         Y_batch = Y(:, (t - 1)*obj.batch + 1:t*obj.batch);
                     end
                     
                     [Yh, values, activations] = obj.fwdprop(X_batch);
                     [dweights, dbiases, C(i)] = obj.backprop(X_batch, Y_batch, Yh, ...
                         values, activations);
                     
                     if decay_bool
                         a_i = a(1)/(1 + a(2)*(i - 1));
                     else
                         a_i = a;
                     end
                     
                     if momentum_bool
                         dweights_av = b*dweights_av + (1 - b)*dweights;
                         dbiases_av = b*dbiases_av + (1 - b)*dbiases;
                         
                         obj.weights(in_w) = obj.weights(in_w) - a_i*dweights_av(in_w);
                         obj.biases(in_b) = obj.biases(in_b) - a_i*dbiases_av(in_b);
                     elseif rms_bool
                         dweights_rms = b*dweights_av + (1 - b)*dweights.^2;
                         dbiases_rms = b*dbiases_av + (1 - b)*dbiases.^2;
                         
                         obj.weights(in_w) = obj.weights(in_w) - ...
                             a_i*dweights(in_w)./sqrt(dweights_rms(in_w) + 10^-8); % 10^-8 for numerical stability
                         obj.biases(in_b) = obj.biases(in_b) - ...
                             a_i*dbiases(in_b)./sqrt(dbiases_rms(in_b) + 10^-8);
                     elseif adam_bool
                         dweights_av = b(1)*dweights_av + (1 - b(1))*dweights;
                         dbiases_av = b(1)*dbiases_av + (1 - b(1))*dbiases;
                         
                         dweights_av_c = dweights_av/(1 - b(1)^i_tot);
                         dbiases_av_c = dbiases_av/(1 - b(1)^i_tot);
                         
                         dweights_rms = b(2)*dweights_rms + (1 - b(2))*dweights.^2;
                         dbiases_rms = b(2)*dbiases_rms + (1 - b(2))*dbiases.^2;
                         
                         dweights_rms_c = dweights_rms/(1 - b(2)^i_tot);
                         dbiases_rms_c = dbiases_rms/(1 - b(2)^i_tot);
                         
                         obj.weights(in_w) = obj.weights(in_w) - ...
                             a_i*dweights_av_c(in_w)./sqrt(dweights_rms_c(in_w) + 10^-8); % 10^-8 for numerical stability
                         obj.biases(in_b) = obj.biases(in_b) - ...
                             a_i*dbiases_av_c(in_b)./sqrt(dbiases_rms_c(in_b) + 10^-8);
                     else
                         obj.weights(in_w) = obj.weights(in_w) - a_i*dweights(in_w);
                         obj.biases(in_b) = obj.biases(in_b) - a_i*dbiases(in_b);
                     end
                     
                     if isnan(C(i))
                         break
                     end
                 end
             end
         end
         
         function [out, values, activations] = fwdprop(obj, X)
             % Inputs:
             %    X = matrix of column vectors of input values
             % Outputs:
             %    out = matrix of column vectors of output values
             % Other useful values:
             %    m = number of input & output value sets
             
             szX = size(X);
             m = szX(2);
             
             values = zeros([sum(obj.widths(2:end)), m]);
             activations = values;
             
             Al = X;
             
             node_offset = 0;
             weight_offset = 0;
             
             for l = 1:obj.L
                 Wl_size = [obj.widths(l + 1), obj.widths(l)];
                 Wl_length = prod(Wl_size); Wl_end = weight_offset + Wl_length;
                 Wl = reshape(obj.weights(weight_offset + 1:Wl_end), Wl_size);
                 weight_offset = Wl_end;
                 
                 %bl_length = obj.widths(l + 1); bl_end = bias_offset + bl_length;
                 node_end = node_offset + obj.widths(l + 1);
                 bl = obj.biases(node_offset + 1:node_end);
                 
                 Zl = Wl*Al + bl;
                 Al = transfer(Zl, obj.T(l), 0);
                 
                 values(node_offset + 1:node_end, :) = Zl;
                 activations(node_offset + 1:node_end, :) = Al;
                 node_offset = node_end;
             end
             
             out = Al;
         end
         
         function [dweights, dbiases, C] = backprop(obj, X, Y, Yh, values, activations)
             % Inputs:
             %    X = matrix of column vectors of input values
             %    Y = matrix of column vectors of target output values
             %    Yh = matrix of column vectors of output from last fwdprop
             %    activations = matrix containing all node activations from fwdprop
             %    values = matrix containing all node values from fwdprop
             %    values_c = matrix containing all batch normalized node values from fwdprop
             % Outputs:
             %    dweights = column vector same size as obj.weights containing
             %        change from this backprop iteration
             %    dbiases = column vector same size as obj.biases containing
             %        change from this backprop iteration
             % Other useful values:
             %    m = number of input & output value sets
             
             szX = size(X);
             m = szX(2);
             
             dweights = zeros(size(obj.weights));
             dbiases = zeros(size(obj.biases));
             
             dAl = Yh - Y; % simplified loss; seems to work
             %dAl = -Y./Yh + (1 - Y)./(1 - Yh); % for logistic output nodes
             C = (1/m) * sum(sum(abs(Y - Yh))); % L2 regularization term added in layer loop
             
             node_offset = sum(obj.widths(2:end - 1));
             weight_offset = numel(obj.weights) - obj.widths(end)*obj.widths(end - 1);
             
             for l = obj.L:-1:1
                 Wl_size = [obj.widths(l + 1), obj.widths(l)];
                 Wl_length = prod(Wl_size); Wl_end = weight_offset + Wl_length;
                 Wl = reshape(obj.weights(weight_offset + 1:Wl_end), Wl_size);
                 
                 node_end = node_offset + obj.widths(l + 1);
                 Zl = values(node_offset + 1:node_end, :);
                 
                 if l == 1
                     Alm1 = X;
                 else
                     Alm1 = activations(node_offset + 1 - obj.widths(l):node_offset, :);
                 end
                 
                 dZl = dAl .* transfer(Zl, obj.T(l), 1);
                 dWl = (1/m) * dZl*Alm1' + (obj.lam/m) * Wl;
                 dbl = (1/m) * sum(dZl, 2);
                 dAl = Wl'*dZl;
                 
                 dweights(weight_offset + 1:Wl_end) = reshape(dWl, [Wl_length, 1]);
                 dbiases(node_offset + 1:node_end) = dbl;
                 
                 if l == 1
                     weight_offset = 0;
                 else
                     weight_offset = weight_offset - obj.widths(l)*obj.widths(l - 1); 
                 end
                 node_offset = node_offset - obj.widths(l);
                 
                 C = C + (obj.lam/(2*m)) * norm(Wl, 'fro').^2;
             end
         end
     
         function nn_out = stack(obj, nn_in)
             % Inputs:
             %    nn_in = network to follow the network that calls this method
             %        nn_in.widths(1) must equal obj.widths(end)
             % Outputs:
             %    nn_out = network constructed by stacking obj and nn_in
             
             nn_out = nndyn([obj.widths; nn_in.widths(2:end)], [obj.T; nn_in.T]);
             nn_out.weights = [obj.weights; nn_in.weights];
             nn_out.biases = [obj.biases; nn_in.biases];
         end
     end

end