classdef gandyn < handle
    %GANDYN Summary of this class goes here
    %   Detailed explanation goes here

     properties
         gen % nndyn for generating data
         disc % nndyn for classifying data as external or generated
         % N_gen % number of examples to generate per epoch
     end

     methods
         function obj = gandyn(widths_disc, T_disc)
             szw = size(widths_disc); % make inputs into column
             szT = size(T_disc);      % vectors if accidentally
                                      % given row vectors
             if length(widths_disc) == szw(2)
                 widths_disc = widths_disc';
             end
             if length(T_disc) == szT(2)
                 T_disc = T_disc';
             end
             
             widths_gen = flipud(widths_disc);
             T_gen = flipud(T_disc);
             
             obj.gen = nndyn(widths_gen, T_gen);
             obj.disc = nndyn(widths_disc, T_disc);
         end
     
         function [C_gen, C_disc] = train(obj, Xe, Ye, N_gen, method, a_gen, a_disc, b_gen, b_disc, max_i)
             % Input:
             %    Xe = external examples
             %    Ye = external example classifications
             %    N_gen = number of examples generated per batch
             %    method = type of optimization method
             %    a = learning rate; if 2 elements, second term is decay rate
             %    b = 2 elements: momentum & rms terms
             %    max_i = number of iterations
             % Output:
             %    C_gen = vector containing generator cost for each iteration
             %    C_disc = vecor containing discriminator cost for each iteration
             
             szXe = size(Xe);
             m = szXe(2);
             
             Ye_full = [Ye, zeros([1, m])];
             Xe_full = [Xe, rand(szXe)];
             
             C_gen = zeros([max_i, 1]);
             C_disc = C_gen;
             
             for i = 1:max_i
                 [C_gen(i), C_disc(i)] = obj.train_iteration(Xe_full, Ye_full, N_gen, method, a_gen, a_disc, b_gen, b_disc);
             end
         end
     
         function [C_gen, C_disc] = train_iteration(obj, Xe, Ye, N_gen, method, a_gen, a_disc, b_gen, b_disc)
             % Input:
             %    Xe = external examples
             %    Ye = external example classifications
             %    N_gen = number of examples generated per batch
             %    method = type of optimization method
             %    a = learning rate; if 2 elements, second term is decay rate
             %    b = 2 elements: momentum & rms terms
             % Output:
             %    C_gen = cost function of generator
             %    C_disc = cost function of discriminator
             
             Xh = obj.generate(N_gen);
             [C_gen, C_disc, X, Y] = obj.discriminate(Xh, Xe, Ye, N_gen);
             
             obj.disc.grad_desc(X, Y, method, a_disc, b_disc, 1);
             
             nn_gendisc = obj.gen.stack(obj.disc);
             nn_gendisc.lam = obj.gen.lam;
             N_gen_weights = length(obj.gen.weights);
             N_gen_biases = length(obj.gen.biases);
             i_disc_weights = N_gen_weights + 1:2*N_gen_weights;
             i_disc_biases = N_gen_biases + 1:2*N_gen_biases;
             
             X_gen = random('Normal', 0, 1, [obj.gen.widths(1), N_gen]);
             Y_gendisc = ones(obj.disc.widths(end), N_gen);
             
             nn_gendisc.grad_desc(X_gen, Y_gendisc, method, a_gen, b_gen, 1, i_disc_weights, i_disc_biases);
             obj.gen.weights = nn_gendisc.weights(1:N_gen_weights);
             obj.gen.biases = nn_gendisc.biases(1:N_gen_biases);
         end
     
         function [Xh] = generate(obj, N_gen)
             % Output:
             %    Xh = generated output; obj.N_gen examples
             
             R = random('Normal', 0, 1, [obj.gen.widths(1), N_gen]);
             Xh = obj.gen.fwdprop(R);
         end
     
         function [C_gen, C_disc, X, Y] = discriminate(obj, Xh, Xe, Ye, N_gen)
             % Input:
             %    Xh = generated examples
             %    Xe = external examples
             %    Ye = external example classifications
             % Output:
             %    C_gen = cost of generator for this iteration
             %    C_disc = cost of discriminator for this iteration
             %    X = entire training set - including both generated and external examples
             
             X = [Xh, Xe];
             
             % Y = [zeros(obj.disc.widths(end), N_gen), ones(obj.disc.widths(end), N_ext)];
             Y = [zeros([obj.disc.widths(end), N_gen]), Ye];
             szY = size(Y);
             m = szY(2);
             
             columnperm = randperm(m);
             X = X(:, columnperm);
             Y = Y(:, columnperm);
             
             Yh = obj.disc.fwdprop(X);
             
             L_disc = Y - Yh;
             C_disc = (1/m) * sum(sum(abs(L_disc))); % regularization term added later
             
             %L_gen = 
             C_gen = -C_disc;
         end
     end

end