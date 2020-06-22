function chosen_set = mcmc_sampling_nonsym_dpp(B, C, num_to_choose, num_iterations)
% Markov-chain Monte-Carlo sampling of DPP.
% Inputs:
%   B: M-by-K matrix
%   C: K-by-K matrix 
%   num_to_choose: integer that should be less than or equal to K
%   num_iterations(optional): number of transitions of swapping items.
%     Default is set to num_to_choose^2 * log(num_to_choose * 10)
% Output:
%   chosen_set: subset of {1, ..., M} with size num_to_choose

P = B * C;
Q = B;

N = size(P, 1);
if nargin <= 3
  epsilon = 0.1;
  num_iterations = floor(num_to_choose^2 * log(num_to_choose / epsilon));
end

% Initialize a random subset with size num_to_choose.
rand_perm_N = randperm(N);
chosen_set = rand_perm_N(1:num_to_choose);
C_comple = rand_perm_N(num_to_choose+1:end);

for num_to_choose = 1 : num_iterations
   
  pos_i = randi(length(chosen_set), 1);
  pos_j = randi(length(C_comple), 1);
  
  C_minus_ii_plus_jj = chosen_set;
  C_minus_ii_plus_jj(pos_i) = C_comple(pos_j);
  
  % Swap i and j with probability min(1, det(L(C',C')) / det(L(C,C)) 
  % where C' = C u {i} \ {j}.
  prob = det(submatrix(P, Q, C_minus_ii_plus_jj)) / det(submatrix(P, Q, chosen_set));
  if isnan(prob)
    prob = 1;
  end
  prob = min(1, prob);
  
  if rand < prob
    C_comple(pos_j) = chosen_set(pos_i);
    chosen_set = C_minus_ii_plus_jj;
  end
end
chosen_set = sort(chosen_set);
end

function L_submatrix = submatrix(P, Q, set)
L_submatrix = P(set,:) * Q(set,:)';
end
