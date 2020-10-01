function Y = greedy_nonsym_dpp_localsearch(B, C, num_to_choose, num_iterations)
% The greedy local search to approximate MAP infernce of low-rank non-symmetric 
% DPP, that is, 
%
%   argmax_{S} det(L(S,S) such that |S| <= num_to_choose
%
% where L = B * C * B'. This begins with the standard greedy algorithm and 
% locally searches an item i not in the greedy output then swaps it to single 
% item in the greedy output whose improvement is maximized.
% See Kathuria, Tarun, and Amit Deshpande. "On sampling and greedy map inference 
% of constrained determinantal point processes." arXiv 2016.
% Inputs:
%   B: M-by-K matrix
%   C: K-by-K matrix 
%   num_to_choose: integer that should be less than or equal to K
%   num_iterations(optional): number of local search steps. Default is set to
%     num_to_choose^2 * log(10 * num_to_choose)
% Output:
%   chosen_set: subset of {1, ..., M} with size num_to_choose

if size(C,1) ~= size(C,2)
  error('C should be a square matrix');
end
if size(B,2) ~= size(C,1)
  error('Number of rows in B and C should be equal');
end

[N, num_rank] = size(B);
if num_to_choose > num_rank
  error('num_to_choose should be less than or equal to the rank of B');
end

if nargin <= 3
  % If num_iterations is not given, we set the number of search steps to 
  % O(num_to_choose^2 * log(num_to_choose/epsilon)).
  epsilon = 0.1;
  num_iterations = floor(num_to_choose^2 * log(num_to_choose / epsilon));
end

% Initialize the output subset by the greedy algorithm.
Y = greedy_nonsym_dpp(B, C, num_to_choose);
det_val = compute_det_submatrix_(B, C, Y);

P = B * C;
Q = B;

for step = 1 : num_iterations
  % Each iteration finds (i,j) such that i \in Y and j \notin Y and
  % swaps Y <- Y \{i} \cup {j} gives the maximum improvement.
    
  marginals = zeros(num_to_choose, N - num_to_choose);
  for i = 1 : num_to_choose
    
    % Pick a single element in Y.
    a = Y(i);

    % Compute the DPP kernel conditioned Y\{i}
    Y_minus_a = setdiff(Y, a);
    
    marginal_gains = greedy_conditioned_(P, Q, Y_minus_a);
    pos_Y_minus_a = setdiff(1:N, Y_minus_a);
    marginal_gains = marginal_gains(pos_Y_minus_a);
    marginal_gains_minus_a = marginal_gains(pos_Y_minus_a ~= a);
    
    det_Y_minus_a = det(P(Y_minus_a,:) * Q(Y_minus_a,:)');
    
    % Multiply marginal gains by det(L(Y\{i}, Y\{i})) to obtain 
    % det(L \cup {j} \ {a}) for j \notin Y.
    marginals(i, :) = marginal_gains_minus_a' * det_Y_minus_a;
  end
  Y_complement = setdiff(1:N, Y);
  if max(marginals(:)) < det_val
    return;
  end
    
  % Extract imax from Y and add jmax to Y.
  [imax, jmax] = find(marginals == max(marginals(:)));
  Y = [setdiff(Y, Y(imax)) Y_complement(jmax)];
  assert(length(unique(Y)) == num_to_choose);
  det_val = max(marginals(:));
end
end

function det_L_submatrix = compute_det_submatrix_(B, C, subset)
B_subset = B(subset, :);
det_L_submatrix = det(B_subset * C * B_subset');
end

function conditioned_prob = greedy_conditioned_(P, Q, conditioned_set)
marginal_gain = sum(P.*Q, 2);
j = 1;
a = conditioned_set(j);
j = j + 1;
C1 = [];
C2 = [];
for i = 2 : length(conditioned_set) + 1

  e1 = Q * P(a,:)';
  e2 = P * Q(a,:)';
  if i > 2
    e1 = e1 - C1 * C2(a,:)';
    e2 = e2 - C2 * C1(a,:)';  
  end
  e1 = e1 / marginal_gain(a);
  C1 = [C1 e1];
  C2 = [C2 e2];
    
  marginal_gain = marginal_gain - (e1 .* e2);
  if j > length(conditioned_set)
    break
  end
  a = conditioned_set(j);
  j = j + 1;
end
conditioned_prob = marginal_gain;
end