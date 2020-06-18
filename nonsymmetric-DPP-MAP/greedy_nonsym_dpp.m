function chosen_set = greedy_nonsym_dpp(B, C, num_to_choose)
% The greedy MAP inference algorithm for nonsymmetric DPP. This approximatly 
% finds that
% 
%   argmax_{S} det(L(S,S) such that |S| = num_to_choose
%
% where L = B * C * B'. Note that the matrix L should be P_0 matrix whose
% all principal minors should be nonnegative. To ensure this, one can consider 
% C to be the form of eye(K) + D - D' where eye(K) is the identity matrix with 
% size K and D is any K-by-K matrix.
% See Algorithm 1 in https://arxiv.org/pdf/2006.09862.pdf.
%
% Inputs:
%   B: M-by-K matrix
%   C: K-by-K matrix 
%   num_to_choose: integer that should be less than or equal to K
% Output:
%   chosen_set: subset of {1, ..., M} with size num_to_choose

if size(C,1) ~= size(C,2)
  error('C should be a square matrix');
end
if size(B,2) ~= size(C,1)
  error('Number of rows in B and C should be equal');
end

num_rank = size(B, 2);
if num_to_choose > num_rank
  error('num_to_choose should be less than or equal to the rank of B');
end

% Compute the diagonal of the kernel matrix L = B * C * B';
marginal_gain = sum(B .* (B * C'), 2);

% Find an item that maximizes the marginal gain and add it to the output subset.
[max_marginal_gain, item_argmax] = max(marginal_gain );
chosen_set = item_argmax;

% Initialize the matrices for updating the marginal gain.
P = []; Q = [];

for i = 1 : num_to_choose
  % Compute vectors for updating the marginal gain and store them.
  b_argmax = B(item_argmax,:);
  if length(chosen_set) == 1
    p = b_argmax / max_marginal_gain;
    q = b_argmax;
  else
    p = (b_argmax - ((b_argmax * C') * Q') * P) / max_marginal_gain;
    q = b_argmax - ((b_argmax * C) * P') * Q;
  end
  P = [P; p];
  Q = [Q; q];
  
  % Update the marginal gains.
  marginal_gain = marginal_gain - ((B * C * p') .* (B * C' * q'));
  if length(chosen_set) >= num_to_choose
    break;
  end
  
  % Find a next item that maximizes the marginal gain and add it to the
  % output subset.
  [max_marginal_gain, item_argmax] = max(marginal_gain);
  chosen_set(end+1) = item_argmax;
end

end