function chosen_set = greedy_nonsym_dpp_stochastic(B, C, num_to_choose, num_samples)
% The stochastic greedy MAP inference algorithm for nonsymmetric DPP. 
% See Mirzasoleiman, Baharan, et al. "Lazier than lazy greedy." AAAI 2015.
%
% Inputs:
%   B: M-by-K matrix
%   C: K-by-K matrix 
%   num_to_choose: integer that should be less than or equal to K
%   num_samples: number of random samples to compute the marginal gain.  
%     Default is set to floor(M/num_to_choose * log(10)) as proposed in the paper.
% Output:
%   chosen_set: subset of {1, ..., M} with size num_to_choose

M = size(B, 1);
if nargin <= 3
  % If num_samples is not given, set the number of samples to default.
  epsilon = 0.1;
  num_samples = floor((M / num_samples) * log(1 / epsilon));
end

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
[max_marginal_gain, item_argmax] = max(marginal_gain);
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

  % Choose a random subset to compute marginal gains.  If num_samples is 
  % greater than the size of ground-set, we set the random subset to the 
  % entire set.
  if M < num_samples
    samples = 1 : M;
  else
    samples = randsample(1 : M, num_samples, true);
  end

  % Find a next item in the random subset that maximizes the marginal gain 
  % and add it to the output subset.
  marginal_gain_subset = marginal_gain(samples);
  [max_marginal_gain, item_argmax_subset] = max(marginal_gain_subset);
  item_argmax = samples(item_argmax_subset);
  assert(max_marginal_gain == marginal_gain(item_argmax));
  chosen_set(end+1) = item_argmax;
end

end