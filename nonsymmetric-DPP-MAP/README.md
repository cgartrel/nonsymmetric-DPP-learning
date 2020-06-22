# The greedy MAP inference of nonsymmetric DPP
MATLAB implementation of the greedy MAP inference of nonsymmetric DPP.
The algorithm approximatly finds that
 
  argmax_S det(L(S,S)) such that |S|=k

where L = B * C * B^T, B is a M-by-K matrix and C is a K-by-K matrix (k <= K). 
Note that L is assumed to be P_0 matrix whose all principal minors should be nonnegative.
To ensure this, we can consider C = I + D - D^T where I is the identity matrix with size K and D is any K-by-K matrix.

## Usage

To test the greedy algorithm, run: 

```console
>> M = 10;
>> K = 5;
>> num_to_choose = 5;
>> B = randn(M, K);
>> C = randn(K, K);
>> C = eye(K) + C - C';
>> Y = greedy_nonsym_dpp(B, C, k);
```

To test the stochastic greedy algorith, run:

```console
>> num_samples = 4;
>> Y_stoch = greedy_stochastic_nonsym_dpp(B, C, k, num_samples);
```

