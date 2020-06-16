# The greedy MAP inference of nonsymmetric DPP
MATLAB implementation of the greedy MAP inference of nonsymmetric DPP (Algorithm 1). See greedy_nonsym_dpp.m.
The algorithm approximatly finds that
 
  argmax_S det(L(S,S) such that |S|=k

where L = B * C * B^T is M-by-M matrix. Note that L is assumed to be P_0 matrix whose all principal minors should be nonnegative.
To ensure this, we can consider C = I + D - D^T where I is the identity matrix with size K and D is any K-by-K matrix (See Lemma 1 for details).
This algorithm runs in in O(MK^2) time. See Lemma 2 and Theorem 3 for the details.

## Usage
```console
M = 10;
K = 5;
k = 5;
B = randn(M, K);
C = randn(K, K);
C = eye(K) + C - C';
Y = greedy_nonsym_dpp(B, C, k);
```

## Reproduce

To reproduce results in Table 3, excute run_dppmapinference.m file. Due to limited file size, we do not provide pretrained models for 'Instacart'.

```console
run_dppmapinference('amazonretail')
run_dppmapinference('amazonthree')
run_dppmapinference('ukretail')
```
