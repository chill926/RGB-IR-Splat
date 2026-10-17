# Sparse temperature interpolation

The fixed interpolation weights are now represented by a coalesced sparse matrix
and evaluated with sparse matrix multiplication. Zero-weight padding is omitted,
eliminating the concentrated scatter-add backward into control 0.

The model equations, parameterization, losses, hyperparameters, sample ordering,
budgets and checkpoint selection are unchanged. Floating-point summation order
changes; bitwise-identical trajectories are not promised.

The sparse matrix is a non-persistent derived buffer, rebuilt from the existing
indices/weights after construction and checkpoint loading. Old checkpoints remain
loadable, and the frozen-structure hash retains its original definition. Source
hashes in new experiment protocols distinguish this implementation; use a new
output prefix rather than mixing old and new runs under an existing lock.

The earlier memory-only benchmark on 2080 Ti measured 146.1 -> 24.9 ms per C
training step (20 samples after warmup, excluding validation and I/O). Check the
verification report for measurements of the actual updated implementation.
