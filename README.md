# interp
A simple RBF interpolation utility considering only k nearest neighbors.
It uses the [nanoflann](https://github.com/jlblancoc/nanoflann) library (header only) for nearest neighbor search.
`nanoflann` comes included with the distribution so it is not technically a dependency.

## Requirements
- C++ Eigen library

## Build
    ./build.sh
## Run
    ===== Parameters ====
    numPoints: 200000
    numTargetPoints: 4
    numNeighbors: 8
    numFields: 1
    =====================
    Clustering point clouds into 2 clusters
    cluster 0 with centroid (0.497855 0.496606 0.749471) and 100103 points
    cluster 1 with centroid (0.501046 0.504052 0.250395) and 99897 points
    Finished in 44 millisecs.
    Constructing interpolation matrix ...
    Constructing interpolation matrix ...
    Finished in 143 millisecs.
    Started factorization ...
    Finished in 145 millisecs.
    Started factorization ...
    Finished in 29491 millisecs.
    ==== Interpolating field 0 ====
    Started solve ...
    Finished in 29561 millisecs.
    ==== Interpolating field 0 ====
    Started solve ...
    Finished in 108 millisecs.
    Finished in 101 millisecs.
    ===================
    0.840188 0.394383 0.783099
    0.364784 0.513401  0.95223
    0.911647 0.197551 0.335223
    0.277775  0.55397 0.477397
    ===================
    0.259484
    0.178334
    0.0603727
    0.0734613

# To do

- [ ] Add regularization to high-order RBF interpolation
- [ ] Make iterative solvers CG and BiCGstab robust
- [ ] Add NetCDF/HDF5 reading/writing capability for input/output resp
- [ ] Add max iterations or tolerance stopping for iterative solvers
- [ ] RBF with radius search gives symmetric matrix but difficult to chose threshold?
- [ ] Direct solvers upto 3 neighbors very fast probably due to TDMA
- [ ] Investigate banded matrix solvers
