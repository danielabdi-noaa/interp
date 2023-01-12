# interp
A simple RBF interpolation utility considering only k nearest neighbors.
It uses the [nanoflann](https://github.com/jlblancoc/nanoflann) library (header only) for nearest neighbor search.
`nanoflann` comes included with the distribution so it is not technically a dependency.

## Requirements
- C++ Eigen library

## Build
    ./build.sh
## Run
    $ mpirun -n 4 ./interp

    ===== Parameters ====
    numPoints: 200000
    numTargetPoints: 4
    numNeighbors: 8
    numFields: 1
    use_cutoff_radius: false
    cutoff_radius: 0.5
    =====================
    Clustering point clouds into 4 clusters
    cluster 0 with centroid (0.732482  0.77461 0.529731) and 50295 points
    cluster 1 with centroid (0.737033 0.228173  0.47419) and 49705 points
    cluster 2 with centroid (0.266575 0.527163 0.226509) and 50166 points
    cluster 3 with centroid (0.261717 0.467934 0.771812) and 49834 points
    Finished in 323 millisecs.
    Constructing interpolation matrix ...
    Constructing interpolation matrix ...
    Constructing interpolation matrix ...
    Constructing interpolation matrix ...
    Finished in 66 millisecs.
    Started factorization ...
    Finished in 66 millisecs.
    Started factorization ...
    Finished in 67 millisecs.
    Started factorization ...
    Finished in 68 millisecs.
    Started factorization ...
    Finished in 7579 millisecs.
    ==== Interpolating field 0 ====
    Started solve ...
    Finished in 46 millisecs.
    Finished in 7775 millisecs.
    ==== Interpolating field 0 ====
    Started solve ...
    Finished in 7798 millisecs.
    ==== Interpolating field 0 ====
    Started solve ...
    Finished in 43 millisecs.
    Finished in 42 millisecs.
    Finished in 8134 millisecs.
    ==== Interpolating field 0 ====
    Started solve ...
    Finished in 37 millisecs.
    ===================
    0.840188 0.394383 0.783099
    0.911647 0.197551 0.335223
    0.277775  0.55397 0.477397
    0.364784 0.513401  0.95223
    ===================
     0.259484
    0.0603727
    0.0734613
     0.178334

# To do

- [ ] Add regularization to high-order RBF interpolation
- [ ] Make iterative solvers CG and BiCGstab robust
- [ ] Add NetCDF/HDF5 reading/writing capability for input/output resp
- [ ] Add max iterations or tolerance stopping for iterative solvers
- [ ] RBF with radius search gives symmetric matrix but difficult to chose threshold?
- [ ] Direct solvers upto 3 neighbors very fast probably due to TDMA
- [ ] Investigate banded matrix solvers
