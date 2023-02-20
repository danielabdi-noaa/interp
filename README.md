# interp
A simple RBF interpolation utility considering only k nearest neighbors.
It uses the [nanoflann](https://github.com/jlblancoc/nanoflann) library (header only) for nearest neighbor search.
`nanoflann` comes included with the distribution so it is not technically a dependency.

    Interpolate fields in a grib2 file onto another grid or scattered observation locations.
    
    Example:
        OMP_NUM_THREADS=8 ./interp -i rrfs_a.t06z.bgdawpf007.tm00.grib2 -t rrfs.t06z.prslev.f007.ak.grib2 -f 0,3
    
    This does interpolation of fields 0 and 3 using 8 threads from the North-american domain to the Alaska grid.
    
    usage: ./interp [-h] [--input INPUT] [--output OUTPUT] [--template TEMPLATE]
                         [ --clusters-per-rank CLUSTERS_PER_RANK] [--fields FIELDS]
                         [ --neighbors NEIGHBORS] [--neighbors-interp NEIGHBORS_INTERP]
    
    arguments:
      -h, --help               show this help message and exit
      -i, --input              grib file containing fields to interpolate
      -o, --output             output grib file contiainig result of interpolation
      -t, --template           template grib file that the output grib file is based on
      -c, --clusters-per-rank  number of clusters per MPI rank
      -f, --fields             comma separated list indices of fields in grib file that are to be interpolated
      -n, --neighbors          number of neighbors to be used during solution for weights using source points
      -ni, --neighbors-interp  number of neighbors to be used during interpolation at target points

## Requirements
- C++ Eigen library
- ECMWF eccodes library

## Build
    ./build.sh
## Run
    $ OMP_NUM_THREADS=8 ./interp -i rrfs_a.t06z.bgdawpf007.tm00.grib2 -t rrfs.t06z.prslev.f007.ak.grib2 
    Eigen will be using 1 threads.
    ===== Parameters ====
    numDims: 2
    numNeighbors: 8
    numNeighborsTarget: 32
    rbfShape: 40
    useCutoffRadius: false
    cutoffRadius: 0.08
    cutoffRadiusTarget: 0.64
    nonParametric: true
    blendInterp: false
    numClustersPerRank: 1
    matEpsilon: 0
    rbfSmoothing: 0
    monomials: 0
    =====================
    Reading input grib file
    Finished in 1.43058 secs.
    Reading interpolation grid from grib file
    Finished in 0.262793 secs.
    ===== Data size ====
    numPoints: 14452641
    numTargetPoints: 1822145
    numFields: 1
    =====================
    Clustering point clouds into 1 clusters
    cluster 0 with centroid (246.751 41.7439) and 14452641 points
    Finished in 0.119363 secs.
    ===========================
    Interpolating fields
    Finished in 3.37725 secs.
    Writing input and output fields for plotting
    Finished in 9.2294 secs.

# To do

- [X] Add regularization to high-order RBF interpolation
- [X] Process multiple clusters per mpi rank to increase usability of direct solvers
- [ ] Make iterative solvers CG and BiCGstab robust
- [X] Add NetCDF/HDF5 reading/writing capability for input/output resp
- [X] Add max iterations or tolerance stopping for iterative solvers
- [X] RBF with radius search gives symmetric matrix but difficult to chose threshold?
- [X] Direct solvers upto 3 neighbors very fast probably due to TDMA
- [X] Investigate banded matrix solvers
