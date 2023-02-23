# interp
A simple RBF interpolation utility considering only k nearest neighbors.
It uses the [nanoflann](https://github.com/jlblancoc/nanoflann) library (header only) for nearest neighbor search.
`nanoflann` comes included with the distribution so it is not technically a dependency.

    Interpolate fields in a grib2 file onto another grid or scattered observation locations.
    
    Example:
        OMP_NUM_THREADS=8 ./interp -i rrfs_a.t06z.bgdawpf007.tm00.grib2 -t rrfs.t06z.prslev.f007.ak.grib2 -f 0,3
    
    This does interpolation of fields 0 and 3 using 8 threads from the North-american domain to the Alaska grid.
    
    usage: ./interp [-h] [--input INPUT] [--output OUTPUT] [--template TEMPLATE]
                         [--clusters-per-rank CLUSTERS_PER_RANK] [--fields FIELDS]
                         [--neighbors NEIGHBORS] [--neighbors-interp NEIGHBORS_INTERP]
                         [--rbf-shape RBF_SHAPE] [--use-cutoff-radius USE_CUTOFF_RADIUS]
                         [--cutoff-radius CUTOFF_RADIUS] [--cutoff-radius-interp CUTOFF_RADIUS_INTERP]
    
    arguments:
      -h, --help               show this help message and exit
      -i, --input              grib file containing fields to interpolate
      -o, --output             output grib file contiainig result of interpolation
      -t, --template           template grib file that the output grib file is based on
      -c, --clusters-per-rank  number of clusters per MPI rank
      -f, --fields             comma separated list indices of fields in grib file that are to be interpolated
                               hyphen(-) can be used to indicate range of fields e.g. 0-3 means fields 0,1,2
                               question(?) can be used to indicate all fields in a grib file
      -n, --neighbors          number of neighbors to be used during solution for weights using source points
      -ni, --neighbors-interp  number of neighbors to be used during interpolation at target points
      -r, --rbf-shape          shape factor for RBF kernel
      -ucr, --use-cutoff-radius      use cutoff radius instead of fixed number of nearest neighbors
      -cr, --cutoff-radius           cutoff radius used during solution
      -cri, --cutoff-radius-interp   cutoff radius used during interpolation
      -r, --rbf-smoothing      smoothing factor for rbf interpolation
      -m, --monomials          number of monomials (supported 0 or 1)
      -utf, --use-test-field   use test field function for initializing fields (applies even if grib2 file input is used)
                               this could be useful for tuning parameters with L2 error of ground truth.

## Requirements
- [C++ Eigen library](https://eigen.tuxfamily.org/dox/)
- [ECMWF eccodes library](https://github.com/ecmwf/eccodes)

## Build

    ./build.sh

## Run

To interpolate from the North-American(NA) domain to the conus domain, we should provide the input grib file and a sample template
grib file containing the conus grid lat/lon. We can select fields to interpolate by providing comma-separated list with 0-based index.

    $ OMP_NUM_THREADS=2 ./interp -i rrfs_a.t06z.bgdawpf007.tm00.grib2 -t rrfs.t06z.prslev.f007.conus_3km.grib2 -f 16,17,751,754,771
    Threads: 2
    ===== Parameters ====
    numDims: 2
    numNeighbors: 1
    numNeighborsInterp: 32
    rbfShape: 0
    useCutoffRadius: false
    cutoffRadius: 0.08
    cutoffRadiusInterp: 0.64
    numClustersPerRank: 2
    rbfSmoothing: 0
    monomials: 0
    =====================
    Eigen will be using 1 threads.
    Reading input grib file
    Finished in 3.98184 secs.
    Reading interpolation grid from grib file
    Finished in 0.331067 secs.
    ===== Data size ====
    numPoints: 14452641
    numTargetPoints: 1905141
    numFields: 5
    =====================
    Clustering point clouds into 1 clusters
    Completed 1 iterations.
    cluster 0 with centroid (246.751 41.7439) and 14452641 points
    Finished in 0.105573 secs.
    Clustering point clouds into 2 clusters
    Completed 25 iterations.
    cluster 0 with centroid (294.837 41.6794) and 7238878 points
    cluster 1 with centroid (198.499 41.8086) and 7213763 points
    Finished in 3.28064 secs.
    Automatically computed shape factor: 69.455
    ===========================
    Computing weights for all fields
    Automatically computed shape factor: 68.9903
    ===========================
    Computing weights for all fields
    Interpolating fields
    Interpolating fields
    Finished in 1.04203 secs.
    Finished in 3.23271 secs.
    Writing input and output fields for plotting
    Finished in 21.4104 secs.


To plot interpolation result over the conus domain

    $ python3 plot.py -i output.txt -o conus.png

We can also plot the NA input domain

    $ python3 plot.py -i input.txt -o na.png

The plotting script will plot all interpolated fields.

# To do

- [X] Add regularization to high-order RBF interpolation
- [X] Process multiple clusters per mpi rank to increase usability of direct solvers
- [X] Make iterative solver BiCGstab robust
- [X] Add NetCDF/HDF5 reading/writing capability for input/output resp
- [X] Add max iterations or tolerance stopping for iterative solvers
- [X] RBF with radius search gives symmetric matrix but difficult to chose threshold?
- [X] Direct solvers upto 3 neighbors very fast probably due to TDMA
- [X] Investigate banded matrix solvers
