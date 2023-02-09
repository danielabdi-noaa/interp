#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <chrono>
#include <eccodes.h>
#include "knn/nanoflann.hpp"

using namespace std;
using namespace Eigen;

/***************************************************
 * Parameters for the interpolation
 **************************************************/

// number of dimesnions 1D,2D,3D are supported
constexpr int numDims = 2;

// matrix coefficients below this value are zeroed (pruned)
constexpr double matrix_epsilon = 1e-5;

// Rbf shape function can be computed approximately from
// the average distance between points
//     rbf_shape  = 0.8 / average_distance
// Given npoints and numDims and [-1, 1] axis ranges
//     average_distance = 2 / npoints^(1/numDims)
//     rbf_shape = 0.4 * npoints^(1/numDims)
constexpr double rbf_shape = 64;

// Rbf smoothing factor, often set to 0 for interpolation
// but can be set to positive value for noisy data.
constexpr double rbf_smoothing = 0.01;

// Number of neighbors to consider for interpolation
constexpr int numNeighbors = 8;

// Cutoff radius for nearest neighbor interpolation
constexpr bool use_cutoff_radius = false;
constexpr double cutoff_radius = 0.5;

// Flag to set non-parameteric RBF interpolation
constexpr bool non_parametric = true;

// Number of clusters to process per MPI rank
constexpr int numClustersPerRank = 1;

/*********************
 *  Timer class
 *********************/
class Timer {
private:
	using Clock = std::chrono::steady_clock;
	using Second = std::chrono::duration<double, std::ratio<1> >;
	std::chrono::time_point<Clock> m_beg { Clock::now() };
public:
	void reset() {
		m_beg = Clock::now();
	}
	void elapsed() {
		auto duration = std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
        std::cout << "Finished in " << duration << " secs." << std::endl;
        reset();
	}
};

/***************************************************
 * Nearest neighbor search using nanoflann library
 **************************************************/

// Point cloud data type
class PointCloud {

    private:
        MatrixXd& point_cloud;
        size_t num_points;

    public:
        PointCloud(MatrixXd& pc, size_t np) :
            point_cloud(pc), num_points(np) {
        }

        /*functions needed by nanoflann*/
        inline size_t kdtree_get_point_count() const { return num_points; }

        inline double kdtree_get_pt(const size_t idx, int dim) const {
            return point_cloud(dim,idx);
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

//typdef nanonflann KDTree
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>, 
    PointCloud, numDims> KDTree;

//find k nearaset neighbors at location "query" and return indices and distances
void knn(const KDTree& index, int k, double* query, size_t* indices, double* distances) {
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(indices, distances);
    index.findNeighbors(resultSet, query, nanoflann::SearchParameters());
}

//find nearaset neighbors with in a given radius at location "query"
unsigned int knn_radius(const KDTree& index, double radius, double* query,
        std::vector<nanoflann::ResultItem<unsigned, double>>& matches
        ) {
    return index.radiusSearch(query, radius * radius, matches, nanoflann::SearchParameters());
}

/********************************************************************
 * Clustering using k-means (Lloyd's algorithm).
 * Easy to implement and not that critical, so lets do it ourselves.
 ********************************************************************/

void kMeansClustering(const MatrixXd& points, int numPoints, int numClusters,
        VectorXi& clusterAssignments, VectorXi& clusterSizes, MatrixXd& clusterCenters) {

    std::cout << "Clustering point clouds into " << numClusters << " clusters" << std::endl;
    Timer t;

    // Initialize the cluster centers
    for (int i = 0; i < numClusters; i++) {
        clusterCenters.col(i) = points.col(i);
    }

    // Perform k-means clustering until the cluster assignments stop changing
    MatrixXd sumClusterCenters(numDims, numClusters);
    VectorXd d(3);

    bool converged = false;
    while (!converged) {

        // Update the cluster assignments
        converged = true;
        for (int i = 0; i < numPoints; i++) {
            int closestCluster = -1;
            double minDistance = std::numeric_limits<double>::max();
            for (int j = 0; j < numClusters; j++) {
                d  = points.col(i) - clusterCenters.col(j);
                double distance = d.dot(d);

                if (distance < minDistance) {
                    minDistance = distance;
                    closestCluster = j;
                }
            }
            if (closestCluster != clusterAssignments(i)) {
                clusterAssignments(i) = closestCluster;
                converged = false;
            }
        }

        // Update the cluster centers
        sumClusterCenters.setZero(numDims,numClusters);
        clusterSizes.setZero(numClusters);

        for (int i = 0; i < numPoints; i++) {
            int cluster = clusterAssignments(i);
            sumClusterCenters.col(cluster) += points.col(i);
            clusterSizes(cluster)++;
        }
        for (int i = 0; i < numClusters; i++) {
            if (clusterSizes(i) > 0) {
                clusterCenters.col(i) = sumClusterCenters.col(i) / clusterSizes(i);
            }
        }
    }

    //Print final cluster sizes
    for (int i = 0; i < numClusters; i++) {
        std::cout << "cluster " << i << " with centroid ("
            << clusterCenters.col(i).transpose() << ") and "
            << clusterSizes(i) << " points" << std::endl;
    }

    t.elapsed();
}

/**************************************************************
 * Define Solver for Ax=B equation from Eigen
 * Depending on how you construct the interpolation matrix choose
 * the right solver. LU vs Cholesky decomp, direct vs iterative etc.
 * Only one solver can be chosen at one time.
 **************************************************************/

typedef SparseLU<SparseMatrix<double>> RbfSolver;
//typedef SimplicialLDLT<SparseMatrix<double>, Upper> RbfSolver;

//typedef BiCGSTAB<SparseMatrix<double>, DiagonalPreconditioner<double>> RbfSolver;
//typedef BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double,int>> RbfSolver;
//typedef ConjugateGradient<SparseMatrix<double>, Lower|Upper, DiagonalPreconditioner<double>> RbfSolver;
//typedef ConjugateGradient<SparseMatrix<double>, Lower|Upper, IncompleteLUT<double,int>> RbfSolver;

/**************************************************************
 * RBF interpolation using nearest neighbor search
 **************************************************************/

//
// Radial basis function
//
double rbf(double r_) {
    double r = (rbf_shape * r_);

    // gaussian
    return exp(-r*r);

    // multiquadric
    //return sqrt(1 + r*r);

    // inverse multiquadric
    //return 1 / sqrt(1 + r*r);

    // thin plate spline
    //return r*r*log(r+1e-5);

    // compact support gaussian bump
    //if(r < 1.0 / rbf_shape)
    //   return exp(-1 / (1 - r*r));
    //else
    //   return 0;

    // inverse-distance interp
    // return 1.0 / pow(r_,3);
}

//
// Build sparse interpolation matrix and LU decompose it
//
void rbf_build(const KDTree& index, const MatrixXd& X,
        const int numPoints, RbfSolver& solver, SparseMatrix<double>& A
        ) {

    //
    // Build sparse Matrix of radial basis function evaluations
    //
    std::cout << "Constructing interpolation matrix ..." << std::endl;
    Timer t;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(numPoints * numNeighbors);

    VectorXd query(numDims);

    if(use_cutoff_radius) {
        std::vector<nanoflann::ResultItem<unsigned, double>> matches;
        for (int i = 0; i < numPoints; i++) {
            // Perform a radius search
            query = X.col(i);
            unsigned nMatches = knn_radius(index, cutoff_radius, query.data(), matches);

            // Add matrix coefficients
            for (int k = 0; k < nMatches; k++) {
                int j = matches[k].first;
                double r = rbf(sqrt(matches[k].second));
                if(i == j)
                    r += rbf_smoothing;
                if(fabs(r) > matrix_epsilon)
                    tripletList.push_back(T(i,j,r));
            }
        }
    } else {
        vector<size_t> indices(numNeighbors);
        vector<double> distances(numNeighbors);
        for (int i = 0; i < numPoints; i++) {

            // Perform the k-nearest neighbor search
            query = X.col(i);
            knn(index, numNeighbors, query.data(), &indices[0], &distances[0]);

            // Add matrix coefficients
            for (int k = 0; k < numNeighbors; k++) {
                int j = indices[k];
                double r = rbf(sqrt(distances[k]));
                if(i == j)
                    r += rbf_smoothing;
                if(fabs(r) > matrix_epsilon)
                    tripletList.push_back(T(i,j,r));
            }
        }
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());

    t.elapsed();

    //
    // LU decomposition of the interpolation matrix
    //
    std::cout << "Started factorization ..." << std::endl;

    solver.compute(A);

    if(solver.info() != Success) {
        std::cout << "Factorization failed." << std::endl;
        exit(0);
    }

    t.elapsed();
}

//
// Solve Ax=b from solver that already has LU decomposed matrix
// Iterative solvers don't need LU decompostion to happen first
//
void rbf_solve(RbfSolver& solver, const VectorXd& F, VectorXd& C) {
    std::cout << "Started solve ..." << std::endl;
    Timer t;

    C = solver.solve(F);

#if 0
    std::cout << "----------------------------------" << std::endl;
    std::cout << "numb iterations: " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error()      << std::endl;
    std::cout << "----------------------------------" << std::endl;
#endif

    t.elapsed();
}

/**************************
 * Global data namespace
 **************************/
namespace GlobalData {

    int numClusters;
    int mpi_rank;
    
    //source points and fields, and target points
    MatrixXd* points_p;
    MatrixXd* fields_p;
    MatrixXd* target_points_p;

    //matrices for storing interpolated points and fields at targets
    //This is needed because order of points is changed after interpolation.
    //May need to maintain same order in the future.
    MatrixXd* interp_target_points_p;
    MatrixXd* interp_target_fields_p;

    //cluster sizes for source and target points
    VectorXi clusterSizes;
    VectorXi target_clusterSizes;

    //parameters
    int g_numPoints;
    int g_numTargetPoints;
    int numFields;

    // lat/lon boundaries, use offest to account
    // for curved HRRR grid
    const double lat_min = 21.14 + 10;
    const double lat_max = 52.63 - 10;
    const double lon_min = 225.9 + 10;
    const double lon_max = 299.1 - 10;

    // input/output grid dimensions
    const int n_lon_i = 1799;
    const int n_lat_i = 1059;
    const int n_lon_o = 7000;
    const int n_lat_o = 3500;

    //initialize global params
    void init(int nc, int r) {
        numClusters = nc;
        mpi_rank = r;
        points_p = nullptr;
        fields_p = nullptr;
        target_points_p = nullptr;
        interp_target_points_p = nullptr;
        interp_target_fields_p = nullptr;
        clusterSizes.resize(numClusters);
        target_clusterSizes.resize(numClusters);

        if(mpi_rank == 0) {
            std::cout << "===== Parameters ====" << std::endl
                      << "numDims: " << numDims << std::endl
                      << "numNeighbors: " << numNeighbors << std::endl
                      << "numClustersPerRank: " << numClustersPerRank << std::endl
                      << "use_cutoff_radius: " << (use_cutoff_radius ? "true" : "false") << std::endl
                      << "cutoff_radius: " << cutoff_radius << std::endl
                      << "non_parametric: " << (non_parametric ? "true" : "false") << std::endl
                      << "=====================" << std::endl;
        }
    }
    //
    // Generate random data
    //
    void generate_random_data(
          MatrixXd*& points, MatrixXd*& fields,
          MatrixXd*& target_points
    ) {
        int numPoints = n_lat_i*n_lon_i;
        int numTargetPoints = n_lat_o*n_lon_o;
        numFields = 1;

        // Save global number of points
        g_numPoints = numPoints;
        g_numTargetPoints = numTargetPoints;

        // Allocate
        points = new MatrixXd(numDims, numPoints);
        fields = new MatrixXd(numFields, numPoints);
        target_points = new MatrixXd(numDims, numTargetPoints);
        
        // Generate a set of random 3D points and associated field values
        points->setRandom();
        for (int i = 0; i < numPoints; i++) {
            for(int j = 0; j < numFields; j++) {
                const double x = points->col(i).norm();
                constexpr double pi = 3.14159265358979323846;
                // wiki example function for rbf interpolation
                (*fields)(j, i) = exp(x*cos(3*pi*x)) * (j + 1);
            }
        }
        points->row(0) = (points->row(0).array() + 1.0) /
                                2.0 * (lat_max - lat_min) + lat_min;
        points->row(1) = (points->row(1).array() + 1.0) /
                                2.0 * (lon_max - lon_min) + lon_min;

        // Generate random set of target points
        target_points->setRandom();
        target_points->row(0) = (target_points->row(0).array() + 1.0) /
                                2.0 * (lat_max - lat_min) + lat_min;
        target_points->row(1) = (target_points->row(1).array() + 1.0) /
                                2.0 * (lon_max - lon_min) + lon_min;
    }

    //
    // Read grib file
    //
    void read_grib_file(
          const char* filename,
          MatrixXd*& points, MatrixXd*& fields,
          MatrixXd*& target_points
    ) {
        int ret;
        size_t numPoints, numTargetPoints;

        //hard code longitude and latitude index
        const int idx_nlat = 990;
        const int idx_elon = 991;

        // Count the total number of fields in the GRIB2 file
        FILE* fp = fopen(filename, "r");
        if(!fp) return;

        Timer t;
        std::cout << "Reading input grib file" << std::endl;

        numFields = 0;
        while (codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret)) {
          if(numFields == 0) {
              CODES_CHECK(codes_get_size(h, "values", &numPoints), 0);
          }
          codes_handle_delete(h);
          ++numFields;
        }
        rewind(fp);

        // set these values just for testing
        numTargetPoints = n_lat_o * n_lon_o;
        numFields = 1;

        // Save global number of points
        g_numPoints = numPoints;
        g_numTargetPoints = numTargetPoints;

        // Allocate
        points = new MatrixXd(numDims, numPoints);
        fields = new MatrixXd(numFields, numPoints);
        target_points = new MatrixXd(numDims, numTargetPoints);

        // loop through all fields and read data
        VectorXd values(numPoints);
        int idx = 0, f = 0;
        while (codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret)) {

          if(idx < numFields) {
            CODES_CHECK(codes_get_double_array(h, "values",
                        values.data(), &numPoints), 0);

            fields->row(f) = values;
            f++;
          } else if(idx == idx_nlat) {
            CODES_CHECK(codes_get_double_array(h, "values",
                        values.data(), &numPoints), 0);

            points->row(0) = values;
          } else if(idx == idx_elon) {
            CODES_CHECK(codes_get_double_array(h, "values",
                        values.data(), &numPoints), 0);

            points->row(1) = values;
          }

          codes_handle_delete(h);
          idx++;
        }

        t.elapsed();

#if 1
        std::cout << "Writing input field for plotting" << std::endl;
        FILE* fh = fopen("input.txt", "w");
        for(int i = 0; i < g_numPoints; i++) {
           fprintf(fh, "%.2f %.2f %.2f\n",
               (*points)(0,i),
               (*points)(1,i),
               (*fields)(0,i));
        }
        fclose(fh);

        t.elapsed();
#endif

        // Generate random target points in the given lat/lon range
#if 0
        target_points->setRandom();
        target_points->row(0) = (target_points->row(0).array() + 1.0) /
                                2.0 * (lat_max - lat_min) + lat_min;
        target_points->row(1) = (target_points->row(1).array() + 1.0) /
                                2.0 * (lon_max - lon_min) + lon_min;
#else
        std::cout << "Reading interpolation grid" << std::endl;
        fh = fopen("grid.txt", "r");
        char buffer[256];
        idx = 0;
        while(fgets(buffer, 256, fh)) {
           float lat,lon;
           sscanf(buffer, "%f %f", &lat, &lon);
           (*target_points)(0, idx) = lat;
           (*target_points)(1, idx) = lon;
           idx++;
        }

        t.elapsed();
#endif
    }

    //
    // write grib file
    //
    void write_grib_file() {
        const char* filename_s = "rrfs.t21z.prslev.f002.conus_3km.grib2";
        const char* filename_d = "out.grib2";
        size_t size = g_numTargetPoints;
        VectorXd values(size);
        FILE* fp_s = fopen(filename_s, "r");
        if(!fp_s) return;

        Timer t;
#if 1
        std::cout << "Writing input field for plotting" << std::endl;
        FILE* fh = fopen("output.txt", "w");
        for(int i = 0; i < g_numTargetPoints; i++) {
           fprintf(fh, "%.2f %.2f %.2f\n",
             (*interp_target_points_p)(0,i),
             (*interp_target_points_p)(1,i),
             (*interp_target_fields_p)(0,i));
        }
        fclose(fh);
        t.elapsed();
#endif

        std::cout << "Writing output grib file." << std::endl;

        int ret, idx = 0;
        while (codes_handle* h = codes_handle_new_from_file(0, fp_s, PRODUCT_GRIB, &ret)) {

          long bitsPerValue = 0;
          codes_get_long(h, "bitsPerValue", &bitsPerValue);

#if 0
          codes_keys_iterator* iter = codes_keys_iterator_new(h, 0, 0);
          while (codes_keys_iterator_next(iter)) {
            const char* key = codes_keys_iterator_get_name(iter);
            char value[64];
            size_t len = 64;
            CODES_CHECK(codes_get_string(h, key, value, &len), 0);
            std::cout << key << ": " << value << std::endl;
          }
          codes_keys_iterator_delete(iter);
#endif

          // clone handle and write to output file
          codes_handle* new_h = codes_handle_clone(h);

          codes_set_long(new_h, "Nx", n_lon_o);
          codes_set_long(new_h, "Ny", n_lat_o);
          codes_set_long(new_h, "numberOfDataPoints", size);
          codes_set_long(new_h, "numberOfValues", size);
          codes_set_long(new_h, "getNumberOfValues", size);

          if(bitsPerValue > 0) {
             values = interp_target_fields_p->row(idx);

             CODES_CHECK(codes_set_double_array(new_h, "values",
                      values.data(), size), 0);
          }

          codes_write_message(new_h, filename_d, idx ? "a" : "w");
          codes_handle_delete(new_h);

          // delete handle
          codes_handle_delete(h);
          idx++;

          if(idx >= numFields) break;
        }

        t.elapsed();
    }
    //
    // Partition data across MPI ranks
    //
    void partition_data(
          int numPoints, int numTargetPoints,
          const MatrixXd& points, const MatrixXd& fields,
          const MatrixXd& target_points
    ) {

        // Partition the source points into N clusters
        VectorXi clusterAssignments(numPoints);
        MatrixXd clusterCenters(numDims,numClusters);
        
        clusterAssignments.setZero();
        kMeansClustering(points, numPoints, numClusters,
                clusterAssignments, clusterSizes, clusterCenters);
        
        // Sort points and fields
        points_p = new MatrixXd(numDims, numPoints);
        fields_p = new MatrixXd(numFields, numPoints);
        MatrixXd& sorted_points = *points_p;
        MatrixXd& sorted_fields = *fields_p;
        
        int idx = 0;
        for (int i = 0; i < numClusters; i++) {
            for(int j = 0; j < numPoints; j++) {
                if(clusterAssignments(j) != i)
                    continue;
                sorted_points.col(idx) = points.col(j);
                for(int k = 0; k < numFields; k++)
                    sorted_fields(k, idx) = fields(k, j);
                idx++;
            }
        }
        
        // Partition target points
        VectorXd d(numPoints);
        VectorXi target_clusterAssignments(numTargetPoints);

        target_clusterSizes.setZero(numClusters);
        for(int i = 0; i < numTargetPoints; i++) {
            int closestCluster = -1;
            double minDistance = std::numeric_limits<double>::max();
            for (int j = 0; j < numClusters; j++) {
                d = target_points.col(i) - clusterCenters.col(j);
                double distance = std::sqrt(d.dot(d));
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCluster = j;
                }
            }
            target_clusterAssignments(i) = closestCluster;
            target_clusterSizes(closestCluster)++;
        }
        
        // Sort target points
        target_points_p = new MatrixXd(numDims, numTargetPoints);
        MatrixXd& sorted_target_points = *target_points_p;
        
        idx = 0;
        for (int i = 0; i < numClusters; i++) {
            for(int j = 0; j < numTargetPoints; j++) {
                if(target_clusterAssignments(j) != i)
                    continue;
                sorted_target_points.col(idx) = target_points.col(j);
                idx++;
            }
        }
    }
    //
    // Read and partition data on master node. Assumes node is big enough
    // to store and process the data.
    //
    void read_and_partition_data() {

        MatrixXd *points = nullptr, *fields = nullptr, *target_points = nullptr;

#if 0
        generate_random_data(points, fields, target_points);
#else
        const char* filename_s = "rrfs.t21z.prslev.f002.conus_3km.grib2";
        read_grib_file(filename_s,points, fields, target_points);
#endif

        std::cout << "===== Data size ====" << std::endl
                  << "numPoints: " << g_numPoints << std::endl
                  << "numTargetPoints: " << g_numTargetPoints << std::endl
                  << "numFields: " << numFields << std::endl
                  << "=====================" << std::endl;

        // Partition data across mpi ranks
        partition_data(
            g_numPoints, g_numTargetPoints,
            *points, *fields, *target_points);
    }

};

/*************************************
 * Cluster data structure.
 ************************************/
struct ClusterData {

    int numPoints;
    int numTargetPoints;
    MatrixXd points;
    MatrixXd fields;
    MatrixXd target_points;
    MatrixXd target_fields;
    PointCloud* cloud;
    KDTree* ptree;
    SparseMatrix<double> A;
    RbfSolver solver;
    
    //
    // Scatter data to slave processors
    //
    void scatter() {
        using namespace GlobalData;

        // scatter number of points
        MPI_Scatter(clusterSizes.data(), 1, MPI_INT,
                &numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //allocate space for points and fields
        points.resize(numDims, numPoints);
        fields.resize(numFields, numPoints);

        //offsets and coutns
        VectorXi offsets(numClusters);
        VectorXi counts(numClusters);

        //scatter points
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = clusterSizes(i) * numDims;
                offset += counts(i);
            }
        }
        MPI_Scatterv(points_p ? points_p->data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                points.data(), numPoints * numDims, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //scatter fields
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = clusterSizes(i) * numFields;
                offset += counts(i);
            }
        }
        MPI_Scatterv(fields_p ? fields_p->data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                fields.data(), numPoints * numFields, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Get local target points assinged to this mpi_rank
        MPI_Scatter(target_clusterSizes.data(), 1, MPI_INT,
                &numTargetPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

        target_points.resize(numDims, numTargetPoints);
        target_fields.resize(numFields, numTargetPoints);

        //Scatter target points
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = target_clusterSizes(i) * numDims;
                offset += counts(i);
            }
        }

        MPI_Scatterv(target_points_p ? target_points_p->data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                target_points.data(), numTargetPoints * numDims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
    }
    //
    // Gather data from slave processors
    //
    void gather() {
        using namespace GlobalData;

        //offsets and coutns
        VectorXi offsets(numClusters);
        VectorXi counts(numClusters);
    
        //Gather target points
        interp_target_points_p = nullptr;
        if(mpi_rank == 0)
            interp_target_points_p = new MatrixXd(numDims, g_numTargetPoints);
    
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = target_clusterSizes(i) * numDims;
                offset += counts(i);
            }
        }
        MPI_Gatherv(target_points.data(), numTargetPoints * numDims, MPI_DOUBLE,
                interp_target_points_p ? interp_target_points_p->data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
        //Gather interpolated fields
        interp_target_fields_p = nullptr;
        if(mpi_rank == 0)
            interp_target_fields_p = new MatrixXd(numFields, g_numTargetPoints);
    
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = target_clusterSizes(i) * numFields;
                offset += counts(i);
            }
        }
        MPI_Gatherv(target_fields.data(), numTargetPoints * numFields, MPI_DOUBLE,
                interp_target_fields_p ? interp_target_fields_p->data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

#if 0
        // print interpolated fields with associated coordinates
        // Note that the order of points is changed.
        if(mpi_rank == 0) {
            std::cout << "===================" << std::endl;
            std::cout << interp_target_points_p->transpose() << std::endl;
            std::cout << "===================" << std::endl;
            std::cout << interp_target_fields_p->transpose() << std::endl;
        }
#endif
    }
    //
    // Build KD tree needed for fast nearest neighbor search
    //
    void build_kdtree() {
        cloud = new PointCloud(points, numPoints);
        ptree = new KDTree(numDims, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams());
        ptree->buildIndex();
    }
    //
    // Build sparse RBF interpolation matrix using either k nearest neigbors
    // or cutoff radius criteria
    //
    void build_rbf() {
        if(non_parametric)
            return;
        A.resize(numPoints, numPoints);
        rbf_build(*ptree, points, numPoints, solver, A);
    }
    //
    // Interpolate each field using parameteric RBF
    //
    void solve_rbf() {
        using namespace GlobalData;
        VectorXd F(numPoints);
        MatrixXd W(numPoints, numFields);

        target_fields.setZero();

        std::cout << "===========================" << std::endl;

        //compute weights for each field
        if(!non_parametric) {
            VectorXd Wf(numPoints);
            std::cout << "Computing weights for all fields" << std::endl;
            for(int f = 0; f < numFields; f++) {
                F = fields.row(f);
                rbf_solve(solver, F, Wf);
                W.col(f) = Wf;
            }
        }

        //interpolate for target fields
        std::cout << "Interpolating fields" << std::endl;
        Timer t;

        if(use_cutoff_radius) {
            VectorXd query(numDims);
            std::vector<nanoflann::ResultItem<unsigned, double>> matches;
            for (int i = 0; i < numTargetPoints; i++) {
                // Perform a radius search
                query = target_points.col(i);
                unsigned nMatches = knn_radius(*ptree, cutoff_radius, query.data(), matches);

                // interpolate
                double sum = 0;
                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = rbf(sqrt(matches[k].second));
                    if(!non_parametric) {
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += W(j, f) * r;
                    } else
                        sum += std::max(r,1e-6);
                }
                if(non_parametric)
                {
                    for (int k = 0; k < nMatches; k++) {
                        int j = matches[k].first;
                        double r = rbf(sqrt(matches[k].second));
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += fields(f, j) * (std::max(r,1e-6) / sum);
                    }
                }
            }
        } else {
            VectorXd query(numDims);
            vector<size_t> indices(numNeighbors);
            vector<double> distances(numNeighbors);
            for (int i = 0; i < numTargetPoints; i++) {
                // Perform the k-nearest neighbor search
                query = target_points.col(i);
                knn(*ptree, numNeighbors, query.data(), &indices[0], &distances[0]);

                // interpolate
                double sum = 0;
                for (int k = 0; k < numNeighbors; k++) {
                    int j = indices[k];
                    double r = rbf(sqrt(distances[k]));
                    if(!non_parametric) {
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += W(j, f) * r;
                    } else
                        sum += std::max(r,1e-6);
                }
                if(non_parametric)
                {
                    for (int k = 0; k < numNeighbors; k++) {
                        int j = indices[k];
                        double r = rbf(sqrt(distances[k]));
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += fields(f, j) * (std::max(r,1e-6) / sum);
                    }
                }
            }
        }

        t.elapsed();
    }
    //
    // Conveneince function to build and solve rbf interpolation
    //
    void build_and_solve() {
        build_kdtree();
        build_rbf();
        solve_rbf();
    }
};

//
// Split ClusterData into multiple partitions to be processed by threads.
// Its main advantage is however to help direct solvers that scale with O(n^3).
// Splitting cluster into 2, helps by 8x times more for dense matrices, so even if
// one thread processes all sub-clusters, it is still very helpful given cluster
// size does not compromise interpolation at boundaries by a lot. 
//
void split_cluster(const ClusterData& parent, int numClusters,
                   ClusterData* subclusters, VectorXi& target_clusterAssignments
) {
    using GlobalData::numFields;

    MatrixXd clusterCenters(numDims,numClusters);

    VectorXi clusterAssignments(parent.numPoints);
    VectorXi clusterSizes;
    VectorXi target_clusterSizes;
       
    // Create sub clusters
    kMeansClustering(parent.points, parent.numPoints, numClusters,
            clusterAssignments, clusterSizes, clusterCenters);
    
    // Initialize subcluster source points & fields
    for(int i = 0; i < numClusters; i++) {
        ClusterData& cd = subclusters[i];
        cd.numPoints = clusterSizes(i);
        cd.points.resize(numDims, cd.numPoints);
        cd.fields.resize(numFields, cd.numPoints);
    }

    // Populate source point and fields of subclusters
    std::vector<int> cidx(numClusters, 0);
    for(int i = 0; i < parent.numPoints; i++) {
        int owner = clusterAssignments(i);
        int idx = cidx[owner];

        ClusterData& cd = subclusters[owner];
        cd.points.col(idx) = parent.points.col(i);
        for(int f = 0; f < numFields; f++)
            cd.fields(f,idx) = parent.fields(f,i);
        cidx[owner]++;
    }

    // Partition target points
    VectorXd d(3);
    target_clusterSizes.setZero(numClusters);
    for(int i = 0; i < parent.numTargetPoints; i++) {
        int closestCluster = -1;
        double minDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < numClusters; j++) {
            d = parent.target_points.col(i) - clusterCenters.col(j);
            double distance = std::sqrt(d.dot(d));
            if (distance < minDistance) {
                minDistance = distance;
                closestCluster = j;
            }
        }
        target_clusterAssignments(i) = closestCluster;
        target_clusterSizes(closestCluster)++;
    }

    // Initialize subcluster target points & fields
    for(int i = 0; i < numClusters; i++) {
        ClusterData& cd = subclusters[i];
        cd.numTargetPoints = target_clusterSizes(i);
        cd.target_points.resize(numDims, cd.numTargetPoints);
        cd.target_fields.resize(numFields, cd.numTargetPoints);
    }

    // Populate target points of subclusters
    std::vector<int> target_cidx(numClusters, 0);
    for(int i = 0; i < parent.numTargetPoints; i++) {
        int owner = target_clusterAssignments(i);
        int idx = target_cidx[owner];

        ClusterData& cd = subclusters[owner];
        cd.target_points.col(idx) = parent.target_points.col(i);
        target_cidx[owner]++;
    }
}

//
// Merge subcluster interpolation result back into parent cluster
// We only need to update the target fields.
//
void merge_cluster(ClusterData& parent, int numClusters,
                   const ClusterData* subclusters, const VectorXi& target_clusterAssignments
) {
    using GlobalData::numFields;

    std::vector<int> target_cidx(numClusters, 0);
    for(int i = 0; i < parent.numTargetPoints; i++) {
        int owner = target_clusterAssignments(i);
        int idx = target_cidx[owner];

        const ClusterData& cd = subclusters[owner];
        for(int f = 0; f < numFields; f++)
            parent.target_fields(f,i) = cd.fields(f,idx);
        target_cidx[owner]++;
    } 
}

/*****************
 * Test driver
 *****************/

int main(int argc, char** argv) {

    // Initialize MPI
    int nprocs, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    //
    // Cluster and decompose data
    //

    GlobalData::init(nprocs, mpi_rank);
    if(mpi_rank == 0)
        GlobalData::read_and_partition_data();

    //
    // Each rank gets one cluster that it solves independently of other ranks
    // It may choose to "divide and conquer" the cluster for the sake of
    // direct solvers. Decide if to use threading here or linear equation solvers?
    //

    ClusterData parent_cluster;

    // scatter data to slave nodes
    parent_cluster.scatter();

    // build and solve rbf interpolation
    if(numClustersPerRank <= 1)
    {
        parent_cluster.build_and_solve();
    }
    else
    {
        ClusterData child_clusters[numClustersPerRank];
        VectorXi target_clusterAssignments(parent_cluster.numTargetPoints);

        //split cluster
        split_cluster(parent_cluster, numClustersPerRank,
                      child_clusters, target_clusterAssignments);

        //solve mini-clusters independently
        for(int i = 0; i < numClustersPerRank; i++)
            child_clusters[i].build_and_solve();

        //merge clusters
        merge_cluster(parent_cluster, numClustersPerRank,
                      child_clusters, target_clusterAssignments);
    }

    // Gather result from slave nodes
    parent_cluster.gather();

    // Write result to a grib file
    GlobalData::write_grib_file();

    //
    // Finalize MPI
    //

    MPI_Finalize();

    return 0;
}

