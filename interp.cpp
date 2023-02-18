#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <string>
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
constexpr double matrix_epsilon = 0.0;

// Rbf shape function can be computed approximately from
// the average distance between points
//     rbf_shape  = 0.8 / average_distance
// If D is width of the domain
//     rbf_shape = 0.8 / (D / npoints^(1/numDims))
constexpr double rbf_shape = 48.98; //1.85=40; //48.98=1059;

// Rbf smoothing factor, often set to 0 for interpolation
// but can be set to positive value for noisy data.
constexpr double rbf_smoothing = 0.0;

// Number of neighbors to consider for interpolation
constexpr int numNeighbors = 32;

// Cutoff radius for nearest neighbor interpolation
constexpr bool use_cutoff_radius = false;
constexpr double cutoff_radius = 4 * (0.8 / rbf_shape);

// Flag to set non-parametric RBF interpolation
constexpr bool non_parametric = false;

// Blend non-parameteric vs parametric by radial distance
constexpr bool blend_interp = false;

// Number of clusters to process per MPI rank
constexpr int numClustersPerRank = 1;

// Polynomial degree
constexpr int degree = 0;

/*********************
 *  Grid lat/long
 *********************/
// lat/lon boundaries, use offest to account
// for curved lambert-conformal conic grid
constexpr double lat_min = 21.14 + 10;
constexpr double lat_max = 52.63 - 10;
constexpr double lon_min = 225.9 + 10;
constexpr double lon_max = 299.1 - 10;

// input/output grid dimensions
constexpr int n_lon_i = 1799;
constexpr int n_lat_i = 1059;
constexpr int n_lon_o = 3300; //7000;
constexpr int n_lat_o = 2000; //3500;

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
        VectorXi& clusterAssignments, VectorXi& clusterSizes, MatrixXd& clusterCenters,
        bool use_random_init = false
    ) {

    std::cout << "Clustering point clouds into " << numClusters << " clusters" << std::endl;
    Timer t;

    // Initialize the cluster centers
    if(use_random_init) {
        for (int i = 0; i < numClusters; i++) {
            clusterCenters.col(i) = points.col(rand() % numPoints);
        }
    } else {
        const int lat_partition[] = {
           0, 1, 1, 1, 2, 0, 2, 0, 2, 3, 2, 0, 3, 0, 2, 3, 4, 0, 3, 0, 4,
              0, 0, 0, 4, 5, 0, 0, 4, 0, 5, 0, 4, 0, 0, 5, 4, 0, 0, 0, 4};
        if(numClusters > 40 || lat_partition[numClusters] == 0) {
            std::cout << "Please use a different number of MPI ranks suitable for lat-lon partioning." << std::endl;
            exit(0);
        }
        int n_lat = lat_partition[numClusters];
        int n_lon = (numClusters / n_lat);
        double lat_d = (lat_max - lat_min) / n_lat;
        double lon_d = (lon_max - lon_min) / n_lon;

        for (int i = 0; i < n_lat; i++) {
            for (int j = 0; j < n_lon; j++) {
                clusterCenters(0,i * n_lon + j) = lon_min + (lon_d / 2) + j * lon_d;
                clusterCenters(1,i * n_lon + j) = lat_min + (lat_d / 2) + i * lat_d;
            }
        }
    }

    // Perform k-means clustering until the cluster assignments stop changing
    MatrixXd sumClusterCenters(numDims, numClusters);
    VectorXd d(numDims);

    bool converged = false;
    while (!converged) {

        // Update the cluster assignments
        converged = true;
#pragma omp parallel for private(d)
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
#pragma omp paralle for reduction(+:sumClusterCenters,clusterSizes)
        for (int i = 0; i < numPoints; i++) {
            int cluster = clusterAssignments(i);
            sumClusterCenters.col(cluster) += points.col(i);
            clusterSizes(cluster)++;
        }
        for (int i = 0; i < numClusters; i++) {
            if (clusterSizes(i) > 0)
                clusterCenters.col(i) = sumClusterCenters.col(i) / clusterSizes(i);
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
    //return 1.0 / pow(r,3);
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
    tripletList.reserve(numPoints * numNeighbors + 2 * degree * numPoints);

    VectorXd query(numDims);

    if(use_cutoff_radius) {
        std::vector<nanoflann::ResultItem<unsigned, double>> matches;
        for (int i = 0; i < numPoints; i++) {
            // Perform a radius search
            query = X.col(i);
            unsigned nMatches = knn_radius(index, cutoff_radius, query.data(), matches);

            // Compute matrix coefficients
            double sum = 0;
            for (int k = 0; k < nMatches; k++) {
                int j = matches[k].first;
                double r = rbf(sqrt(matches[k].second));
                if(i == j)
                    r += rbf_smoothing;
                matches[k].second = r;
                sum += r;
            }

            // Normalize the coefficients per row
            for (int k = 0; k < nMatches; k++) {
                int j = matches[k].first;
                double r = matches[k].second / sum;
                if(fabs(r) > matrix_epsilon)
                    tripletList.push_back(T(i,j,r));
            }

            // Add polynomial
            for(int j = 0; j < degree; j++) {
                tripletList.push_back(T(i,numPoints + j,1));
                tripletList.push_back(T(numPoints + j,i,1));
            }
        }
    } else {
        vector<size_t> indices(numNeighbors);
        vector<double> distances(numNeighbors);
        for (int i = 0; i < numPoints; i++) {

            // Perform the k-nearest neighbor search
            query = X.col(i);
            knn(index, numNeighbors, query.data(), &indices[0], &distances[0]);

            // Compute matrix coefficients
            double sum = 0;
            for (int k = 0; k < numNeighbors; k++) {
                int j = indices[k];
                double r = rbf(sqrt(distances[k]));
                if(i == j)
                    r += rbf_smoothing;
                distances[k] = r;
                sum += r;
            }

            // Normalize the coefficients per row
            for (int k = 0; k < numNeighbors; k++) {
                int j = indices[k];
                double r = distances[k] / sum;
                if(fabs(r) > matrix_epsilon)
                    tripletList.push_back(T(i,j,r));
            }

            // Add polynomial
            for(int j = 0; j < degree; j++) {
                tripletList.push_back(T(i,numPoints + j,1));
                tripletList.push_back(T(numPoints + j,i,1));
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
    
    //source/target points and fields
    MatrixXd* points_p;
    MatrixXd* fields_p;
    MatrixXd* target_points_p;
    MatrixXd* target_fields_p;

    //cluster sizes for source and target points
    VectorXi clusterSizes;
    VectorXi target_clusterSizes;

    //cluster center for source points
    MatrixXd clusterCenters;

    //parameters
    int g_numPoints;
    int g_numTargetPoints;
    int numFields;

    //initialize global params
    void init(int nc, int r) {
        numClusters = nc;
        mpi_rank = r;
        points_p = nullptr;
        fields_p = nullptr;
        target_points_p = nullptr;
        target_fields_p = nullptr;
        clusterSizes.resize(numClusters);
        target_clusterSizes.resize(numClusters);
        clusterCenters.resize(numDims,numClusters);

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
    //read/generate target interpolation points
    //
    void read_target_points(MatrixXd*& target_points, std::string tmpl) {
        Timer t;
#if 1
        std::cout << "Creating interpolation grid" << std::endl;
        g_numTargetPoints = n_lat_o*n_lon_o;
        target_points = new MatrixXd(numDims, g_numTargetPoints);

        for(int i = 0; i < n_lat_o; i++) {
            for(int j = 0; j < n_lon_o; j++) {
                 (*target_points)(0, i * n_lon_o + j) =
                     lon_min + (j * (lon_max - lon_min))/ (n_lon_o - 1);
                 (*target_points)(1, i * n_lon_o + j) =
                     lat_min + (i * (lat_max - lat_min))/ (n_lat_o - 1);
            }
        }
#elif 0
        std::cout << "Creating random interpolation grid" << std::endl;
        g_numTargetPoints = n_lat_o*n_lon_o;
        target_points = new MatrixXd(numDims, g_numTargetPoints);

        target_points->setRandom();
        target_points->row(0) = (target_points->row(1).array() + 1.0) /
                                2.0 * (lon_max - lon_min) + lon_min;
        target_points->row(1) = (target_points->row(0).array() + 1.0) /
                                2.0 * (lat_max - lat_min) + lat_min;
#elif 0
        std::cout << "Reading interpolation grid from grib file" << std::endl;
        FILE* fp_s = fopen(tmpl.c_str(), "r");
        if(!fp_s) return;
        int ret;
        long numTargetPoints;

        codes_handle* h = codes_handle_new_from_file(0, fp_s, PRODUCT_GRIB, &ret);

        CODES_CHECK(codes_get_long(h, "numberOfPoints", &numTargetPoints), 0);
        g_numTargetPoints = numTargetPoints;
        target_points = new MatrixXd(numDims, g_numTargetPoints);

        VectorXd lons, lats, values;
        lons.resize(g_numTargetPoints);
        lats.resize(g_numTargetPoints);
        values.resize(g_numTargetPoints);
        CODES_CHECK(codes_grib_get_data(h, lats.data(), lons.data(), values.data()), 0);
        target_points->row(0) = lons;
        target_points->row(1) = lats;

        codes_handle_delete(h);
#else
        std::cout << "Reading interpolation grid from text file" << std::endl;
        g_numTargetPoints = n_lat_o*n_lon_o;
        target_points = new MatrixXd(numDims, g_numTargetPoints);

        FILE* fh = fopen("mrms.txt", "r");
        char buffer[256];
        int idx = 0;
        while(fgets(buffer, 256, fh)) {
           double lat,lon;
           sscanf(buffer, "%lf %lf", &lat, &lon);
           (*target_points)(0, idx) = lon;
           (*target_points)(1, idx) = lat;
           idx++;
        }
#endif
        t.elapsed();
    }

    //
    // Generate random data
    //
    void generate_random_data(
          MatrixXd*& points, MatrixXd*& fields
    ) {
        int numPoints = n_lat_i*n_lon_i;
        numFields = 1;

        // Save global number of points
        g_numPoints = numPoints;

        // Allocate
        points = new MatrixXd(numDims, numPoints);
        fields = new MatrixXd(numFields, numPoints);
        
        // Generate a set of 3D points and associated field values
        for(int i = 0; i < n_lat_i; i++) {
            for(int j = 0; j < n_lon_i; j++) {
                (*points)(0, i * n_lon_i + j) =
                    lon_min + (j * (lon_max - lon_min)) / (n_lon_i - 1);
                (*points)(1, i * n_lon_i + j) =
                    lat_min + (i * (lat_max - lat_min)) / (n_lat_i - 1);
            }
        }

        VectorXd p(numDims);
        for (int i = 0; i < numPoints; i++) {
            for(int j = 0; j < numFields; j++) {
                p(0) = 2 * ((*points)(0,i) - lon_min) / (lon_max - lon_min) - 1;
                p(1) = 2 * ((*points)(1,i) - lat_min) / (lat_max - lat_min) - 1;
                const double x = p(0), y = p(1);
                constexpr double pi = 3.14159265358979323846;
                (*fields)(j, i) = sqrt( exp(x*cos(3*pi*x)) * exp(y*cos(3*pi*y)) )* (j + 1);
            }
        }
    }

    //
    // Read grib file
    //
    void read_grib_file(
          std::string src,
          MatrixXd*& points, MatrixXd*& fields
    ) {
        // Hard code longitude and latitude index
        constexpr int idx_nlat = 990;
        constexpr int idx_elon = 991;
        constexpr int idx_fields[] = {0, 301};

        // Get the number of points
        FILE* fp = fopen(src.c_str(), "r");
        if(!fp) {
            std::cout << "Input file: " << src << " not found!";
            exit(0);
        }

        Timer t;
        size_t numPoints;
        int ret;

        std::cout << "Reading input grib file" << std::endl;
        numFields = sizeof(idx_fields) / sizeof(int);
        {
          codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret);
          CODES_CHECK(codes_get_size(h, "values", &numPoints), 0);
          codes_handle_delete(h);
          rewind(fp);
        }
        g_numPoints = numPoints;

        // Allocate
        points = new MatrixXd(numDims, numPoints);
        fields = new MatrixXd(numFields, numPoints);

        // loop through all fields and read data
        VectorXd values(numPoints);
        int idx = 0, f = 0;
        while (codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret)) {

          if(f < numFields && idx == idx_fields[f]) {
            CODES_CHECK(codes_get_double_array(h, "values",
                        values.data(), &numPoints), 0);

            fields->row(f) = values;
            f++;
          } else if(idx == idx_nlat) {
            CODES_CHECK(codes_get_double_array(h, "values",
                        values.data(), &numPoints), 0);

            points->row(1) = values;
          } else if(idx == idx_elon) {
            CODES_CHECK(codes_get_double_array(h, "values",
                        values.data(), &numPoints), 0);

            points->row(0) = values;
          }

          codes_handle_delete(h);
          idx++;
        }

        t.elapsed();
    }

    //
    // write grib file
    //
    void write_grib_file(std::string tmpl, std::string dst) {
        size_t size = g_numTargetPoints;
        VectorXd values(size);
        Timer t;
#if 1
        std::cout << "Writing input and output fields for plotting" << std::endl;
        FILE* fh = fopen("input.txt", "w");
        for(int i = 0; i < g_numPoints; i++) {
           fprintf(fh, "%.2f %.2f ",
               (*points_p)(0,i),
               (*points_p)(1,i));
           for(int j = 0; j < numFields; j++)
               fprintf(fh, "%.2f ", (*fields_p)(j,i));
           fprintf(fh, "\n");
        }
        fclose(fh);

        fh = fopen("output.txt", "w");
        for(int i = 0; i < g_numTargetPoints; i++) {
           fprintf(fh, "%.2f %.2f ",
             (*target_points_p)(0,i),
             (*target_points_p)(1,i));
           for(int j = 0; j < numFields; j++)
               fprintf(fh, "%.2f ", (*target_fields_p)(j,i));
           fprintf(fh, "\n");
        }
        fclose(fh);
#else
        if(dst.empty() || tmpl.empty())
          return;

        std::cout << "Writing output grib file." << std::endl;

        FILE* fp_s = fopen(tmpl.c_str(), "r");
        if(!fp_s) return;

        int ret, idx = 0;
        while (codes_handle* h = codes_handle_new_from_file(0, fp_s, PRODUCT_GRIB, &ret)) {
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

          values = target_fields_p->row(idx);

          CODES_CHECK(codes_set_double_array(new_h, "values",
                   values.data(), size), 0);

          codes_write_message(new_h, dst.c_str(), idx ? "a" : "w");
          codes_handle_delete(new_h);

          // delete handle
          codes_handle_delete(h);
          idx++;

          if(idx >= numFields) break;
        }
#endif
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
        target_fields_p = new MatrixXd(numFields, numTargetPoints);
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
    void read_and_partition_data(std::string src, std::string tmpl) {

        MatrixXd *points = nullptr, *fields = nullptr, *target_points = nullptr;

        // read source points and fields
#if 1
        generate_random_data(points, fields);
#else
        read_grib_file(src, points, fields);
#endif
        // read target points
        read_target_points(target_points, tmpl);

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
    VectorXd center;
    PointCloud* cloud;
    KDTree* ptree;
    SparseMatrix<double> A;
    RbfSolver solver;
    
    //
    // Scatter data to slave processors
    //
    void scatter() {
        using namespace GlobalData;

        // scatter number of fields
        MPI_Bcast(&GlobalData::numFields, 1, MPI_INT,
                0, MPI_COMM_WORLD);

        // scatter cluster centers
        center.resize(numDims);
        MPI_Scatter(clusterCenters.data(), numDims, MPI_DOUBLE,
                center.data(), numDims, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = target_clusterSizes(i) * numDims;
                offset += counts(i);
            }
        }
        MPI_Gatherv(target_points.data(), numTargetPoints * numDims, MPI_DOUBLE,
                target_points_p ? target_points_p->data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
        //Gather interpolated fields
        if(mpi_rank == 0) {
            int offset = 0;
            for(int i = 0; i < numClusters; i++) {
                offsets(i) = offset;
                counts(i) = target_clusterSizes(i) * numFields;
                offset += counts(i);
            }
        }
        MPI_Gatherv(target_fields.data(), numTargetPoints * numFields, MPI_DOUBLE,
                target_fields_p ? target_fields_p->data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        A.resize(numPoints + degree, numPoints + degree);
        rbf_build(*ptree, points, numPoints, solver, A);
    }
    //
    // Interpolate each field using parameteric RBF
    //
    void solve_rbf() {
        using namespace GlobalData;
        VectorXd F(numPoints+degree);
        MatrixXd W(numPoints+degree, numFields);

        target_fields.setZero();

        std::cout << "===========================" << std::endl;

        VectorXd d(numDims);
        double avg_radius = 0;
        if(!non_parametric) {
            //compute weights for each field
            VectorXd Wf(numPoints+degree);
            std::cout << "Computing weights for all fields" << std::endl;
            F.setZero();
            Wf.setZero();
            for(int f = 0; f < numFields; f++) {
                F.segment(0,numPoints) = fields.row(f);
                rbf_solve(solver, F, Wf);
                W.col(f) = Wf;
            }
            //compute average radius of cluster for weighing purposes
            for(int i = 0; i < numPoints; i++) {
                d = points.col(i) - center;
                avg_radius += sqrt(d.dot(d));
            }
            avg_radius /= numPoints;
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

                // compute blending factor
                double blend;
                if(blend_interp) {
                    d = target_points.col(i) - center;
                    double radius = sqrt(d.dot(d));
                    constexpr double r_max = 1.2, r_min = 0.5;
                    if(radius > r_max * avg_radius)
                        blend = 0;
                    else if(radius < r_min * avg_radius)
                        blend = 1;
                    else
                        blend = (r_max * avg_radius - radius) / ((r_max - r_min) * avg_radius);
                } else {
                    if(non_parametric)
                        blend = 0;
                    else
                        blend = 1;
                }

                // interpolate
                double sum = 0;
                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = rbf(sqrt(matches[k].second));
                    matches[k].second = r;
                    sum += r;
                }

                for (int k = 0; k < nMatches; k++)
                    matches[k].second /= sum;

                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = matches[k].second;

                    if(!non_parametric && (blend > 0)) {
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += W(j, f) * r;
                        if(k == 0) {
                            for (int j = 0; j < degree; j++)
                                for(int f = 0; f < numFields; f++)
                                    target_fields(f, i) += W(numPoints + j, f);
                        }
                    }

                }
                target_fields.col(i) *= blend;
                sum /= (1 - blend);

                // add non-parameteric interp
                if(non_parametric || (blend < 1)) {
                    for (int k = 0; k < nMatches; k++) {
                        int j = matches[k].first;
                        double r = rbf(sqrt(matches[k].second));
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += fields(f, j) * (r / std::max(sum,1e-6));
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

                // compute blending factor
                double blend;
                if(blend_interp) {
                    d = target_points.col(i) - center;
                    double radius = sqrt(d.dot(d));
                    constexpr double r_max = 1.2, r_min = 0.5;
                    if(radius > r_max * avg_radius)
                        blend = 0;
                    else if(radius < r_min * avg_radius)
                        blend = 1;
                    else
                        blend = (r_max * avg_radius - radius) / ((r_max - r_min) * avg_radius);
                } else {
                    if(non_parametric)
                        blend = 0;
                    else
                        blend = 1;
                }

                // interpolate
                double sum = 0;
                for (int k = 0; k < numNeighbors; k++) {
                    int j = indices[k];
                    double r = rbf(sqrt(distances[k]));
                    distances[k] = r;
                    sum += r;
                }

                for (int k = 0; k < numNeighbors; k++)
                    distances[k] /= sum;

                for (int k = 0; k < numNeighbors; k++) {
                    int j = indices[k];
                    double r = distances[k];

                    if(!non_parametric && (blend > 0)) {
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += W(j, f) * r;
                        if(k == 0) {
                            for (int j = 0; j < degree; j++)
                                for(int f = 0; f < numFields; f++)
                                    target_fields(f, i) += W(numPoints + j, f);
                        }
                    }

                }

                target_fields.col(i) *= blend;
                sum /= (1 - blend);

                // add non-parameteric interp
                if(non_parametric || (blend < 1)) {
                    for (int k = 0; k < numNeighbors; k++) {
                        int j = indices[k];
                        double r = rbf(sqrt(distances[k]));
                        for(int f = 0; f < numFields; f++)
                            target_fields(f, i) += fields(f, j) * (r / std::max(sum,1e-6));
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
            clusterAssignments, clusterSizes, clusterCenters,
            (GlobalData::numClusters > 1) ? true : false);
    
    // Initialize subcluster source points & fields
    for(int i = 0; i < numClusters; i++) {
        ClusterData& cd = subclusters[i];
        cd.numPoints = clusterSizes(i);
        cd.center = clusterCenters.col(i);
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
    VectorXd d(numDims);
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
            parent.target_fields(f,i) = cd.target_fields(f,idx);
        target_cidx[owner]++;
    } 
}

/*****************
 * Test driver
 *****************/

void usage() {
    std::cout << "usage: ./interp [-h] --src SRC --dst DST" << std::endl << std::endl
              << "Interpolate fields in a grib2 file onto another grid or scattered obs locations." << std::endl << std::endl
              << "arguments:" << std::endl
              << "  -h, --help      show this help message and exit" << std::endl
              << "  -i, --input     grib file containing fields to interpolate" << std::endl
              << "  -o, --output    output grib file contiainig result of interpolation" << std::endl
              << "  -t, --template  template grib file that the output grib file is based on" << std::endl;
}

int main(int argc, char** argv) {

    // Initialize MPI
    int nprocs, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    //
    // Process command line options
    //
    std::string src, dst = "out.grib2", tmpl;
    if(mpi_rank == 0) {
        std::vector<std::string> args(argv + 1, argv + argc);
        for (auto it = args.begin(); it != args.end(); ++it) {
            if(*it == "-h" || *it == "--help") {
                usage();
                exit(0);
            } else if(*it == "-i" || *it == "--input") {
                src = *++it;
            } else if(*it == "-o" || *it == "--output") {
                dst = *++it;
            } else if(*it == "-t" || *it == "--template") {
                tmpl = *++it;
            }
        }
    }

    //
    // Cluster and decompose data
    //

    GlobalData::init(nprocs, mpi_rank);
    if(mpi_rank == 0)
        GlobalData::read_and_partition_data(src, tmpl);

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
#pragma omp parallel for
        for(int i = 0; i < numClustersPerRank; i++)
            child_clusters[i].build_and_solve();

        //merge clusters
        merge_cluster(parent_cluster, numClustersPerRank,
                      child_clusters, target_clusterAssignments);
    }

    // Gather result from slave nodes
    parent_cluster.gather();

    // Write result to a grib file
    if(mpi_rank == 0)
        GlobalData::write_grib_file(tmpl, dst);

    //
    // Finalize MPI
    //

    MPI_Finalize();

    return 0;
}

