#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <chrono>
#include "knn/nanoflann.hpp"

using namespace std;
using namespace Eigen;

constexpr int numDims = 2;

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
    auto start = chrono::high_resolution_clock::now();

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

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Finished in " << duration.count() << " millisecs." << std::endl;
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
double rbf(double r) {
    return exp(-1*r*r);
}

//
// Build sparse interpolation matrix and LU decompose it
//
void rbf_build(const KDTree& index, const MatrixXd& X,
        const int numPoints, const int numNeighbors,
        RbfSolver& solver, SparseMatrix<double>& A
        ) {

    //
    // Build sparse Matrix of radial basis function evaluations
    //
    std::cout << "Constructing interpolation matrix ..." << std::endl;
    auto start = chrono::high_resolution_clock::now();

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(numPoints * numNeighbors);

    vector<size_t> indices(numNeighbors);
    vector<double> distances(numNeighbors);
    VectorXd query(numDims);
    for (int i = 0; i < numPoints; i++) {

        // Perform the k-nearest neighbor search
        query = X.col(i);
        knn(index, numNeighbors, query.data(), &indices[0], &distances[0]);

        // Add matrix coefficients
        for (int k = 0; k < numNeighbors; k++) {
            int j = indices[k];
            double r = rbf(sqrt(distances[k]));
            if(fabs(r) > 1e-5)
                tripletList.push_back(T(i,j,r));
        }
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Finished in " << duration.count() << " millisecs." << std::endl;

    //
    // LU decomposition of the interpolation matrix
    //
    std::cout << "Started factorization ..." << std::endl;
    start = stop;

    solver.compute(A);

    if(solver.info() != Success) {
        std::cout << "Factorization failed." << std::endl;
        exit(0);
    }

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Finished in " << duration.count() << " millisecs." << std::endl;
}

//
// Build symmetric sparse interpolation matrix and LU decompose it
// Consider neighbors of specific distance
//
void rbf_build_symm(const KDTree& index, const MatrixXd& X,
        const int numPoints, const int numNeighbors, const double cutoff_radius,
        RbfSolver& solver, SparseMatrix<double>& A
        ) {

    //
    // Build sparse Matrix of radial basis function evaluations
    //
    std::cout << "Constructing interpolation matrix ..." << std::endl;
    auto start = chrono::high_resolution_clock::now();

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(numPoints * numNeighbors);

    VectorXd query(numDims);
    std::vector<nanoflann::ResultItem<unsigned, double>> matches;
    for (int i = 0; i < numPoints; i++) {

        // Perform a radius search
        query = X.col(i);
        unsigned nMatches = knn_radius(index, cutoff_radius, query.data(), matches);

        // Add matrix coefficients
        for (int k = 0; k < nMatches; k++) {
            int j = matches[k].first;
            double r = rbf(sqrt(matches[k].second));
            if(fabs(r) > 1e-5)
                tripletList.push_back(T(i,j,r));
        }

    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Finished in " << duration.count() << " millisecs." << std::endl;

    //
    // LU decomposition of the interpolation matrix
    //
    std::cout << "Started factorization ..." << std::endl;
    start = stop;

    solver.compute(A);

    if(solver.info() != Success) {
        std::cout << "Factorization failed." << std::endl;
        exit(0);
    }

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Finished in " << duration.count() << " millisecs." << std::endl;
}

//
// Solve Ax=b from solver that already has LU decomposed matrix
// Iterative solvers don't need LU decompostion to happen first
//
void rbf_solve(RbfSolver& solver, const VectorXd& F, VectorXd& C) {
    std::cout << "Started solve ..." << std::endl;
    auto start = chrono::high_resolution_clock::now();

    C = solver.solve(F);

#if 0
    std::cout << "----------------------------------" << std::endl;
    std::cout << "numb iterations: " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error()      << std::endl;
    std::cout << "----------------------------------" << std::endl;
#endif

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Finished in " << duration.count() << " millisecs." << std::endl;
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
    const int g_numPoints = 10000; //1799*1059;
    const int g_numTargetPoints = 4;
    const int numNeighbors = 3;
    const int numFields = 1;
    const int numClustersPerRank = 1;
    const bool use_cutoff_radius = false;
    const double cutoff_radius = 0.5;
    const bool non_parametric = true;

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
                      << "numPoints: " << g_numPoints << std::endl
                      << "numTargetPoints: " << g_numTargetPoints << std::endl
                      << "numNeighbors: " << numNeighbors << std::endl
                      << "numFields: " << numFields << std::endl
                      << "numClustersPerRank: " << numClustersPerRank << std::endl
                      << "use_cutoff_radius: " << (use_cutoff_radius ? "true" : "false") << std::endl
                      << "cutoff_radius: " << cutoff_radius << std::endl
                      << "non_parametric: " << (non_parametric ? "true" : "false") << std::endl
                      << "=====================" << std::endl;
        }
    }
    //
    // Read and partition data on master node. Assumes node is big enough
    // to store and process the data. Currently it generates random data.
    //
    void read_and_partition_data() {
        // coordinates and fields to interpolate
        int numPoints = g_numPoints;
        int numTargetPoints = g_numTargetPoints;
        
        MatrixXd points(numDims,numPoints);
        MatrixXd fields(numFields,numPoints);
        
        VectorXi clusterAssignments(numPoints);
        MatrixXd clusterCenters(numDims,numClusters);
        
        // Generate a set of random 3D points and associated field values
        points.setRandom();
        clusterAssignments.setZero();
        for (int i = 0; i < numPoints; i++) {
            for(int j = 0; j < numFields; j++) {
                const double x = points.col(i).norm();
                constexpr double pi = 3.14159265358979323846;
                // wiki example function for rbf interpolation
                fields(j, i) = exp(x*cos(3*pi*x)) * (j + 1);
            }
        }
#if 0
        std::cout << "===================" << std::endl;
        std::cout << points.transpose() << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << fields.row(0).transpose() << std::endl;
        std::cout << "===================" << std::endl;
#endif
        // Initialize the target points
        MatrixXd target_points(numDims, numTargetPoints);
        VectorXi target_clusterAssignments(numTargetPoints);

        //target_points = points.block(0,0,numDims,numTargetPoints);
        target_points.setRandom();
        
        // Partition the points into N clusters using k-means clustering
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

        // print interpolated fields with associated coordinates
        // Note that the order of points is changed.
        if(mpi_rank == 0) {
            std::cout << "===================" << std::endl;
            std::cout << interp_target_points_p->transpose() << std::endl;
            std::cout << "===================" << std::endl;
            std::cout << interp_target_fields_p->transpose() << std::endl;
        }
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
        using namespace GlobalData;
        if(non_parametric)
            return;
        A.resize(numPoints, numPoints);
        if(use_cutoff_radius)
            rbf_build_symm(*ptree, points, numPoints, numNeighbors, cutoff_radius, solver, A);
        else
            rbf_build(*ptree, points, numPoints, numNeighbors, solver, A);
    }
    //
    // Interpolate each field using parameteric RBF
    //
    void solve_rbf() {
        using namespace GlobalData;
        VectorXd F(numPoints);
        VectorXd W(numPoints);

        target_fields.setZero();

        for(int f = 0; f < numFields; f++) {

            std::cout << "==== Interpolating field " << f << " ====" << std::endl;

            //solve weights for this field
            F = fields.row(f);
            if(!non_parametric)
                rbf_solve(solver, F, W);

            //interpolate for target fields
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
                        if(!non_parametric)
                            target_fields(f, i) += W(j) * r;
                        else
                            sum += r;
                    }
                    if(non_parametric)
                    {
                        for (int k = 0; k < nMatches; k++) {
                            int j = matches[k].first;
                            double r = rbf(sqrt(matches[k].second));
                            target_fields(f, i) += F(j) * (r / sum);
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
                        if(!non_parametric)
                            target_fields(f, i) += W(j) * r;
                        else
                            sum += r;
                    }
                    if(non_parametric)
                    {
                        for (int k = 0; k < numNeighbors; k++) {
                            int j = indices[k];
                            double r = rbf(sqrt(distances[k]));
                            target_fields(f, i) += F(j) * (r / sum);
                        }
                    }
                }
            }
        }
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
    using GlobalData::numClustersPerRank;

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

    //
    // Finalize MPI
    //

    MPI_Finalize();

    return 0;
}

