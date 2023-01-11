#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace Eigen;

/*****************
 * Clustering
 *****************/

// Function for partitioning a set of 3D points into N clusters using the k-means clustering algorithm
void kMeansClustering(const MatrixXd& points, int numPoints, int numClusters,
        VectorXi& clusterAssignments, VectorXi& clusterSizes, MatrixXd& clusterCenters) {

    // Initialize the cluster centers
    for (int i = 0; i < numClusters; i++) {
        clusterCenters(0,i) = points(0,i);
        clusterCenters(1,i) = points(1,i);
        clusterCenters(2,i) = points(2,i);
    }

    // Perform k-means clustering until the cluster assignments stop changing
    MatrixXd newClusterCenters(3, numClusters);

    unsigned int iter = 0;
    bool converged = false;
    while (!converged) {
        // Update the cluster assignments
        converged = true;
        for (int i = 0; i < numPoints; i++) {
            int closestCluster = -1;
            double minDistance = std::numeric_limits<double>::max();
            for (int j = 0; j < numClusters; j++) {
                double dx = points(0,i) - clusterCenters(0,j);
                double dy = points(1,i) - clusterCenters(1,j);
                double dz = points(2,i) - clusterCenters(2,j);
                double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
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
        newClusterCenters.setZero(3,numClusters);
        clusterSizes.setZero(numClusters);

        for (int i = 0; i < numPoints; i++) {
            int cluster = clusterAssignments(i);
            newClusterCenters(0,cluster) += points(0,i);
            newClusterCenters(1,cluster) += points(1,i);
            newClusterCenters(2,cluster) += points(2,i);
            clusterSizes(cluster) = clusterSizes(cluster) + 1;
        }
        for (int i = 0; i < numClusters; i++) {
            if (clusterSizes(i) > 0) {
                clusterCenters(0,i) = newClusterCenters(0,i) / clusterSizes(i);
                clusterCenters(1,i) = newClusterCenters(1,i) / clusterSizes(i);
                clusterCenters(2,i) = newClusterCenters(2,i) / clusterSizes(i);
            }
        }
        iter++;
    }
    //final cluster sizes
    std::cout << "Cluster sizes" << std::endl;
    for (int i = 0; i < numClusters; i++) {
        std::cout << "[" << i << "] " << clusterSizes(i) << std::endl;
    }
}

/*****************
 * RBF interpolation
 *****************/

// Radial basis function
double rbf(double r) {
  return exp(-r*r);
}

// Solve for RBF coefficients
void rbf_solve(const MatrixXd& X, const VectorXd& F,
              const int numPoints, const int numNeighbors, VectorXd& C) {

  //
  // Sparse Matrix of radial basis function evaluations
  //
  std::cout << "Constructing interpolation matrix ..." << std::endl;
  auto start = chrono::high_resolution_clock::now();

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(numPoints * numNeighbors);

  SparseMatrix<double> A(numPoints, numPoints);
  vector<pair<double, int>> neighbors(numPoints);
  for (int i = 0; i < numPoints; i++) {
    for (int j = 0; j < numPoints; j++) {
      double dx = X(0, i) - X(0, j);
      double dy = X(1, i) - X(1, j);
      double dz = X(2, i) - X(2, j);
      double r = sqrt(dx*dx + dy*dy + dz*dz);
      neighbors[j] = make_pair(r, j);
    }

    std::partial_sort(neighbors.begin(), neighbors.begin() + numNeighbors, neighbors.end());

    for (int k = 0; k < numNeighbors; k++) {
      int j = neighbors[k].second;
      double r = neighbors[k].first;
      tripletList.push_back(T(i,j,rbf(r)));
    }
  }

  A.setFromTriplets(tripletList.begin(), tripletList.end());

  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
  std::cout << "Finished in " << duration.count() << " secs." << std::endl;

  //
  // Solve for the weights x of the Ax=B equations
  //
  std::cout << "Started solve ..." << std::endl;
  start = stop;

  //BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double,int>> solver;
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(A);
  C = solver.solve(F);

  //std::cout << "#iterations:     " << solver.iterations() << std::endl;
  //std::cout << "estimated error: " << solver.error()      << std::endl;

  stop = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<chrono::seconds>(stop - start);
  std::cout << "Finished in " << duration.count() << " secs." << std::endl;
}

/*****************
 * Test
 *****************/

int main(int argc, char** argv) {

    // Initialize MPI
    int nprocs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int numFields = 1;
    const int g_numPoints = 20000; //1000000;
    const int g_numTargetPoints = 5; //100000 
    const int numNeighbors = 20;

    // Cluster and decompose on rank 0
    const int numClusters = nprocs;

    MatrixXd* points_p = nullptr;
    MatrixXd* fields_p = nullptr;
    MatrixXd* target_points_p = nullptr;

    VectorXi clusterSizes(numClusters);
    VectorXi target_clusterSizes(numClusters);

    /*****************************
     * Cluster and decompose data
     *****************************/
    if(rank == 0)
    {
        const int numPoints = g_numPoints;
        const int numTargetPoints = g_numTargetPoints;
 
        // coordinates and fields to interpolate
        MatrixXd points(3,numPoints);
        MatrixXd fields(numFields,numPoints);

        VectorXi clusterAssignments(numPoints);
        MatrixXd clusterCenters(3,numClusters);

        // Generate a set of random 3D points and associated field values
        for (int i = 0; i < numPoints; i++) {
            points(0, i) = rand() / (double)RAND_MAX;
            points(1, i) = rand() / (double)RAND_MAX;
            points(2, i) = rand() / (double)RAND_MAX;
            clusterAssignments(i) = rand() % numClusters;
            for(int j = 0; j < numFields; j++) {
                double x = points(0, i);
                double y = points(1, i);
                double z = points(2, i);
                fields(j, i) = x * y *z; //rand() / (double)RAND_MAX;
            }
        }

        // Initialize the target points
        MatrixXd target_points(3, numTargetPoints);
        VectorXi target_clusterAssignments(numTargetPoints);
        for (int i = 0; i < numTargetPoints; i++) {
            //*/
            //use same data as sources for testing
            //should give identical results as the source
            target_points(0, i) = points(0, i);
            target_points(1, i) = points(1, i);
            target_points(2, i) = points(2, i);
            /*/
            //use random 3D points
            target_points(0, i) = rand() / (double)RAND_MAX;
            target_points(1, i) = rand() / (double)RAND_MAX;
            target_points(2, i) = rand() / (double)RAND_MAX;
            //*/
        }

        // Partition the points into N clusters using k-means clustering
        kMeansClustering(points, numPoints, numClusters,
                        clusterAssignments, clusterSizes, clusterCenters);

        // Sort points and fields
        points_p = new MatrixXd(3, numPoints);
        fields_p = new MatrixXd(numFields, numPoints);
        MatrixXd& sorted_points = *points_p;
        MatrixXd& sorted_fields = *fields_p;

        int idx = 0;
        for (int i = 0; i < numClusters; i++) {
            for(int j = 0; j < numPoints; j++) {
                if(clusterAssignments(j) != i)
                    continue;
                sorted_points(0, idx) = points(0, j);
                sorted_points(1, idx) = points(1, j);
                sorted_points(2, idx) = points(2, j);
                for(int k = 0; k < numFields; k++)
                    sorted_fields(k, idx) = fields(k, j);
                idx++;
            }
        }

        // Partition target points
        target_clusterSizes.setZero(numClusters);
        for(int i = 0; i < numTargetPoints; i++) {
            int closestCluster = -1;
            double minDistance = std::numeric_limits<double>::max();
            for (int j = 0; j < numClusters; j++) {
                double dx = target_points(0, i) - clusterCenters(0, j);
                double dy = target_points(1, i) - clusterCenters(1, j);
                double dz = target_points(2, i) - clusterCenters(2, j);
                double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCluster = j;
                }
            }
            target_clusterAssignments(i) = closestCluster;
            target_clusterSizes(closestCluster)++;
        }

        // Sort target points
        target_points_p = new MatrixXd(3, numTargetPoints);
        MatrixXd& sorted_target_points = *target_points_p;

        idx = 0;
        for (int i = 0; i < numClusters; i++) {
            for(int j = 0; j < numTargetPoints; j++) {
                if(target_clusterAssignments(j) != i)
                    continue;
                sorted_target_points(0, idx) = target_points(0, j);
                sorted_target_points(1, idx) = target_points(1, j);
                sorted_target_points(2, idx) = target_points(2, j);
                idx++;
            }
        }

    }

    /*****************************
     * Scatter data
     *****************************/
    // Get the local points and fields assigned to this rank
    int numPoints;
    MPI_Scatter(clusterSizes.data(), 1, MPI_INT,
                &numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MatrixXd points(3, numPoints);
    MatrixXd fields(numFields, numPoints);
    VectorXi offsets(numClusters);
    VectorXi counts(numClusters);

    //scatter points
    if(rank == 0) {
        int offset = 0;
        for(int i = 0; i < numClusters; i++) {
            offsets(i) = offset;
            counts(i) = clusterSizes(i) * 3;
            offset += counts(i);
        }
    }
    MPI_Scatterv(points_p ? points_p->data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                points.data(), numPoints * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //scatter fields
    if(rank == 0) {
        int offset = 0;
        for(int i = 0; i < numClusters; i++) {
            offsets(i) = offset;
            counts(i) = clusterSizes(i) * numFields;
            offset += counts(i);
        }
    }
    MPI_Scatterv(fields_p ? fields_p->data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                fields.data(), numPoints * numFields, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Get local target points assinged to this rank
    int numTargetPoints;
    MPI_Scatter(target_clusterSizes.data(), 1, MPI_INT,
                &numTargetPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MatrixXd target_points(3, numTargetPoints);
    MatrixXd target_fields(numFields, numTargetPoints);

    //Scatter target points
    if(rank == 0) {
        int offset = 0;
        for(int i = 0; i < numClusters; i++) {
            offsets(i) = offset;
            counts(i) = target_clusterSizes(i) * 3;
            offset += counts(i);
        }
    }

    MPI_Scatterv(target_points_p ? target_points_p->data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                target_points.data(), numTargetPoints * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*****************************
     * Interpolate
     *****************************/

    //Interpolate on target points
    VectorXd C(numPoints);
    target_fields.setZero();

    for(int i = 0; i < numFields; i++) {

        //solve for rbf coefficients for the given field
        rbf_solve(points, fields.row(i), numPoints, numNeighbors, C);

        //interpolate for target fields
        vector<pair<double, int>> neighbors(numPoints);
        for(int j = 0; j < numTargetPoints; j++) {
            for (int k = 0; k < numPoints; k++) {
                double dx = target_points(0, j) - points(0, k);
                double dy = target_points(1, j) - points(1, k);
                double dz = target_points(2, j) - points(2, k);
                double r = sqrt(dx*dx + dy*dy + dz*dz);
                neighbors[k] = make_pair(r, k);
            }

            std::partial_sort(neighbors.begin(), neighbors.begin() + numNeighbors, neighbors.end());

            for (int m = 0; m < numNeighbors; m++) {
              int k = neighbors[m].second;
              double r = neighbors[m].first;
              target_fields(i, j) += C(k) * rbf(r);
            }
        }
    }

    /*****************************
     * Gather target points
     *****************************/

    //Gather target points
    MatrixXd* all_target_points = nullptr;
    if(rank == 0)
        all_target_points = new MatrixXd(3, g_numTargetPoints);

    if(rank == 0) {
        int offset = 0;
        for(int i = 0; i < numClusters; i++) {
            offsets(i) = offset;
            counts(i) = clusterSizes(i) * 3;
            offset += counts(i);
        }
    }
    MPI_Gatherv(target_points.data(), numTargetPoints * 3, MPI_DOUBLE,
                all_target_points ? all_target_points->data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Gather interpolated fields
    MatrixXd* all_target_fields = nullptr;
    if(rank == 0)
        all_target_fields = new MatrixXd(numFields, g_numTargetPoints);

    if(rank == 0) {
        int offset = 0;
        for(int i = 0; i < numClusters; i++) {
            offsets(i) = offset;
            counts(i) = clusterSizes(i) * numFields;
            offset += counts(i);
        }
    }
    MPI_Gatherv(target_fields.data(), numTargetPoints * numFields, MPI_DOUBLE,
                all_target_fields ? all_target_fields->data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*****************************
     * print
     *****************************/

    //print
    if(rank == 0) {
        std::cout << "===================" << std::endl;
        std::cout << all_target_points->transpose() << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << all_target_fields->transpose() << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}


