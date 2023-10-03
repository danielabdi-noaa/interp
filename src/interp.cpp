#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <chrono>
#if ENABLE_GRIB
#include <eccodes.h>
#endif
#include <omp.h>

#include "knn/nanoflann.hpp"

using namespace std;
using namespace Eigen;

/***************************************************
 * Parameters for the interpolation
 **************************************************/

// number of dimesnions 1D,2D,3D are supported
#ifdef ENABLE_3D
constexpr int numDims = 3;
#else
constexpr int numDims = 2;
#endif

// Rbf shape function can be computed approximately from
// the average distance between points
//     rbfShape  = 0.8 / average_distance
// If D is width of the domain
//     rbfShape = 0.8 / (D / npoints^(1/numDims))
static double rbfShapeGlobal;

// Number of neighbors to consider for interpolation
static int numNeighbors;
static int numNeighborsInterp;

// Cutoff radius for nearest neighbor search
static bool useCutoffRadius = false;
static double cutoffRadius = 0.08;
static double cutoffRadiusInterp = 0.64;

// Rbf smoothing factor, often set to 0 for interpolation
// but can be set to positive value for noisy data.
static double rbfSmoothing;

// Number of monomials to consider
// Currently only monomials=1 is supported
static int monomials;

// Use test function for field initialization
static bool useTestField;

// Average duplicate fields
static bool averageDuplicates;
constexpr double dupsTolerance = 1e-6;

// mutex for cout 
static std::mutex cout_mutex;

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
        cout_mutex.lock();
        std::cout << "Finished in " << duration << " secs." << std::endl;
        cout_mutex.unlock();
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
        VectorXi& clusterAssignments, VectorXi& clusterSizes, MatrixXd& clusterCenters
    ) {

    std::cout << "Clustering point clouds into " << numClusters << " clusters" << std::endl;
    Timer t;
    VectorXd d(numDims);

    //
    // Initialize the cluster centers using kMeans++
    //
    clusterCenters.col(0) = points.col(rand() % numPoints);
    for (int i = 1; i < numClusters; i++) {
        double maxDistance = 0.0;
        int newCenter;
#pragma omp parallel for private(d)
        for (int j = 0; j < numPoints; j++) {
            double minDistance = std::numeric_limits<double>::max();
            for (int m = 0; m < i; m++) {
                d = points.col(j) - clusterCenters.col(m);
                double distance = d.dot(d);
                if (distance < minDistance)
                    minDistance = distance;
            }
            if (minDistance > maxDistance) {
                maxDistance = minDistance;
                newCenter = j;
            }
        }
        clusterCenters.col(i) = points.col(newCenter);
    }

    // Perform k-means clustering until the cluster assignments stop changing
    MatrixXd sumClusterCenters(numDims, numClusters);
    bool converged = false;
    int iterations = 0;
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
#pragma omp parallel
        {
            MatrixXd sumClusterCenters_(numDims, numClusters);
            VectorXi clusterSizes_(numClusters);
            sumClusterCenters_.setZero(numDims,numClusters);
            clusterSizes_.setZero(numClusters);
#pragma omp for nowait
            for (int i = 0; i < numPoints; i++) {
                int cluster = clusterAssignments(i);
                sumClusterCenters_.col(cluster) += points.col(i);
                clusterSizes_(cluster)++;
            }
#pragma omp critical
            {
                sumClusterCenters += sumClusterCenters_;
                clusterSizes += clusterSizes_;
            }
        }

#pragma omp parallel for
        for (int i = 0; i < numClusters; i++) {
            if (clusterSizes(i) > 0)
                clusterCenters.col(i) = sumClusterCenters.col(i) / clusterSizes(i);
        }


        iterations++;
    }

    //Print final cluster sizes
    std::cout << "Completed " << iterations << " iterations." << std::endl;
    for (int i = 0; i < numClusters; i++) {
        std::cout << "cluster " << i << " with centroid ("
            << clusterCenters.col(i).transpose() << ") and "
            << clusterSizes(i) << " points" << std::endl;
    }

    t.elapsed();
}

/**************************************************************
 * Pick a linear equations systems solver from Eigen (direct/iterative)
 **************************************************************/

#ifdef USE_ITERATIVE
typedef BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double,int>> RbfSolver;
#else
typedef SparseLU<SparseMatrix<double>> RbfSolver;
#endif

/**************************************************************
 * RBF interpolation using nearest neighbor search
 **************************************************************/

//
// Radial basis function
//
enum rbf_t {
    GAUSSIAN, MULTIQUADRIC, INVERSE_MULTIQUADRIC, THIN_SPLINE, COMPACT_GAUSSIAN, IDW
};

template<rbf_t type = GAUSSIAN>
double rbf(double r_, double rbfShape) {
    double r = (rbfShape * r_);
    if constexpr(type == GAUSSIAN) {
        return exp(-r*r);
    } else if constexpr(type == MULTIQUADRIC) {
        return sqrt(1 + r*r);
    } else if constexpr(type == INVERSE_MULTIQUADRIC) {
        return 1 / sqrt(1 + r*r);
    } else if constexpr(type == THIN_SPLINE) {
        return r*r*log(r+1e-5);
    } else if constexpr(type == COMPACT_GAUSSIAN) {
        if(r < 1.0 / rbfShape)
           return exp(-1 / (1 - r*r));
        else
           return 0;
    } else if constexpr(type == IDW) {
        return 1.0 / pow(r,3);
    }
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
    bool useHeader;
    int numClustersPerRank;
    std::vector<int> field_indices;

    //input/output grid dimensions used for testing purposes.
    //if you read grid from grib2 files, these are ignored.
    constexpr double lat_min = -37.0;
    constexpr double lat_max =  37.0;
    constexpr double lon_min =  61.0;
    constexpr double lon_max = 299.0;
    constexpr double hgt_min =   0.0;
    constexpr double hgt_max =   2.0;
    constexpr int n_lon_i = 120;
    constexpr int n_lat_i =  40;
    constexpr int n_hgt_i =  (numDims == 3) ? 2 : 1;
    constexpr int n_lon_o = 240;
    constexpr int n_lat_o =  80;
    constexpr int n_hgt_o =  (numDims == 3) ? 3 : 1;

    //random points are structured/unstructured
    constexpr bool source_is_structured = true;
    constexpr bool target_is_structured = true;

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
    }
    //
    //read/generate target interpolation points
    //
    void read_input_file(
          std::string src,
          MatrixXd*& points,
          MatrixXd*& fields,
          bool is_target);

    void read_target_points(MatrixXd*& target_points, std::string tmpl) {
        Timer t;

        if(!tmpl.empty()) {
#if ENABLE_GRIB
            if(tmpl.find("grib") != string::npos) {
                std::cout << "Reading interpolation grid from grib file" << std::endl;
                FILE* fp = fopen(tmpl.c_str(), "r");
                if(!fp) return;
                int ret;
                long numTargetPoints;

                codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret);

                CODES_CHECK(codes_get_long(h, "numberOfPoints", &numTargetPoints), 0);
                g_numTargetPoints = numTargetPoints;
                target_points = new MatrixXd(numDims, g_numTargetPoints);

                VectorXd lons, lats, values;
                lons.resize(g_numTargetPoints);
                lats.resize(g_numTargetPoints);
                values.resize(g_numTargetPoints);
                CODES_CHECK(codes_grib_get_data(h, lats.data(), lons.data(), values.data()), 0);
                lons = (lons.array() < 0).select(lons.array() + 360, lons.array());
                target_points->row(0) = lons;
                target_points->row(1) = lats;

                codes_handle_delete(h);
            } else
#endif
            {
                std::cout << "Reading interpolation grid" << std::endl;
                read_input_file(tmpl, target_points, target_points, true);
                g_numTargetPoints = target_points->cols();
            }
        } else {
            g_numTargetPoints = n_lat_o*n_lon_o*n_hgt_o;

            target_points = new MatrixXd(numDims, g_numTargetPoints);

            if(target_is_structured) {
                std::cout << "Creating interpolation grid" << std::endl;
                for(int i = 0; i < n_lat_o; i++) {
                    for(int j = 0; j < n_lon_o; j++) {
                        for(int k = 0; k < n_hgt_o; k++) {
                           int idx = i * n_lon_o * n_hgt_o + j * n_hgt_o + k;
                           double x, y;
                           x = lon_min + (j * (lon_max - lon_min)) / (n_lon_o - 1);
                           y = lat_min + (i * (lat_max - lat_min)) / (n_lat_o - 1);
                           (*target_points)(0, idx) = x;
                           (*target_points)(1, idx) = y;
                           if constexpr (numDims >= 3) {
                               constexpr double pi = 3.14159265358979323846;
                               x = 2 * (x - lon_min) / (lon_max - lon_min) - 1;
                               y = 2 * (y - lat_min) / (lat_max - lat_min) - 1;
                               double v = sqrt( exp(x*cos(3*pi*x)) * exp(y*cos(3*pi*y)) ) +
                                          (hgt_min + k * (hgt_max - hgt_min) / n_hgt_o);
                               (*target_points)(2, idx) = v;
                           }
                        }
                    }
                }
            } else {
                std::cout << "Creating random scattered interpolation points" << std::endl;
                target_points->setRandom();
                target_points->row(0) = (target_points->row(0).array() + 1.0) /
                                        2.0 * (lon_max - lon_min) + lon_min;
                target_points->row(1) = (target_points->row(1).array() + 1.0) /
                                        2.0 * (lat_max - lat_min) + lat_min;
                if constexpr (numDims >= 3)
                    target_points->row(2) = (target_points->row(2).array() + 1.0) /
                                        2.0 * (hgt_max - hgt_min) + hgt_min;
            }
        }

        t.elapsed();
    }

    //
    // Compute test field values at given locations
    //
    void compute_test_fields(const MatrixXd& points, MatrixXd& fields) {
        VectorXd p(numDims);
        VectorXd maxC = points.rowwise().maxCoeff();
        VectorXd minC = points.rowwise().minCoeff();

        int numPoints = fields.cols(), numFields = fields.rows();

        for (int i = 0; i < numPoints; i++) {
            for(int j = 0; j < numFields; j++) {
                p(0) = 2 * (points(0,i) - minC(0)) / (maxC(0) - minC(0)) - 1;
                p(1) = 2 * (points(1,i) - minC(1)) / (maxC(1) - minC(1)) - 1;
                if constexpr (numDims >= 3)
                    p(2) = 2 * (points(2,i) - minC(2)) / (maxC(2) - minC(2)) - 1;

                const double x = p(0), y = p(1);
                constexpr double pi = 3.14159265358979323846;
                double v = sqrt( exp(x*cos(3*pi*x)) * exp(y*cos(3*pi*y)) ) * (j + 1);
                if constexpr (numDims >= 3)
                    v *= p(2);
                fields(j, i) = v;
            }
        }
    }
    //
    // Compute L2-norm of error of interpolation
    //
    void compute_L2norm_error() {
        MatrixXd fields_error(numFields,g_numTargetPoints);
        compute_test_fields(*target_points_p, fields_error);
        fields_error -= *target_fields_p;
        std::cout << "L2 norm of error of interpolation: " << fields_error.norm() << std::endl;
    }
    //
    // Generate random data
    //
    void generate_random_data(
          MatrixXd*& points, MatrixXd*& fields
    ) {
        int numPoints = n_lat_i*n_lon_i*n_hgt_i;
        numFields = 1;

        // Save global number of points
        g_numPoints = numPoints;

        // Allocate
        points = new MatrixXd(numDims, numPoints);
        fields = new MatrixXd(numFields, numPoints);
        
        // Generate a set of 3D points and associated field values
        if(source_is_structured) {
            for(int i = 0; i < n_lat_i; i++) {
                for(int j = 0; j < n_lon_i; j++) {
                    for(int k = 0; k < n_hgt_i; k++) {
                       int idx = i * n_lon_i * n_hgt_i + j * n_hgt_i + k;
                       double x, y;
                       x = lon_min + (j * (lon_max - lon_min)) / (n_lon_i - 1);
                       y = lat_min + (i * (lat_max - lat_min)) / (n_lat_i - 1);
                       (*points)(0, idx) = x;
                       (*points)(1, idx) = y;
                       if constexpr (numDims >= 3) {
                           constexpr double pi = 3.14159265358979323846;
                           x = 2 * (x - lon_min) / (lon_max - lon_min) - 1;
                           y = 2 * (y - lat_min) / (lat_max - lat_min) - 1;
                           double v = sqrt( exp(x*cos(3*pi*x)) * exp(y*cos(3*pi*y)) ) +
                                      (hgt_min + k * (hgt_max - hgt_min) / n_hgt_o);
                           (*points)(2, idx) = v;
                       }
                    }
                }
            }
        } else {
            points->setRandom();
            points->row(0) = (points->row(0).array() + 1.0) /
                             2.0 * (lon_max - lon_min) + lon_min;
            points->row(1) = (points->row(1).array() + 1.0) /
                             2.0 * (lat_max - lat_min) + lat_min;
            if constexpr (numDims >= 3)
                points->row(2) = (points->row(2).array() + 1.0) /
                                2.0 * (hgt_max - hgt_min) + hgt_min;
        }

        // Compute test field values at given source locations
        compute_test_fields(*points,*fields);
    }

    //
    // Read file
    //
    void read_input_file(
          std::string src,
          MatrixXd*& points,
          MatrixXd*& fields,
          bool is_target = false
    ) {
        Timer t;
        int numPoints;
        int numFields;

        // Open input file
        std::string mode = "r";
        if(src.find("txt") == string::npos)
            mode = "rb";

        FILE* fp = fopen(src.c_str(), mode.c_str());
        if(!fp) {
            std::cout << "Input file: " << src << " not found!";
            exit(0);
        }

        // Text input file
        if(src.find("txt") != string::npos) {
            std::cout << "Reading input text file" << std::endl;
            size_t elements_read = 0;
            if(is_target) {
                numPoints = g_numTargetPoints;
                numFields = GlobalData::numFields;
            } else {
                numPoints = g_numPoints;
                numFields = GlobalData::numFields;
            }
            if(!numPoints)
                elements_read += fscanf(fp, "%d %d", &numPoints, &numFields);
            points = new MatrixXd(numDims, numPoints);
            if(!is_target)
                fields = new MatrixXd(numFields, numPoints);

            for(int i = 0; i < numPoints; i++) {
                for(int j = 0; j < numDims; j++)
                   elements_read += fscanf(fp, "%lf", &((*points)(j,i)));
                if(is_target) {
                    double v;
                    for(int j = 0; j < numFields; j++)
                       elements_read += fscanf(fp, "%lf", &v);
                } else {
                    for(int j = 0; j < numFields; j++)
                       elements_read += fscanf(fp, "%lf", &((*fields)(j,i)));
                }
            }
        // Binary input file
        } else if(src.find("grib") == string::npos) {
            std::cout << "Reading input binary file" << std::endl;
            size_t elements_read = 0;
            if(is_target) {
                numPoints = g_numTargetPoints;
                numFields = GlobalData::numFields;
            } else {
                numPoints = g_numPoints;
                numFields = GlobalData::numFields;
            }
            if(!numPoints) {
                elements_read += fread(&numPoints, sizeof(numPoints), 1, fp);
                elements_read += fread(&numFields, sizeof(numFields), 1, fp);
            }
            points = new MatrixXd(numDims, numPoints);
            if(!is_target)
                fields = new MatrixXd(numFields, numPoints);

            for(int i = 0; i < numPoints; i++) {
                double v;
                for(int j = 0; j < numDims; j++) {
                    elements_read += fread(&v, sizeof(v), 1, fp);
                    (*points)(j,i) = v;
                }
                if(is_target) {
                    for(int j = 0; j < numFields; j++)
                        elements_read += fread(&v, sizeof(v), 1, fp);
                } else {
                    for(int j = 0; j < numFields; j++) {
                        elements_read += fread(&v, sizeof(v), 1, fp);
                        (*fields)(j,i) = v;
                    }
                }
            }
        // Grib2 file
        } 
#if ENABLE_GRIB
        else {
            std::cout << "Reading input grib file" << std::endl;

            int ret;
            numFields = field_indices.size();
            if(numFields == 0) {
                if (codes_count_in_file(0, fp, &numFields) != 0) {
                    std::cout << "Failed to get field count" << std::endl;
                    exit(0);
                }
                for(int i = 0; i < numFields; i++)
                    field_indices.push_back(i);
            }

            // read lat/lon
            {
              codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret);

              long np;
              CODES_CHECK(codes_get_long(h, "numberOfPoints", &np), 0);
              numPoints = np;

              // Allocate
              points = new MatrixXd(numDims, numPoints);
              if(!is_target)
                  fields = new MatrixXd(numFields, numPoints);

              // Get latitude and longitude
              VectorXd lons, lats, values;
              lons.resize(numPoints);
              lats.resize(numPoints);
              values.resize(numPoints);
              CODES_CHECK(codes_grib_get_data(h, lats.data(), lons.data(), values.data()), 0);
              lons = (lons.array() < 0).select(lons.array() + 360, lons.array());
              points->row(0) = lons;
              points->row(1) = lats;

              codes_handle_delete(h);

              rewind(fp);
            }

            // loop through all fields and read data
            VectorXd values(numPoints);
            int idx = 0, f = 0;
            while (codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret)) {

              if(f < numFields && idx == field_indices[f]) {
                CODES_CHECK(codes_get_double_array(h, "values",
                            values.data(), &numPoints), 0);

                fields->row(f) = values;
                f++;
              }

              codes_handle_delete(h);
              idx++;
            }
        }
#endif

        // Close file
        fclose(fp);
        t.elapsed();
    }

    //
    // write output file
    //
    void write_output_file(std::string tmpl, std::string dst) {
        Timer t;

        // Text output file
        if(dst.find("txt") != string::npos) {
            std::cout << "Writing output text file" << std::endl;
            FILE* fp = fopen(dst.empty() ? "output.txt" : dst.c_str(), "w");
            if(GlobalData::useHeader)
                fprintf(fp, "%d %d\n", g_numTargetPoints, numFields);
            for(int i = 0; i < g_numTargetPoints; i++) {
               for(int j = 0; j < numDims; j++)
                   fprintf(fp, "%f ", (*target_points_p)(j,i));
               for(int j = 0; j < numFields; j++)
                   fprintf(fp, "%f ", (*target_fields_p)(j,i));
               fprintf(fp, "\n");
            }
            fclose(fp);

        // Binary output file
        } else if(dst.find("grib") == string::npos) {
            std::cout << "Writing output binary file" << std::endl;
            FILE* fp = fopen(dst.empty() ? "output.bin" : dst.c_str(), "wb");
            size_t elements_written = 0;
            if(GlobalData::useHeader) {
                elements_written += fwrite(&g_numTargetPoints, sizeof(g_numTargetPoints), 1, fp);
                elements_written += fwrite(&numFields, sizeof(numFields), 1, fp);
            }
            for(int i = 0; i < g_numTargetPoints; i++) {
               double v;
               for(int j = 0; j < numDims; j++) {
                   v = (*target_points_p)(j,i);
                   elements_written += fwrite(&v, sizeof(v), 1, fp);
               }
               for(int j = 0; j < numFields; j++) {
                   v = (*target_fields_p)(j,i);
                   elements_written += fwrite(&v, sizeof(v), 1, fp);
               }
            }
            fclose(fp);
        // Grib2 ouptput file
        }
#if ENABLE_GRIB
        else {
            std::cout << "Writing output grib file." << std::endl;

            FILE* fp = fopen(tmpl.c_str(), "r");
            if(!fp) return;
            size_t size = g_numTargetPoints;
            VectorXd values(size);

            int ret, idx = 0;
            while (codes_handle* h = codes_handle_new_from_file(0, fp, PRODUCT_GRIB, &ret)) {
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
        if(src.empty())
            generate_random_data(points, fields);
        else {
            read_input_file(src, points, fields);

            g_numPoints = points->cols();
            numFields = fields->rows();
            // Overwrite with a test field if requested for it
            // Useful for tuning parameters with a known field
            if(numFields && useTestField) {
                std::cout << "Overwriting with a test field" << std::endl;
                compute_test_fields(*points,*fields);
            }
        }

#if 0
        // Write input file in text format
        {
            std::cout << "Writing input text file" << std::endl;
            FILE* fp = fopen("input.txt", "w");
            fprintf(fp, "%d %d\n", g_numPoints, numFields);
            for(int i = 0; i < g_numPoints; i++) {
               for(int j = 0; j < numDims; j++)
                   fprintf(fp, "%f ", (*points)(j,i));
               for(int j = 0; j < numFields; j++)
                   fprintf(fp, "%f ", (*fields)(j,i));
               fprintf(fp, "\n");
            }
            fclose(fp);
        }
        // Write input file in binary format
        {
            std::cout << "Writing input binary file" << std::endl;
            FILE* fp = fopen("input.bin", "wb");
            size_t elements_written = 0;
            elements_written += fwrite(&g_numPoints, sizeof(g_numPoints), 1, fp);
            elements_written += fwrite(&numFields, sizeof(numFields), 1, fp);
            for(int i = 0; i < g_numPoints; i++) {
               double v;
               for(int j = 0; j < numDims; j++) {
                   v = (*points)(j,i);
                   elements_written += fwrite(&v, sizeof(v), 1, fp);
               }
               for(int j = 0; j < numFields; j++) {
                   v = (*fields)(j,i);
                   elements_written += fwrite(&v, sizeof(v), 1, fp);
               }
            }
            fclose(fp);
        }
        exit(0);
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
    
    // Rbf shape factor specific to cluster
    double rbfShape;

    //
    // Scatter data to slave processors
    //
    void scatter() {
        using namespace GlobalData;

        // scatter number of fields and parameters
        MPI_Bcast(&GlobalData::numFields, 1, MPI_INT,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&GlobalData::numClustersPerRank, 1, MPI_INT,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&numNeighbors, 1, MPI_INT,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&numNeighborsInterp, 1, MPI_INT,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&rbfShapeGlobal, 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&useCutoffRadius, 1, MPI_C_BOOL,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&cutoffRadius, 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&cutoffRadiusInterp, 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        MPI_Bcast(&rbfSmoothing, 1, MPI_DOUBLE,
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

        //offsets and counts
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

        //offsets and counts
        VectorXi offsets(numClusters);
        VectorXi counts(numClusters);

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
    // Average duplicates
    //
    struct PointHash {
        size_t operator()(const VectorXd& p) const {
            size_t h1 = std::hash<double>{}(p(0) / dupsTolerance);
            size_t h2 = std::hash<double>{}(p(1) / dupsTolerance);
            size_t h3 = (numDims == 3) ? std::hash<double>{}(p(2) / dupsTolerance) : 0;
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    struct PointEqual {
        bool operator()(const VectorXd& p1, const VectorXd& p2) const {
            return std::abs(p1(0) - p2(0)) < dupsTolerance &&
                   std::abs(p1(1) - p2(1)) < dupsTolerance &&
                   (numDims == 2 || std::abs(p1(2) - p2(2)) < dupsTolerance);
        }
    };
    void average_duplicates() {
        Timer t;

        cout_mutex.lock();
        std::cout << "Averaging duplicates ..." << std::endl;
        cout_mutex.unlock();

        int numPoints = fields.cols(), numFields = fields.rows();
        std::unordered_map<VectorXd, int, PointHash, PointEqual> uniquePoints;

        // Find and process duplicates
        VectorXi duplicates(numPoints), duplicates_count(numPoints);
        duplicates.setZero(numPoints);
        duplicates_count.setZero(numPoints);

        int dups = 0;
        for (int i = 0; i < numPoints; i++) {
            if(duplicates[i]) continue;

            auto it = uniquePoints.find(points.col(i));
            if (it != uniquePoints.end()) {
                duplicates[i] = it->second + 1;
                dups++;
            } else {
                uniquePoints[points.col(i)] = i;
            }
        }

        // Average duplicates
        for (int i = 0; i < numPoints; i++) {
            if(!duplicates[i]) continue;
            int idx = duplicates[i] - 1;
            fields.col(idx) += fields.col(i);
            duplicates_count(idx)++;
        }
        for (int i = 0; i < numPoints; i++) {
            if(!duplicates_count[i]) continue;
            fields.col(i) /= (duplicates_count[i] + 1);
        }

        // Resize arrays
        MatrixXd points_(numDims, numPoints - dups);
        MatrixXd fields_(numFields, numPoints - dups);
        int idx = 0;
        for (int i = 0; i < numPoints; i++) {
            if(duplicates[i]) continue;
            points_.col(idx) = points.col(i);
            fields_.col(idx) = fields.col(i);
            idx++;
        }
        points.resize(numDims, numPoints - dups);
        fields.resize(numDims, numPoints - dups);
        points = points_;
        fields = fields_;
        ClusterData::numPoints = numPoints - dups;

        // Finish
        cout_mutex.lock();
        std::cout << "Found and averaged " << dups << " duplicates." << std::endl;
        cout_mutex.unlock();

        t.elapsed();
    }
    //
    // Build sparse interpolation matrix and LU decompose it
    //
    void rbf_build() {
    
        //
        // Build sparse Matrix of radial basis function evaluations
        //
        cout_mutex.lock();
        std::cout << "Constructing interpolation matrix ..." << std::endl;
        cout_mutex.unlock();
    
        Timer t;
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(numPoints * numNeighbors + 2 * monomials * numPoints);
    
        VectorXd query(numDims);
    
        if(useCutoffRadius) {
            std::vector<nanoflann::ResultItem<unsigned, double>> matches;
            for (int i = 0; i < numPoints; i++) {
                // Perform a radius search
                query = points.col(i);
                unsigned nMatches = knn_radius(*ptree, cutoffRadius, query.data(), matches);
    
                // Compute matrix coefficients
                double sum = 0;
                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = rbf(sqrt(matches[k].second),rbfShape);
                    if(i == j)
                        r += rbfSmoothing;
                    matches[k].second = r;
                    sum += r;
                }
    
                // Normalize the coefficients per row
                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = matches[k].second / sum;
                    tripletList.push_back(T(i,j,r));
                }
    
                // Add polynomial
                for(int j = 0; j < monomials; j++) {
                    tripletList.push_back(T(i,numPoints + j,1));
                    tripletList.push_back(T(numPoints + j,i,1));
                }
            }
        } else {
            vector<size_t> indices(numNeighbors);
            vector<double> distances(numNeighbors);
            for (int i = 0; i < numPoints; i++) {
    
                // Perform the k-nearest neighbor search
                query = points.col(i);
                knn(*ptree, numNeighbors, query.data(), &indices[0], &distances[0]);
    
                // Compute matrix coefficients
                double sum = 0;
                for (int k = 0; k < numNeighbors; k++) {
                    int j = indices[k];
                    double r = rbf(sqrt(distances[k]),rbfShape);
                    if(i == j)
                        r += rbfSmoothing;
                    distances[k] = r;
                    sum += r;
                }
    
                // Normalize the coefficients per row
                for (int k = 0; k < numNeighbors; k++) {
                    int j = indices[k];
                    double r = distances[k] / sum;
                    tripletList.push_back(T(i,j,r));
                }
    
                // Add polynomial
                for(int j = 0; j < monomials; j++) {
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
        cout_mutex.lock();
        std::cout << "Started factorization ..." << std::endl;
        cout_mutex.unlock();
    
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
    void rbf_solve(const MatrixXd& F, MatrixXd& C) {
    
        cout_mutex.lock();
        std::cout << "Started solve ..." << std::endl;
        cout_mutex.unlock();
    
        Timer t;
    
        C = solver.solve(F);
    
    #ifdef USE_ITERATIVE
        cout_mutex.lock();
        std::cout << "----------------------------------" << std::endl;
        std::cout << "numb iterations: " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;
        std::cout << "----------------------------------" << std::endl;
        cout_mutex.lock();
    #endif
    
        t.elapsed();
    }

    //
    // Build KD tree needed for fast nearest neighbor search
    //
    void build_kdtree() {
        cout_mutex.lock();
        std::cout << "Started building KD tree ..." << std::endl;
        cout_mutex.unlock();

        Timer t;

        cloud = new PointCloud(points, numPoints);
        ptree = new KDTree(numDims, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams());
        ptree->buildIndex();

        t.elapsed();
    }
    //
    // Build sparse RBF interpolation matrix using either k nearest neigbors
    // or cutoff radius criteria
    //
    void build_rbf() {

        // Compute rbf shape factor
        if(rbfShapeGlobal == 0) {
            cout_mutex.lock();
            std::cout << "Started computing shape factor ..." << std::endl;
            cout_mutex.unlock();

            Timer t;

            size_t nindex[2];
            double ndistance[2], total = 0; 
            VectorXd query(numDims);
            for (int i = 0; i < numPoints; i++) {
                query = points.col(i);
                knn(*ptree, 2, query.data(), &nindex[0], &ndistance[0]);
                total += sqrt(ndistance[1]);
            }
            rbfShape = pow(32.0 / numNeighbors, 0.25) * (0.8 / (total / numPoints));

            cout_mutex.lock();
            std::cout << "Automatically computed shape factor: " << rbfShape << std::endl;
            cout_mutex.unlock();

            t.elapsed();
        } else
            rbfShape = rbfShapeGlobal;

        // non-parametric rbf
        if(numNeighbors == 1)
            return;

        // build rbf interpolation matrix
        A.resize(numPoints + monomials, numPoints + monomials);
        rbf_build();
    }
    //
    // Interpolate each field using parameteric RBF
    //
    void solve_rbf() {
        using namespace GlobalData;
        MatrixXd F(numPoints+monomials, numFields);
        MatrixXd W(numPoints+monomials, numFields);

        target_fields.setZero();

        //compute weights for each field
        cout_mutex.lock();
        std::cout << "===========================" << std::endl;
        std::cout << "Computing weights for all fields" << std::endl;
        cout_mutex.unlock();

        F.setZero();
        F.block(0,0,numPoints,numFields) = fields.transpose();
        if(numNeighbors == 1)
            W = F;
        else
            rbf_solve(F, W);

        //interpolate for target fields
        cout_mutex.lock();
        std::cout << "Interpolating fields" << std::endl;
        cout_mutex.unlock();

        Timer t;

        if(useCutoffRadius) {
            VectorXd query(numDims);
            std::vector<nanoflann::ResultItem<unsigned, double>> matches;
            for (int i = 0; i < numTargetPoints; i++) {

                // Perform a radius search
                query = target_points.col(i);
                unsigned nMatches = knn_radius(*ptree, cutoffRadiusInterp, query.data(), matches);

                // interpolate
                double sum = 0;
                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = rbf(sqrt(matches[k].second),rbfShape);
                    matches[k].second = r;
                    sum += r;
                }

                for (int k = 0; k < nMatches; k++)
                    matches[k].second /= std::max(sum,1e-6);

                for (int k = 0; k < nMatches; k++) {
                    int j = matches[k].first;
                    double r = matches[k].second;

                    for(int f = 0; f < numFields; f++)
                        target_fields(f, i) += W(j, f) * r;
                    if(k == 0) {
                        for (int j = 0; j < monomials; j++)
                            for(int f = 0; f < numFields; f++)
                                target_fields(f, i) += W(numPoints + j, f);
                    }
                }
            }
        } else {
            VectorXd query(numDims);
            vector<size_t> indices(numNeighborsInterp);
            vector<double> distances(numNeighborsInterp);
            for (int i = 0; i < numTargetPoints; i++) {

                // Perform the k-nearest neighbor search
                query = target_points.col(i);
                knn(*ptree, numNeighborsInterp, query.data(), &indices[0], &distances[0]);

                // interpolate
                double sum = 0;
                for (int k = 0; k < numNeighborsInterp; k++) {
                    int j = indices[k];
                    double r = rbf(sqrt(distances[k]),rbfShape);
                    distances[k] = r;
                    sum += r;
                }

                for (int k = 0; k < numNeighborsInterp; k++)
                    distances[k] /= std::max(sum,1e-6);

                for (int k = 0; k < numNeighborsInterp; k++) {
                    int j = indices[k];
                    double r = distances[k];

                    for(int f = 0; f < numFields; f++)
                        target_fields(f, i) += W(j, f) * r;
                    if(k == 0) {
                        for (int j = 0; j < monomials; j++)
                            for(int f = 0; f < numFields; f++)
                                target_fields(f, i) += W(numPoints + j, f);
                    }
                }
            }
        }

        t.elapsed();
    }
    //
    // Clear
    //
    void clear_rbf() {
        if(cloud) delete cloud;
        if(ptree) delete ptree;
        A.resize(1,1);
        solver.compute(A);
        points.resize(0,0);
        fields.resize(0,0);
        target_points.resize(0,0);
    }
    //
    // Conveneince function to build and solve rbf interpolation
    //
    void build_and_solve() {
        if(averageDuplicates)
            average_duplicates(); 
        build_kdtree();
        build_rbf();
        solve_rbf();
        clear_rbf();
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
    std::cout << "Interpolate fields from one grid onto another grid or scattered locations using grib2/text/binary formats." << std::endl << std::endl
              << "Example:" << std::endl
              << "    OMP_NUM_THREADS=8 ./interp -i rrfs_a.t06z.bgdawpf007.tm00.grib2 -t rrfs.t06z.prslev.f007.ak.grib2 -f 0,3" << std::endl << std::endl
              << "This example performs interpolation of fields 0 and 3 using 8 threads from the North American domain to the Alaska grid." << std::endl << std::endl
              << "usage: ./interp [-h] [--input INPUT] [--output OUTPUT] [--template TEMPLATE]" << std::endl
              << "                     [--clusters-per-rank CLUSTERS_PER_RANK] [--fields FIELDS]" << std::endl
              << "                     [--neighbors NEIGHBORS] [--neighbors-interp NEIGHBORS_INTERP]" << std::endl
              << "                     [--rbf-shape RBF_SHAPE] [--use-cutoff-radius USE_CUTOFF_RADIUS]" << std::endl
              << "                     [--cutoff-radius CUTOFF_RADIUS] [--cutoff-radius-interp CUTOFF_RADIUS_INTERP]" << std::endl << std::endl
              << "arguments:" << std::endl
              << "  -h, --help               Show this help message and exit." << std::endl
              << "  -i, --input              Input file in grib or other text/binary format containing coordinates and fields for interpolation." << std::endl
              << "  -o, --output             Output file in grib or other text/binary format containing the result of the interpolation." << std::endl
              << "  -t, --target             Target file in grib or other text/binary format containing points of interpolation." << std::endl
              << "  -c, --clusters-per-rank  Number of point clusters (point clouds) per MPI rank" << std::endl
              << "  -f, --fields             Comma-separated list of field indices in the grib file to be interpolated." << std::endl
              << "                           Use hyphens (-) to indicate a range of fields (e.g., 0-3 for fields 0, 1, 2)." << std::endl
              << "                           Use a question mark (?) to indicate all fields." << std::endl
              << "  -n, --neighbors          Number of neighbors to be used during the solution for weights using source points." << std::endl
              << "  -ni, --neighbors-interp  Number of neighbors to be used during the interpolation at target points." << std::endl
              << "  -r, --rbf-shape          Shape factor for the RBF (Radial Basis Function) kernel." << std::endl
              << "  -ucr, --use-cutoff-radius      Use a cutoff radius instead of fixed number of nearest neighbors." << std::endl
              << "  -cr, --cutoff-radius           Cutoff radius used during the solution." << std::endl
              << "  -cri, --cutoff-radius-interp   Cutoff radius used during interpolation." << std::endl
              << "  -r, --rbf-smoothing      Smoothing factor for RBF interpolation." << std::endl
              << "  -m, --monomials          Number of monomials (0 or 1 supported)." << std::endl
              << "  -a, --average-duplicates Average duplicate entries in input files." << std::endl
              << "  -h, --header             Provide three integers for the total number of source points, target points and fields in the input files." << std::endl
              << "                              e.g. --header 12000 24000 1" << std::endl
              << "                           In this case, the ouput file will have no headers as well." << std::endl
              << "  -utf, --use-test-field   Use a test field function for initializing fields. This applies even if input is read from a file." << std::endl
              << "                           It can be useful for tuning parameters with the L2 error interpolation from ground truth." << std::endl;
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
    using GlobalData::numClustersPerRank;
    using GlobalData::field_indices;
    std::string src, dst, tmpl;

    numNeighbors = 1;
    numNeighborsInterp = 32;
    numClustersPerRank = 1;
    rbfShapeGlobal = 0;
    useCutoffRadius = false;
    cutoffRadius = 0.08;
    cutoffRadiusInterp = 0.64;
    rbfSmoothing = 0.0;
    monomials = 0;
    useTestField = false;
    averageDuplicates = false;
    field_indices.push_back(0);

    GlobalData::numFields = 0;
    GlobalData::g_numPoints = 0;
    GlobalData::g_numTargetPoints = 0;
    GlobalData::useHeader = true;

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
            } else if(*it == "-t" || *it == "--target") {
                tmpl = *++it;
            } else if(*it == "-c" || *it == "--clusters-per-rank") {
                numClustersPerRank = stoi(*++it);
            } else if(*it == "-f" || *it == "--fields") {
                std::stringstream ss(*++it);
                std::string token;
                field_indices.clear();
                while (std::getline(ss, token, ',')) {
                    size_t idx = token.find("-");
                    if(idx == std::string::npos) {
                        if(token != "?")
                            field_indices.push_back(stoi(token));
                    } else {
                        int start = stoi(token.substr(0,idx));
                        int end = stoi(token.substr(idx+1));
                        for(int i = start; i < end; i++)
                            field_indices.push_back(i);
                    }
                }
            } else if(*it == "-r" || *it == "--rbf-shape") {
                rbfShapeGlobal = stof(*++it);
            } else if(*it == "-n" || *it == "--neighbors") {
                numNeighbors = stoi(*++it);
            } else if(*it == "-ni" || *it == "--neighbors-interp") {
                numNeighborsInterp = stoi(*++it);
            } else if(*it == "-ucr" || *it == "--use-cutoff-radius") {
                useCutoffRadius = true;
            } else if(*it == "-cr" || *it == "--cutoff-radius") {
                cutoffRadius = stof(*++it);
            } else if(*it == "-cri" || *it == "--cutoff-radius-interp") {
                cutoffRadiusInterp = stof(*++it);
            } else if(*it == "-s" || *it == "--rbf-smoothing") {
                rbfSmoothing = stof(*++it);
            } else if(*it == "-m" || *it == "--monomials") {
                monomials = stoi(*++it);
            } else if(*it == "-utf" || *it == "--use-test-field") {
                useTestField = true;
            } else if(*it == "-a" || *it == "--average-duplicates") {
                averageDuplicates = true;
            } else if(*it == "-hd" || *it == "--header") {
                GlobalData::g_numPoints = stoi(*++it);
                GlobalData::g_numTargetPoints = stoi(*++it);
                GlobalData::numFields = stoi(*++it);
                GlobalData::useHeader = false;
            }
        }

        int nthreads;
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        std::cout << "Threads: " << nthreads << std::endl;
        numClustersPerRank = std::max(numClustersPerRank, nthreads);

        std::cout << "===== Parameters ====" << std::endl
                  << "numDims: " << numDims << std::endl
                  << "numNeighbors: " << numNeighbors << std::endl
                  << "numNeighborsInterp: " << numNeighborsInterp << std::endl
                  << "rbfShapeGlobal: " << rbfShapeGlobal << std::endl
                  << "useCutoffRadius: " << (useCutoffRadius ? "true" : "false") << std::endl
                  << "cutoffRadius: " << cutoffRadius << std::endl
                  << "cutoffRadiusInterp: " << cutoffRadiusInterp << std::endl
                  << "numClustersPerRank: " << numClustersPerRank << std::endl
                  << "rbfSmoothing: " << rbfSmoothing << std::endl
                  << "monomials: " << monomials << std::endl
                  << "=====================" << std::endl;
    }

    //
    // Set Eigen to use single thread
    //
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    std::cout << "Eigen will be using " << Eigen::nbThreads() << " threads." << std::endl;

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
        GlobalData::write_output_file(tmpl, dst);

    // Compute L2 norm of error
    if(src.empty() || useTestField)
        GlobalData::compute_L2norm_error();

    //
    // Finalize MPI
    //

    MPI_Finalize();

    return 0;
}

