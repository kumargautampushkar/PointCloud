#include <iostream>
#include <random>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/io.h>

#include <pcl/io/pcd_io.h>


template <typename PointT>
class PointCloudNoiseInjector {
public:
    // Typedefs for cleaner PCL pointer usage
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

    PointCloudNoiseInjector() {
        // Initialize random engine with a non-deterministic seed
        std::random_device rd;
        rng_.seed(rd());
    }

    /**
     * @brief Adds Gaussian noise to the coordinates of existing points.
     * @param input_cloud The original point cloud.
     * @param std_dev The standard deviation of the Gaussian noise.
     * @return A new point cloud with jittered points.
     */
    PointCloudPtr addGaussianNoise(const PointCloudConstPtr& input_cloud, float std_dev) {
        PointCloudPtr noisy_cloud(new PointCloud(*input_cloud)); // Copy data
        std::normal_distribution<float> dist(0.0f, std_dev);

        for (auto& point : noisy_cloud->points) {
            // Check for NaN/Inf values before modifying
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                point.x += dist(rng_);
                point.y += dist(rng_);
                point.z += dist(rng_);
            }
        }
        return noisy_cloud;
    }

    /**
     * @brief Generates new noise points hovering near the surfaces of the existing cloud.
     * @param input_cloud The original point cloud acting as the surface.
     * @param num_new_points Number of noise points to attempt to generate.
     * @param distance_threshold Maximum allowed distance from the original surface.
     * @param noise_spread The spread of the noise from the seed points.
     * @return A point cloud containing *only* the new near-surface noise points.
     */
    PointCloudPtr generateNearSurfaceNoise(const PointCloudConstPtr& input_cloud, 
                                           int num_new_points, 
                                           float distance_threshold, 
                                           float noise_spread) {
        PointCloudPtr noise_cloud(new PointCloud());
        if (input_cloud->empty()) return noise_cloud;

        std::uniform_int_distribution<size_t> idx_dist(0, input_cloud->points.size() - 1);
        std::normal_distribution<float> noise_dist(0.0f, noise_spread);

        // Setup KD-Tree for distance filtering
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(input_cloud);

        float threshold_sq = distance_threshold * distance_threshold;
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        for (int i = 0; i < num_new_points; ++i) {
            // 1. Pick a random seed point from the surface
            size_t seed_idx = idx_dist(rng_);
            PointT seed_pt = input_cloud->points[seed_idx];

            if (!std::isfinite(seed_pt.x)) continue;

            // 2. Push it off the surface using Gaussian spread
            PointT noisy_pt = seed_pt;
            noisy_pt.x += noise_dist(rng_);
            noisy_pt.y += noise_dist(rng_);
            noisy_pt.z += noise_dist(rng_);

            // 3. Filter using KD-Tree to ensure it's within the strict distance threshold
            if (kdtree.nearestKSearch(noisy_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                if (pointNKNSquaredDistance[0] <= threshold_sq) {
                    noise_cloud->points.push_back(noisy_pt);
                }
            }
        }

        noise_cloud->width = noise_cloud->points.size();
        noise_cloud->height = 1;
        noise_cloud->is_dense = true;

        return noise_cloud;
    }

private:
    std::mt19937 rng_; // Mersenne Twister random number generator
};

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;
    // 1. Create a dummy point cloud for testing (or load a PCD/PLY)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if(pcl::io::loadPCDFile("/home/kumar/coding/PointCloud/data/min_cut_segmentation_tutorial.pcd",*cloud)==-1){
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    // cloud->width = 1000;
    // cloud->height = 1;
    // cloud->points.resize(cloud->width * cloud->height);
    // for (auto& pt : cloud->points) {
    //     pt.x = 1024 * rand() / (RAND_MAX + 1.0f);
    //     pt.y = 1024 * rand() / (RAND_MAX + 1.0f);
    //     pt.z = 1024 * rand() / (RAND_MAX + 1.0f);
    // }

    // 2. Instantiate our templated noise injector
    PointCloudNoiseInjector<pcl::PointXYZ> noise_injector;

    // 3. Add Gaussian noise to existing points
    float point_jitter_std_dev = 0.01f;
    auto jittered_cloud = noise_injector.addGaussianNoise(cloud, point_jitter_std_dev);

    // 4. Generate new volumetric noise near the surfaces
    int num_clutter_points = 5000;
    float max_distance_from_surface = 0.05f;
    float spread = 0.03f;
    auto surface_clutter_cloud = noise_injector.generateNearSurfaceNoise(
        jittered_cloud, num_clutter_points, max_distance_from_surface, spread);

    // 5. Combine the jittered surface and the clutter into a single cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *final_cloud = *jittered_cloud + *surface_clutter_cloud;

    std::cout << "Original points: " << cloud->size() << "\n";
    std::cout << "Final points (Jitter + Surface Clutter): " << final_cloud->size() << "\n";

    if(pcl::io::savePCDFile("/home/kumar/coding/PointCloud/data/noisy_cloud",*final_cloud)==-1){
        std::cout<<"Failed to save pointcloud"<<std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}