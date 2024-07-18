#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/common.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_listener.h>
#include <Eigen/Dense>
#include <mutex>
#include <atomic>

ros::Publisher pub_filtered;
ros::Publisher pub_marker_array;
ros::Publisher pub_clusters;

// ROI parameters
double zMinROI = -0.5, zMaxROI = 2.0;
double xMinROI = 0.0, xMaxROI = 4.0;
double yMinROI = -3.0, yMaxROI = 3.0;

// Human-like dimensions
double xMinBoundingBox = 0.1, xMaxBoundingBox = 1.0;
double yMinBoundingBox = 0.2, yMaxBoundingBox = 1.0;
double zMinBoundingBox = 0.3, zMaxBoundingBox = 2.0;

// Clustering parameters
int minPoints = 10;
double epsilon = 0.1;
double minClusterSize = 20, maxClusterSize = 2000;

// VoxelGrid parameter
float leafSize = 0.05f;

// Tracking parameters
const int MAX_TRACK_HISTORY = 5;
const double MAX_MATCHING_DISTANCE = 1.0;
const size_t MAX_MARKERS = 10000;

struct TrackedObject {
    int id;
    std::vector<Eigen::Vector3f> positions;
    ros::Time last_seen;
    visualization_msgs::Marker marker;
};

std::vector<TrackedObject> tracked_objects;
std::mutex tracked_objects_mutex;
std::atomic<int> next_id{0};

pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

Eigen::Vector3f calculateAveragePosition(const std::vector<Eigen::Vector3f>& positions) {
    Eigen::Vector3f avg(0, 0, 0);
    if (!positions.empty()) {
        for (const auto& pos : positions) {
            avg += pos;
        }
        avg /= static_cast<float>(positions.size());
    }
    return avg;
}

void updateTrackedObjects(const std::vector<visualization_msgs::Marker>& detected_markers, const ros::Time& current_time) {
    std::lock_guard<std::mutex> lock(tracked_objects_mutex);
    std::vector<bool> matched(tracked_objects.size(), false);

    for (const auto& marker : detected_markers) {
        Eigen::Vector3f current_pos(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z);
        double min_distance = std::numeric_limits<double>::max();
        int closest_index = -1;

        for (size_t i = 0; i < tracked_objects.size(); ++i) {
            if (matched[i]) continue;
            Eigen::Vector3f tracked_pos = calculateAveragePosition(tracked_objects[i].positions);
            double distance = (current_pos - tracked_pos).norm();
            if (distance < min_distance && distance < MAX_MATCHING_DISTANCE) {
                min_distance = distance;
                closest_index = i;
            }
        }

        if (closest_index != -1) {
            // Update existing track
            tracked_objects[closest_index].positions.push_back(current_pos);
            if (tracked_objects[closest_index].positions.size() > MAX_TRACK_HISTORY) {
                tracked_objects[closest_index].positions.erase(tracked_objects[closest_index].positions.begin());
            }
            tracked_objects[closest_index].last_seen = current_time;
            tracked_objects[closest_index].marker = marker;
            matched[closest_index] = true;
        } else {
            // Create new track
            TrackedObject new_object;
            new_object.id = next_id.fetch_add(1);
            new_object.positions.push_back(current_pos);
            new_object.last_seen = current_time;
            new_object.marker = marker;
            tracked_objects.push_back(new_object);
        }
    }

    // Remove old tracks
    tracked_objects.erase(
        std::remove_if(
            tracked_objects.begin(),
            tracked_objects.end(),
            [&current_time](const TrackedObject& obj) {
                return (current_time - obj.last_seen).toSec() > 1.0;
            }
        ),
        tracked_objects.end()
    );
}

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    try {
        ROS_INFO("Received point cloud message");

        if (cloud_msg->data.empty()) {
            ROS_WARN("Received empty point cloud message");
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        if (cloud->points.empty()) {
            ROS_WARN("Converted cloud is empty. Skipping further processing.");
            return;
        }

        // Perform pass-through filter
        pcl::PassThrough<pcl::PointXYZI> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(xMinROI, xMaxROI);
        pass.filter(*cloud);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(yMinROI, yMaxROI);
        pass.filter(*cloud);

        pass.setFilterFieldName("z");
        pass.setFilterLimits(zMinROI, zMaxROI);
        pass.filter(*cloud);

        // Perform voxel grid downsampling
        pcl::VoxelGrid<pcl::PointXYZI> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(leafSize, leafSize, leafSize);
        sor.filter(*cloud);

        if (cloud->points.empty()) {
            ROS_WARN("Filtered cloud is empty. Skipping further processing.");
            return;
        }

        // Publish filtered and downsampled cloud
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = cloud_msg->header;
        pub_filtered.publish(output);

        // Perform Euclidean Cluster Extraction
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(epsilon);
        ec.setMinClusterSize(minClusterSize);
        ec.setMaxClusterSize(maxClusterSize);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        colored_cloud->clear();
        std::vector<uint32_t> colors;
        for(size_t i = 0; i < cluster_indices.size(); ++i) {
            colors.push_back(((uint32_t)rand() << 16 | (uint32_t)rand() << 8 | (uint32_t)rand()));
        }

        for(size_t i = 0; i < cluster_indices.size(); ++i) {
            for(const auto& idx : cluster_indices[i].indices) {
                pcl::PointXYZRGB colored_point;
                colored_point.x = cloud->points[idx].x;
                colored_point.y = cloud->points[idx].y;
                colored_point.z = cloud->points[idx].z;
                colored_point.rgb = *reinterpret_cast<float*>(&colors[i]);
                colored_cloud->points.push_back(colored_point);
            }
        }

        colored_cloud->width = colored_cloud->points.size();
        colored_cloud->height = 1;
        colored_cloud->is_dense = true;

        sensor_msgs::PointCloud2 output_clusters;
        pcl::toROSMsg(*colored_cloud, output_clusters);
        output_clusters.header = cloud_msg->header;
        pub_clusters.publish(output_clusters);

        std::vector<visualization_msgs::Marker> detected_markers;

        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto& idx : indices.indices)
                cloud_cluster->points.push_back(cloud->points[idx]);

            pcl::MomentOfInertiaEstimation<pcl::PointXYZI> feature_extractor;
            feature_extractor.setInputCloud(cloud_cluster);
            feature_extractor.compute();

            pcl::PointXYZI min_point_AABB;
            pcl::PointXYZI max_point_AABB;
            feature_extractor.getAABB(min_point_AABB, max_point_AABB);

            // Calculate dimensions
            float width = max_point_AABB.x - min_point_AABB.x;
            float depth = max_point_AABB.y - min_point_AABB.y;
            float height = max_point_AABB.z - min_point_AABB.z;

            // Check if the cluster has human-like dimensions
            if (width >= xMinBoundingBox && width <= xMaxBoundingBox &&
                depth >= yMinBoundingBox && depth <= yMaxBoundingBox &&
                height >= zMinBoundingBox && height <= zMaxBoundingBox) {

                visualization_msgs::Marker marker;
                marker.header = cloud_msg->header;
                marker.ns = "humans";
                marker.id = next_id.fetch_add(1);
                marker.type = visualization_msgs::Marker::CUBE;
                marker.action = visualization_msgs::Marker::ADD;
                marker.pose.position.x = (min_point_AABB.x + max_point_AABB.x) / 2;
                marker.pose.position.y = (min_point_AABB.y + max_point_AABB.y) / 2;
                marker.pose.position.z = (min_point_AABB.z + max_point_AABB.z) / 2;
                marker.scale.x = width;
                marker.scale.y = depth;
                marker.scale.z = height;
                marker.color.a = 0.5; // Semi-transparent
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.lifetime = ros::Duration(0.2);

                detected_markers.push_back(marker);
            }
        }

        // Update tracked objects
        updateTrackedObjects(detected_markers, cloud_msg->header.stamp);

        // Prepare marker array for visualization
        visualization_msgs::MarkerArray marker_array;
        marker_array.markers.reserve(std::min(MAX_MARKERS, tracked_objects.size()));

        {
            std::lock_guard<std::mutex> lock(tracked_objects_mutex);
            for (const auto& tracked_obj : tracked_objects) {
                visualization_msgs::Marker marker = tracked_obj.marker;
                marker.id = tracked_obj.id;

                // Calculate average position
                Eigen::Vector3f avg_pos = calculateAveragePosition(tracked_obj.positions);
                marker.pose.position.x = avg_pos.x();
                marker.pose.position.y = avg_pos.y();
                marker.pose.position.z = avg_pos.z();

                // Modify marker properties for better visibility
                marker.scale.x = std::max(static_cast<float>(marker.scale.x), 0.2f); // Minimum width
                marker.scale.y = std::max(static_cast<float>(marker.scale.y), 0.2f); // Minimum depth
                marker.scale.z = std::max(static_cast<float>(marker.scale.z), 0.2f); // Minimum height
                marker.color.a = 0.7f; // More opaque
                marker.color.r = 1.0f; // Red color
                marker.color.g = 0.0f;
                marker.color.b = 0.0f;
                marker.lifetime = ros::Duration(0.2); // Longer lifetime

                marker_array.markers.push_back(marker);
            }
        }

        pub_marker_array.publish(marker_array);
        ROS_INFO("Published human bounding boxes");
    }
    catch (const std::exception& e) {
        ROS_ERROR_STREAM("Exception in cloud_cb: " << e.what());
    }
    catch (...) {
        ROS_ERROR("Unknown exception in cloud_cb");
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "human_detection_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Load parameters from ROS parameter server
    pnh.param("zMinROI", zMinROI, zMinROI);
    pnh.param("zMaxROI", zMaxROI, zMaxROI);
    pnh.param("xMinROI", xMinROI, xMinROI);
    pnh.param("xMaxROI", xMaxROI, xMaxROI);
    pnh.param("yMinROI", yMinROI, yMinROI);
    pnh.param("yMaxROI", yMaxROI, yMaxROI);
    pnh.param("xMinBoundingBox", xMinBoundingBox, xMinBoundingBox);
    pnh.param("xMaxBoundingBox", xMaxBoundingBox, xMaxBoundingBox);
    pnh.param("yMinBoundingBox", yMinBoundingBox, yMinBoundingBox);
    pnh.param("yMaxBoundingBox", yMaxBoundingBox, yMaxBoundingBox);
    pnh.param("zMinBoundingBox", zMinBoundingBox, zMinBoundingBox);
    pnh.param("zMaxBoundingBox", zMaxBoundingBox, zMaxBoundingBox);
    pnh.param("minPoints", minPoints, minPoints);
    pnh.param("epsilon", epsilon, epsilon);
    pnh.param("minClusterSize", minClusterSize, minClusterSize);
    pnh.param("maxClusterSize", maxClusterSize, maxClusterSize);
    pnh.param("leafSize", leafSize, leafSize);

    // Publishers
    pub_filtered = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
    pub_clusters = nh.advertise<sensor_msgs::PointCloud2>("clusters", 1);
    pub_marker_array = nh.advertise<visualization_msgs::MarkerArray>("human_bounding_boxes", 1);

    // Subscriber
    ros::Subscriber sub = nh.subscribe("input", 1, cloud_cb);

    ros::spin();
    return 0;
}