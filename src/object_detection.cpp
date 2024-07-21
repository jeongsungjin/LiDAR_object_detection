#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>

ros::Publisher pub_filtered;
ros::Publisher pub_marker_array;

// ROI parameters
const double zMinROI = -0.5, zMaxROI = 2.0;

// Clustering parameters
const double epsilon = 0.2; // Increased for faster processing
const int minClusterSize = 10, maxClusterSize = 1000;

// VoxelGrid parameter
const float leafSize = 0.1f; // Increased for more aggressive downsampling

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    if (cloud->empty()) {
        ROS_WARN("Received empty point cloud. Skipping processing.");
        return;
    }

    // Downsample first to reduce computational load
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leafSize, leafSize, leafSize);
    sor.filter(*cloud);

    // Perform simplified ROI filtering
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(zMinROI, zMaxROI);
    pass.filter(*cloud);

    if (cloud->empty()) {
        ROS_WARN("Filtered cloud is empty. Skipping further processing.");
        return;
    }

    // Publish filtered cloud
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header = cloud_msg->header;
    pub_filtered.publish(output);

    // Simple clustering
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

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.reserve(std::min(cluster_indices.size(), size_t(10)));

    for (size_t i = 0; i < cluster_indices.size() && i < 10; ++i) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& idx : cluster_indices[i].indices) {
            cluster->push_back((*cloud)[idx]);
        }
        pcl::PointXYZI min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        visualization_msgs::Marker marker;
        marker.header.frame_id = cloud_msg->header.frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "detected_objects";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = (max_pt.x + min_pt.x) / 2;
        marker.pose.position.y = (max_pt.y + min_pt.y) / 2;
        marker.pose.position.z = (max_pt.z + min_pt.z) / 2;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = max_pt.x - min_pt.x;
        marker.scale.y = max_pt.y - min_pt.y;
        marker.scale.z = max_pt.z - min_pt.z;
        marker.color.a = 0.5;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker_array.markers.push_back(marker);
    }

    pub_marker_array.publish(marker_array);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lightweight_object_detection");
    ros::NodeHandle nh;

    pub_filtered = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
    pub_marker_array = nh.advertise<visualization_msgs::MarkerArray>("detected_objects", 1);

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("input_cloud", 1, cloud_cb);

    ros::spin();

    return 0;
}
