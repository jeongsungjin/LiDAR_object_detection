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

ros::Publisher pub_filtered;
ros::Publisher pub_marker_array;
ros::Publisher pub_clusters;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    ROS_INFO("Received point cloud message");

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    ROS_INFO("Converted to PCL PointXYZI");

    // Perform pass-through filter to retain only forward-facing points
    pcl::PassThrough<pcl::PointXYZI> pass_x;
    pass_x.setInputCloud(cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(0.0, 5.0);  // 0m에서 5m 사이의 점만 유지
    pass_x.filter(*cloud);

    pcl::PassThrough<pcl::PointXYZI> pass_y;
    pass_y.setInputCloud(cloud);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(-3.0, 3.0);  // 좌우 4m 범위 내의 점만 유지
    pass_y.filter(*cloud);

    pcl::PassThrough<pcl::PointXYZI> pass_z;
    pass_z.setInputCloud(cloud);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(-1.0, 2.0);  // 지면에서 2m 높이까지의 점만 유지
    pass_z.filter(*cloud);

    ROS_INFO("Filtered points to keep only forward-facing points within specified range");

    // Perform voxel grid downsampling
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.02f, 0.02f, 0.02f);  // Reduced leaf size for finer resolution
    sor.filter(*cloud);
    ROS_INFO("Performed voxel grid downsampling");

    // Publish filtered and downsampled cloud
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header = cloud_msg->header;  // 원본 메시지의 헤더 정보를 유지
    pub_filtered.publish(output);
    ROS_INFO("Published filtered and downsampled point cloud");

    // Perform Euclidean Cluster Extraction
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.05);  // 5cm
    ec.setMinClusterSize(30);    // Increased minimum cluster size
    ec.setMaxClusterSize(500);   // Decreased maximum cluster size
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    ROS_INFO("Performed Euclidean Cluster Extraction");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<uint32_t> colors;
    for(size_t i = 0; i <cluster_indices.size(); ++i) {
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
    ROS_INFO("Published clustered point cloud");

    visualization_msgs::MarkerArray marker_array;
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
        float height = max_point_AABB.y - min_point_AABB.y;
        float depth = max_point_AABB.z - min_point_AABB.z;

        // Set human-like dimensions
        float min_width = 0.2;  // 20cm
        float max_width = 0.8;  // 80cm
        float min_height = 0.2; // 80cm
        float max_height = 1.5; // 200cm
        float min_depth = 0.1;  // 10cm
        float max_depth = 0.6;  // 60cm
        
        // Check if the cluster has human-like dimensions
        if (width >= min_width && width <= max_width &&
            height >= min_height && height <= max_height &&
            depth >= min_depth && depth <= max_depth) {
            
            visualization_msgs::Marker marker;
            marker.header = cloud_msg->header;
            marker.ns = "human_bounding_boxes";
            marker.id = marker_array.markers.size();
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

            marker_array.markers.push_back(marker);
        }
    }

    pub_marker_array.publish(marker_array);
    ROS_INFO("Published human bounding boxes");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "clustering_node");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("velodyne_points", 1, cloud_cb);
    pub_filtered = nh.advertise<sensor_msgs::PointCloud2>("filtered_points", 1);
    pub_marker_array = nh.advertise<visualization_msgs::MarkerArray>("human_bounding_boxes", 1);
    pub_clusters = nh.advertise<sensor_msgs::PointCloud2>("clustered_points", 1);

    ros::spin();
    return 0;
}
