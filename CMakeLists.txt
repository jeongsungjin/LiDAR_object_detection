cmake_minimum_required(VERSION 3.0.2)
project(lidar_clustering)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  pcl_ros
  pcl_conversions
  std_msgs
  visualization_msgs  # Add this line for MarkerArray
)

find_package(PCL REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Add the new node executable
add_executable(object_detection src/object_detection.cpp)

# Link against the necessary libraries
target_link_libraries(object_detection
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Add the existing node executable
add_executable(clustering_node src/clustering_node.cpp)

target_link_libraries(clustering_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Python executable for existing script
catkin_install_python(PROGRAMS
  src/distance_flag_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
