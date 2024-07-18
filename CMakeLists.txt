
cmake_minimum_required(VERSION 3.0.2)
project(lidar_clustering)

# C++11을 사용합니다
# C++14로 설정
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# catkin 패키지를 찾습니다
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
  tf
)

# PCL 라이브러리를 찾습니다
find_package(PCL REQUIRED)

# Eigen 라이브러리를 찾습니다
find_package(Eigen3 REQUIRED)

# catkin 패키지를 설정합니다
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs visualization_msgs pcl_conversions pcl_ros tf
  DEPENDS PCL EIGEN3
)

# 포함 디렉토리를 지정합니다
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 실행 파일을 추가합니다
add_executable(${PROJECT_NAME}_node src/object_detection.cpp)

# 실행 파일에 라이브러리를 링크합니다
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 설치 규칙을 지정합니다
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
