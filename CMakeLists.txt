cmake_minimum_required(VERSION 3.5)
project(dis_tutorial7)

cmake_policy(SET CMP0144 NEW)
cmake_policy(SET CMP0074 NEW)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(pcl_conversions REQUIRED)
find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})

set(dependencies rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs tf2 tf2_geometry_msgs pcl_conversions)

add_executable(cylinder_segmentation src/cylinder_segmentation.cpp)
ament_target_dependencies(cylinder_segmentation ${dependencies})

target_link_libraries(cylinder_segmentation
  ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
)

install(DIRECTORY
  launch
  meshes
  urdf
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS
  cylinder_segmentation
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/arm_mover_actions.py
  scripts/capture_oakd_image.py
  scripts/face_classification_publisher.py
  scripts/line_follower.py
  scripts/line_follower_left.py
  scripts/detect_rings.py
  scripts/detect_spill.py
  scripts/detect_spill2.py
  scripts/detect_tile_anomaly.py
  scripts/robot_commander.py
  scripts/save_ring_hist.py

  DESTINATION lib/${PROJECT_NAME})

ament_package()
