cmake_minimum_required(VERSION 3.8)
project(propif_bringup)

# Find required packages
find_package(ament_cmake REQUIRED)

# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Package finalization
ament_package()