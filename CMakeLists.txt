cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Name your project
project(pointcloud_noise_injector)

# Find the Point Cloud Library (PCL)
# We request at least version 1.2, but it will find whatever modern version you have installed.
find_package(PCL 1.2 REQUIRED)

# Include the PCL headers and definitions
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add your C++ source file to create the executable
add_executable(noise_injector src/main.cpp)

# Link the required PCL modules to your executable
# Using ${PCL_LIBRARIES} links everything found, which covers common and kdtree.
target_link_libraries(noise_injector ${PCL_LIBRARIES})