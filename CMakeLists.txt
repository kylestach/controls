include_directories ("${CMAKE_CURRENT_SOURCE_DIR}")
cmake_minimum_required (VERSION 2.6)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

enable_testing ()

add_subdirectory (third_party)
add_subdirectory (mpc)
add_subdirectory (ilqr)
add_subdirectory (common)

include_directories (${gtest_SOURCE_DIR/include} ${gtest_SOURCE_DIR})
