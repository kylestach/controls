add_executable (SystemTest system_test.cpp)
target_link_libraries (SystemTest gtest gtest_main Eigen3::Eigen)
add_test (SystemTest SystemTest)
