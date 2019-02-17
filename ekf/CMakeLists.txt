add_library (ekf INTERFACE)

add_executable (ekf_demo ekf_demo.cpp)
target_link_libraries (ekf_demo ekf)
