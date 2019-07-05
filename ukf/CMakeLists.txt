add_library (ukf INTERFACE)

add_executable (ukf_demo ukf_demo.cpp)
target_link_libraries (ukf_demo ukf)
