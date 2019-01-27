add_library (ilqr INTERFACE)

add_executable (ilqr_demo ilqr_demo.cpp)
target_link_libraries (ilqr_demo ilqr)
