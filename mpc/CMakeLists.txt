include_directories ("${PROJECT_SOURCE_DIR}/third_party/qpOASES/include")

add_library (mpc INTERFACE)
target_link_libraries (mpc INTERFACE qpOASES)

add_library (nmpc INTERFACE)
target_link_libraries (nmpc INTERFACE qpOASES)

add_executable (mpc_demo mpc_demo.cpp)
target_link_libraries (mpc_demo mpc)

add_executable (nmpc_demo nmpc_demo.cpp)
target_link_libraries (nmpc_demo nmpc)
