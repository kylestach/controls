include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/../third_party/qpOASES/include")

project (mpc)

add_executable (mpc mpc.cpp)
target_link_libraries (mpc qpOASES)
