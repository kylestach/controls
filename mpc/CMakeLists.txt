include_directories ("${PROJECT_SOURCE_DIR}/third_party/qpOASES/include")

project (mpc)

add_executable (mpc mpc.cpp mpc.h)
target_link_libraries (mpc qpOASES)

add_executable (nmpc nmpc.cpp mpc.h)
target_link_libraries (nmpc qpOASES)
