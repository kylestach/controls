add_library (kdtree INTERFACE)

add_library (rrt INTERFACE)
target_link_libraries (rrt INTERFACE kdtree)

add_executable (kdtree_test KDTreeTest.cpp)
target_link_libraries (kdtree_test kdtree)

add_executable (rrt_test RRTTest.cpp)
target_link_libraries (rrt_test rrt)
