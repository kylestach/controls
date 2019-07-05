#include "KDTree.hpp"
#include <vector>

int main() {
    std::vector<Vector<2>> points;
    for (int i = 0; i < 1000; i++) {
        points.push_back(Vector<2>::Random());
    }

    KDTree<2> tree(points);

    tree.verify();

    Vector<2> query = Vector<2>::Random();
    Vector<2> nearest = tree.nearest_neighbor(query)->position;

    double dist = (nearest - query).norm();

    for (auto v : points) {
        assert((v - query).norm() > dist - 1e-6);
    }
}
