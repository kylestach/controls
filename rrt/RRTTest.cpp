#include "RRT.hpp"

int main() {
    Vector<2> start(-0.8, -0.8), end(0.8, 0.8);
    RRT<2> rrt;
    auto path = rrt.run(start, end);
    for (auto& node : path) {
        std::cout << node.transpose() << std::endl;
    }
}
