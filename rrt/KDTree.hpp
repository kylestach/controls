#include <memory>
#include <vector>
#include <limits>
#include <iostream>

#include "eigen3/Eigen/Core"

#include "ArrayView.hpp"

template<int N>
using Vector = Eigen::Matrix<double, N, 1>;

template<int N>
class KDTree {
public:
    struct TreeNode;

    KDTree(std::vector<Vector<N>>& points) {
        if (points.size() > 0) {
            root = std::make_unique<TreeNode>(
                    ArrayView<Vector<N>>(points, 0, points.size()), 0);
        }
    }

    KDTree() = default;

    TreeNode* add(Vector<N> position) {
        if (root) {
            return root->add(position);
        } else {
            root = std::make_unique<TreeNode>(position, 0);
            return root.get();
        }
    }

    TreeNode* nearest_neighbor(Vector<N> query) {
        if (root) {
            return root->nearest_neighbor(query);
        } else {
            throw std::runtime_error("No values in tree!");
        }
    }

    void reset() {
        root.reset();
    }

    void display() const {
        if (root) root->display(1);
    }

    void verify() const {
        Vector<N> lower, upper;
        for (int i = 0; i < N; i++) {
            lower(i) = -std::numeric_limits<double>::infinity();
            upper(i) = std::numeric_limits<double>::infinity();
        }
        if (root) root->verify(lower, upper);
    }

    bool is_empty() const {
        return root;
    }

    struct TreeNode {
        Vector<N> position;
        int split_dimension;
        std::unique_ptr<TreeNode> children[2];

        TreeNode(Vector<N> position, int split) : position(position), split_dimension(split) {}
        TreeNode(ArrayView<Vector<N>> points, int split = 0) : split_dimension(split) {
            assert(points.size() > 0);
            position = points[0];

            // Partition the array
            int left_marker = 1, right_marker = points.size() - 1;
            while (left_marker <= right_marker) {
                while (left_marker <= right_marker
                        && points[left_marker](split) < position(split)) {
                    left_marker++;
                }

                if (left_marker == right_marker) {
                    right_marker--;
                    break;
                } else {
                    while (left_marker <= right_marker
                            && points[right_marker](split) > position(split)) {
                        right_marker--;
                    }
                }

                // Swap left/right
                if (left_marker <= right_marker) {
                    std::swap(points[left_marker++], points[right_marker--]);
                }
            }

            // Now the pivot is at the right marker
            std::swap(points[0], points[right_marker]);

            // Run recursively
            if (right_marker > 0) {
                children[0] = std::make_unique<TreeNode>(
                        ArrayView<Vector<N>>(
                            points, 0, right_marker), (split + 1) % N);
            }
            if (points.size() > right_marker + 1) {
                children[1] = std::make_unique<TreeNode>(
                        ArrayView<Vector<N>>(
                            points,
                            right_marker + 1,
                            points.size()),
                        (split + 1) % N);
            }
        }

        TreeNode* add(Vector<N> new_point) {
            int direction = position(split_dimension) < new_point(split_dimension);
            if (children[direction]) {
                return children[direction]->add(new_point);
            } else {
                children[direction] = std::make_unique<TreeNode>(
                        new_point, (split_dimension + 1) % N);
                return children[direction].get();
            }
        }

        TreeNode* nearest_neighbor(Vector<N> query) {
            TreeNode* best_result = this;
            double best_score = (position - query).squaredNorm();

            int direction = query(split_dimension) > position(split_dimension);
            if (children[direction]) {
                TreeNode* recurse_result = children[direction]->nearest_neighbor(query);
                double recurse_score = (recurse_result->position - query).squaredNorm();
                if (recurse_score < best_score) {
                    best_result = recurse_result;
                    best_score = recurse_score;
                }
            }

            if (children[!direction] &&
                    best_score > std::pow(query(split_dimension) - position(split_dimension), 2)) {
                // Recurse on opposite side.
                TreeNode* recurse_result = children[!direction]->nearest_neighbor(query);
                double recurse_score = (recurse_result->position - query).squaredNorm();
                if (recurse_score < best_score) {
                    best_result = recurse_result;
                    best_score = recurse_score;
                }
            }
            return best_result;
        }

        void display(int init) const {
            std::cout << init << "(" << position.transpose()
                << ", " << split_dimension << ") -> ";
            if (children[0]) std::cout << init * 2; else std::cout << "*";
            std::cout << ", ";
            if (children[1]) std::cout << init * 2 + 1; else std::cout << "*";
            std::cout << std::endl;
            if (children[0]) children[0]->display(init * 2);
            if (children[1]) children[1]->display(init * 2 + 1);
        }

        void verify(Vector<N> lower, Vector<N> upper) const {
            for (int i = 0; i < N; i++) {
                assert(position(i) >= lower(i));
                assert(position(i) <= upper(i));
            }

            if (children[0]) {
                Vector<N> new_upper = upper;
                new_upper(split_dimension) = position(split_dimension);
                children[0]->verify(lower, new_upper);
            }

            if (children[1]) {
                Vector<N> new_lower = lower;
                new_lower(split_dimension) = position(split_dimension);
                children[1]->verify(new_lower, upper);
            }
        }
    };

private:
    std::unique_ptr<TreeNode> root;
};
