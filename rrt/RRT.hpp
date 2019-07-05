#include "KDTree.hpp"
#include <map>
#include <deque>

template<int N>
Vector<N> embedding(Vector<N> v) {
    return v;
}

Vector<1> embedding(double v) {
    return Vector<1>(v);
}

template<int N>
class SpaceFillingTree {
public:
    using Tree = KDTree<N>;

    typename Tree::TreeNode* extend(Vector<N> towards, double delta) {
        typename Tree::TreeNode* nearest = tree.nearest_neighbor(towards);
        Vector<N> new_point = calculate_extension(nearest->position, towards, delta);
        typename Tree::TreeNode* new_node = tree.add(new_point);
        parents[new_node] = nearest;
        return new_node;
    }

    void reset(Vector<N> start) {
        tree.reset();
        tree.add(start);
        parents.clear();
    }

    typename Tree::TreeNode* lookup_parent(typename Tree::TreeNode* node) {
        auto it = parents.find(node);
        if (it != parents.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

private:
    Vector<N> calculate_extension(Vector<N> from, Vector<N> to, double delta) {
        double dist = (to - from).norm();
        if (dist < delta) {
            return to;
        } else {
            return from + delta * (to - from) / dist;
        }
    }

    Tree tree;
    double default_delta;
    std::map<typename Tree::TreeNode*, typename Tree::TreeNode*> parents;
};

template<int N>
class RRT {
public:
    std::deque<Vector<N>> run(Vector<N> start, Vector<N> end) {
        // Do a thing
        space_tree.reset(start);
        double epsilon = 0.1;
        while (true) {
            Vector<N> sample = Vector<N>::Random();
            auto current = space_tree.extend(sample, 0.1);
            if ((current->position - end).norm() < epsilon) {
                // Done! Trace back the path.
                std::deque<Vector<N>> path;
                while (current) {
                    path.push_front(current->position);
                    current = space_tree.lookup_parent(current);
                }
                return path;
            }
        }
    }

private:
    SpaceFillingTree<N> space_tree;
};
