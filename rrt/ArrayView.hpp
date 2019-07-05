#include <vector>

template<typename T>
struct ArrayView {
public:
    ArrayView(std::vector<T>& vec, int start, int end)
        : initial(&(vec[start])), length(end - start) {}

    ArrayView(ArrayView view, int start, int end)
        : initial(&(view[start])), length(end - start) {}

    T& operator[](int idx) {
        if (idx >= 0 && idx < length) {
            return initial[idx];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

    T& operator[](int idx) const {
        if (idx >= 0 && idx < length) {
            return initial[idx];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

    size_t size() const {
        return length;
    }

    T* begin() { return initial; }
    T* end() { return &(initial[length]); }

    T const* cbegin() const { return initial; }
    T const* cend() const { return &(initial[length]); }

private:
    T* initial;
    size_t length;
};

