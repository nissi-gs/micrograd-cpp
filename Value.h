#pragma once
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <iosfwd>

class Value;
using ValuePtr = std::shared_ptr<Value>;

inline ValuePtr create_value(double value, std::vector<ValuePtr> prev = {}, std::string op = "", bool requires_grad = false, bool is_leaf = true)
{
    return std::make_shared<Value>(value, std::move(prev), std::move(op), requires_grad, is_leaf);
}

inline ValuePtr create_parameter(double value)
{
    return create_value(value, {}, "", true, true);
}

class Value
{
public:
    Value(double v, std::vector<ValuePtr> prev = {}, std::string op = "", bool requires_grad = false, bool is_leaf = true)
        : data(v), prev_(std::move(prev)), op(std::move(op)), requires_grad(requires_grad), is_leaf(is_leaf) {}

    // Get the underlying data
    double get_data() const noexcept;
    double get_grad() const noexcept;
    bool needs_grad() const noexcept;
    // set gradient (for initiating backprop)
    void set_grad(double g) noexcept;
    // zero out the gradient
    void zero_grad() noexcept;
    void set_data(double d) noexcept;
    // Generate DOT representation of the computation graph
    std::string to_dot() const;
    // Create and save a visualization of the computation graph
    void visualize(const std::string &filename) const;
    // Perform backpropagation to compute gradients
    void backward(double grad_out = 1.0, bool retain_grads = true, bool retain_graph = false);

private:
    double data;
    double grad = 0.0;
    std::vector<ValuePtr> prev_;
    std::string op;
    std::function<void()> backward_; // Placeholder for backward function
    bool requires_grad;
    bool is_leaf;

    // Helper functions for building topological order
    void build_topo(const Value *v,
                    std::unordered_map<const Value *, bool> &visited,
                    std::vector<Value *> &topo);

    // Provides topological order of nodes
    std::vector<Value *> build_topological_order();

    // Helper function to build DOT representation recursively
    void build_dot(std::stringstream &ss, std::unordered_map<const Value *, int> &node_ids, int &id) const;

    friend auto operator+(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr;

    // Overload stream insertion operator for printing
    friend std::ostream &operator<<(std::ostream &os, const Value &v);
    friend auto operator*(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr;
    friend auto operator-(const ValuePtr &rhs) -> ValuePtr;
    friend auto operator-(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr;
    friend auto pow(const ValuePtr &base, const ValuePtr &exponent) -> ValuePtr;
    friend auto operator/(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr;
    friend auto exp(const ValuePtr &v) noexcept -> ValuePtr;
    friend auto tanh(const ValuePtr &v) noexcept -> ValuePtr;
};
