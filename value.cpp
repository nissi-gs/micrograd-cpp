#include "Value.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>

// ------ Member functions of Value class ------
double Value::get_data() const noexcept { return data; }
double Value::get_grad() const noexcept { return grad; }
bool Value::needs_grad() const noexcept { return requires_grad; }
void Value::set_grad(double g) noexcept { grad = g; }
void Value::zero_grad() noexcept { grad = 0.0; }
void Value::set_data(double d) noexcept { data = d; }

// ------ Gradient calculation  -----
void Value::build_topo(const Value *v,
                       std::unordered_map<const Value *, bool> &visited,
                       std::vector<Value *> &topo)
{
    if (visited.find(v) != visited.end())
        return;

    visited[v] = true;
    for (const auto &child : v->prev_)
    {
        build_topo(child.get(), visited, topo);
    }
    topo.push_back(const_cast<Value *>(v));
}

std::vector<Value *> Value::build_topological_order()
{
    std::unordered_map<const Value *, bool> visited;
    std::vector<Value *> topo;
    build_topo(this, visited, topo);
    return topo;
}

void Value::backward(double grad_out, bool retain_grads, bool retain_graph)
{
    grad = grad_out;
    auto topo = build_topological_order();
    std::reverse(topo.begin(), topo.end());

    // run backprop
    for (auto *node : topo)
    {
        if (node->requires_grad && node->backward_)
        {
            node->backward_();
        }
    }

    // clear gradients for non-leaf nodes if not retaining grads
    if (!retain_grads)
    {
        for (auto *node : topo)
        {
            if (!node->is_leaf)
            {
                node->grad = 0.0;
            }
        }
    }

    // clear backward functions if not retaining graph
    if (!retain_graph)
    {
        for (auto *node : topo)
        {
            node->backward_ = nullptr;
            node->prev_.clear();
        }
    }
}

// ------ Overloaded operators -----
std::ostream &operator<<(std::ostream &os, const Value &v)
{
    return os << "Value(data=" << v.data << ")";
}

auto operator+(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr
{
    auto output = create_value(lhs->data + rhs->data, {lhs, rhs}, "+", lhs->requires_grad || rhs->requires_grad, false);
    //                                                                 determines if output requires grad        is_leaf = false
    if (lhs->requires_grad || rhs->requires_grad)
    {
        std::weak_ptr<Value> wl = lhs, wr = rhs, wout = output;

        output->backward_ = [wl, wr, wout]()
        {
            auto out = wout.lock();
            auto l = wl.lock();
            auto r = wr.lock();
            assert(out && l && r);
            if (l->requires_grad)
            {
                l->grad += out->grad;
            }
            if (r->requires_grad)
            {
                r->grad += out->grad;
            }
        };
    }
    else
    {
        output->backward_ = nullptr;
    }

    return output;
}

auto operator*(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr
{
    auto output = create_value(lhs->data * rhs->data, {lhs, rhs}, "*", lhs->requires_grad || rhs->requires_grad, false);
    //                                                                 determines if output requires grad        is_leaf = false

    if (lhs->requires_grad || rhs->requires_grad)
    {
        std::weak_ptr<Value> wl = lhs, wr = rhs, wout = output;

        output->backward_ = [wl, wr, wout]()
        {
            auto out = wout.lock();
            auto l = wl.lock();
            auto r = wr.lock();
            assert(out && l && r);
            if (l->requires_grad)
            {
                l->grad += r->data * out->grad;
            }
            if (r->requires_grad)
            {
                r->grad += l->data * out->grad;
            }
        };
    }
    else
    {
        output->backward_ = nullptr;
    }

    return output;
}

auto operator-(const ValuePtr &rhs) -> ValuePtr
{
    return rhs * create_value(-1, {}, "", false);
}

auto operator-(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr
{
    return lhs + (-rhs);
}

auto pow(const ValuePtr &base, const ValuePtr &exponent) -> ValuePtr
{
    auto output = create_value(std::pow(base->data, exponent->data), {base, exponent}, "pow", base->requires_grad || exponent->requires_grad, false);
    //                                                                                        determines if output requires grad        is_leaf = false

    if (base->requires_grad || exponent->requires_grad)
    {
        std::weak_ptr<Value> wbase = base, wexp = exponent, wout = output;
        output->backward_ = [wbase, wexp, wout]()
        {
            auto out = wout.lock();
            auto b = wbase.lock();
            auto e = wexp.lock();
            assert(out && b && e);
            if (b->data == 0 && e->data < 1)
            {
                throw std::runtime_error("Undefined gradient for 0^negative");
            }

            if (b->requires_grad)
            {
                b->grad += e->data * std::pow(b->data, e->data - 1) * out->grad;
            }

            if (b->data > 1e-12 && e->requires_grad)
            {
                e->grad += std::log(b->data) * std::pow(b->data, e->data) * out->grad;
            }
        };
    }
    else
    {
        output->backward_ = nullptr;
    }

    return output;
}

auto operator/(const ValuePtr &lhs, const ValuePtr &rhs) -> ValuePtr
{
    if (rhs->data == 0 || std::abs(rhs->data) < 1e-12)
    {
        throw std::runtime_error("Division by zero");
    }
    return lhs * pow(rhs, create_value(-1, {}, "", false));
}

auto exp(const ValuePtr &v) noexcept -> ValuePtr
{
    auto output = create_value(std::exp(v->data), {v}, "exp", v->requires_grad, false);

    if (!v->requires_grad)
    {
        output->backward_ = nullptr;
        return output;
    }

    std::weak_ptr<Value> wv = v, wout = output;
    output->backward_ = [wv, wout]()
    {
        auto out = wout.lock();
        auto val = wv.lock();
        assert(out && val);
        val->grad += out->data * out->grad;
    };
    return output;
}

auto tanh(const ValuePtr &v) noexcept -> ValuePtr
{
    auto output = create_value(std::tanh(v->data), {v}, "tanh", v->requires_grad, false);
    if (!v->requires_grad)
    {
        output->backward_ = nullptr;
        return output;
    }
    std::weak_ptr<Value> wv = v, wout = output;
    output->backward_ = [wv, wout]()
    {
        auto out = wout.lock();
        auto val = wv.lock();
        assert(out && val);
        val->grad += (1 - out->data * out->data) * out->grad;
    };
    return output;
}

// ------ DOT generation and visualization -----

std::string Value::to_dot() const
{
    std::stringstream ss;
    ss << "digraph G {\n";
    ss << "  rankdir=LR;\n";
    ss << "  node [shape=record, fontname=\"Arial\"];\n";
    ss << "  edge [fontname=\"Arial\"];\n";

    std::unordered_map<const Value *, int> node_ids;
    int id = 0;

    build_dot(ss, node_ids, id);

    ss << "}\n";
    return ss.str();
}

void Value::build_dot(std::stringstream &ss, std::unordered_map<const Value *, int> &node_ids, int &id) const
{
    // Avoid processing the same node multiple times
    auto self = this;
    if (node_ids.find(self) != node_ids.end())
    {
        return;
    }

    // add node to node ids
    int current_id = id++;
    node_ids[self] = current_id;

    // node labels: include op if present
    std::stringstream label;
    if (!op.empty())
    {
        label << "Op: " << op << "\\n";
    }
    label << "Value: " << data << "\\nGrad: " << grad;

    // style for leaf nodes
    bool is_leaf = prev_.empty();
    const char *shape = is_leaf ? "ellipse" : "box";
    const char *fillcolor = is_leaf ? "lightgreen" : "lightblue";

    ss << "  node_" << current_id << "[label=\"" << label.str()
       << "\", shape=" << shape << ", style=filled, fillcolor=" << fillcolor << "];\n";

    // traverse the previous nodes
    for (const auto &child : prev_)
    {
        if (node_ids.find(child.get()) == node_ids.end())
        {
            child->build_dot(ss, node_ids, id);
        }
        int prev_id = node_ids[child.get()];
        ss << "  node_" << prev_id << " -> node_" << current_id << ";\n";
    }
}

void Value::visualize(const std::string &filename) const
{
    // Write DOT
    std::ofstream dot_file(filename + ".dot");
    dot_file << to_dot();
    dot_file.close();

    // Render PNG with Graphviz
    std::string command = "dot -Tpng " + filename + ".dot -o " + filename + ".png";
    if (std::system(command.c_str()) != 0)
    {
        std::cerr << "Failed to run dot. Is Graphviz installed and on PATH?\n";
    }
}
