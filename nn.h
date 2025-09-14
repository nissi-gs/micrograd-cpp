#pragma once
#include "Value.h"
#include <random>

class Module
{
public:
    virtual ~Module() = default;
    virtual std::vector<ValuePtr> parameters() = 0;
    void zero_grad()
    {
        for (auto &p : parameters())
            p->zero_grad();
    }
};

class Neuron : public Module
{
public:
    Neuron(int input_size) : nin(input_size)
    {
        std::mt19937 rn_eng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        weights.reserve(nin);

        for (int i = 0; i < nin; ++i)
        {
            weights.push_back(create_parameter(dist(rn_eng)));
        }
        bias = create_parameter(0.0);
    }

    ValuePtr operator()(const std::vector<ValuePtr> &x)
    {
        if (x.size() != static_cast<size_t>(nin))
        {
            throw std::runtime_error("Input size mismatch");
        }
        auto output = bias;

        for (int i = 0; i < nin; ++i)
        {
            output = output + x[i] * weights[i];
        }

        output = tanh(output);
        return output;
    }

    std::vector<ValuePtr> parameters() override
    {
        std::vector<ValuePtr> params = weights;
        params.push_back(bias);
        return params;
    }

private:
    int nin;
    std::vector<ValuePtr> weights;
    ValuePtr bias;
};

class Layer : public Module
{
private:
    int nin;
    int nout;
    std::vector<Neuron> neurons;

public:
    Layer(int input_size, int output_size) : nin(input_size), nout(output_size)
    {
        neurons.reserve(nout);

        for (int i = 0; i < nout; ++i)
        {
            neurons.emplace_back(nin);
        }
    }

    std::vector<ValuePtr> operator()(const std::vector<ValuePtr> &x)
    {
        std::vector<ValuePtr> outputs;
        outputs.reserve(nout);
        for (auto &neuron : neurons)
        {
            outputs.push_back(neuron(x));
        }
        return outputs;
    }

    std::vector<ValuePtr> parameters() override
    {
        std::vector<ValuePtr> params;
        for (auto &neuron : neurons)
        {
            auto neuron_params = neuron.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};

class MLP : public Module
{
private:
    int nin;
    std::vector<int> nouts;
    std::vector<Layer> layers;

public:
    MLP(int input_size, const std::vector<int> &layer_sizes) : nin(input_size), nouts(layer_sizes)
    {
        layers.reserve(nouts.size());
        int current_input_size = nin;
        for (int size : nouts)
        {
            layers.emplace_back(current_input_size, size);
            current_input_size = size;
        }
    }

    std::vector<ValuePtr> operator()(const std::vector<ValuePtr> &x)
    {
        std::vector<ValuePtr> output = x;
        for (auto &layer : layers)
        {
            output = layer(output);
        }
        return output;
    }

    std::vector<ValuePtr> parameters() override
    {
        std::vector<ValuePtr> params;
        for (auto &layer : layers)
        {
            auto layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};