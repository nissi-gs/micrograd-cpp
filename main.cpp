#include <iostream>
#include <random>
#include "Value.h"
#include "nn.h"

ValuePtr sq_loss(const std::vector<ValuePtr> &predictions, const std::vector<ValuePtr> &targets)
// Squared error loss
{
    if (predictions.size() != targets.size())
    {
        throw std::runtime_error("Predictions and targets size mismatch");
    }

    ValuePtr total_loss = create_value(0.0);
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        auto diff = predictions[i] - targets[i];
        total_loss = total_loss + diff * diff;
    }
    return total_loss;
}

void update_parameters(const std::vector<ValuePtr> &params, double learning_rate)
{
    for (auto &p : params)
    {
        if (p->needs_grad())
        {
            p->set_data(p->get_data() - learning_rate * p->get_grad());
        }
    }
}

int main()
{
    // A simple training loop for a tiny dataset on a tiny MLP
    std::vector<std::vector<ValuePtr>> xs = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}};

    std::vector<ValuePtr> ys = {
        create_value(1.0),
        create_value(-1.0),
        create_value(-1.0),
        create_value(1.0)};

    MLP model(3, {4, 4, 1}); // 3-input, 2 hidden layers of 4 neurons each, 1-output
    std::cout << "Number of parameters: " << model.parameters().size() << "\n";

    std::vector<ValuePtr> predictions;
    predictions.reserve(ys.size());

    int num_epochs = 20;
    double learning_rate = 0.15;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {

        predictions.clear();
        for (const auto &x : xs)
        {
            auto pred = model(x);
            predictions.push_back(pred[0]);
        }

        auto loss = sq_loss(predictions, ys);
        std::cout << "Epoch " << epoch << " loss: " << loss->get_data() << "\n";
        loss->backward(1, false, true);
        // Collect gradient stats (after backward, before update)
        const auto params = model.parameters();
        double max_grad = 0.0;
        double sum_grad = 0.0;
        size_t grad_count = 0;
        size_t tiny_count = 0;
        for (auto &p : params)
        {
            if (!p->needs_grad())
                continue;
            double g = std::abs(p->get_grad());
            max_grad = std::max(max_grad, g);
            sum_grad += g;
            ++grad_count;
            if (g < 1e-6)
                ++tiny_count;
        }
        double mean_grad = grad_count ? (sum_grad / grad_count) : 0.0;
        double frac_tiny = grad_count ? static_cast<double>(tiny_count) / grad_count : 0.0;
        std::cout << "Grad stats: max=" << max_grad
                  << ", mean=" << mean_grad
                  << ", tiny_fraction=" << frac_tiny << "\n";

        update_parameters(params, learning_rate);

        // visualize the computation graph for the last loss
        if (epoch == num_epochs - 1)
        {
            loss->visualize("mlp_graph.dot");
        }
        model.zero_grad();
    }

    return 0;
}
