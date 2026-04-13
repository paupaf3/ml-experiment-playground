#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Neuron {
    std::vector<float> weights;
    float bias;
    float learning_rate;
};

void print_neuron_state(const Neuron &n, const std::string &label) {
    std::cout << label << "\n";
    for (size_t i = 0; i < n.weights.size(); i++) {
        std::cout << "  weight[" << i << "] = " << std::fixed
                  << std::setprecision(4) << n.weights[i] << "\n";
    }
    std::cout << "  bias      = " << std::fixed << std::setprecision(4)
              << n.bias << "\n";
    std::cout << "  lr        = " << std::fixed << std::setprecision(4)
              << n.learning_rate << "\n";
}

Neuron init_neuron(int inputs, float lr) {
    Neuron n;
    n.learning_rate = lr;
    n.weights.resize(inputs);

    for (int i = 0; i < inputs; i++) {
        n.weights[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f -
                       1.0f; // Random weights between -1 and 1
    }
    n.bias = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f -
             1.0f; // Random bias between -1 and 1

    return n;
}

float activation(float sum) {
    return (sum > 0.0f) ? 1.0f : 0.0f;
}

float feedforward(const Neuron &n, const std::vector<float> &inputs) {
    float sum = n.bias;
    std::cout << "Feedforward details:\n";
    std::cout << "  start sum (bias): " << std::fixed << std::setprecision(4)
              << sum << "\n";

    for (size_t i = 0; i < n.weights.size(); i++) {
        float contribution = n.weights[i] * inputs[i];
        sum += contribution;
        std::cout << "  w[" << i << "] * x[" << i << "] = " << std::fixed
                  << std::setprecision(4) << n.weights[i] << " * " << inputs[i]
                  << " = " << contribution << " (running sum: " << sum << ")\n";
    }

    float output = activation(sum);
    std::cout << "  activation(" << std::fixed << std::setprecision(4) << sum
              << ") -> " << std::setprecision(1) << output << "\n";
    return output;
}

void train(Neuron &n, const std::vector<float> &inputs, float target) {
    float output = feedforward(n, inputs);
    float error = target - output;

    std::cout << "Training details:\n";
    std::cout << "  target = " << std::fixed << std::setprecision(1) << target
              << ", output = " << output << ", error = " << error << "\n";

    for (size_t i = 0; i < n.weights.size(); i++) {
        float old_weight = n.weights[i];
        float delta = n.learning_rate * error * inputs[i];
        n.weights[i] += delta;
        std::cout << "  weight[" << i << "]: " << std::fixed
                  << std::setprecision(4) << old_weight << " + " << delta
                  << " -> " << n.weights[i] << "\n";
    }

    float old_bias = n.bias;
    float bias_delta = n.learning_rate * error;
    n.bias += bias_delta;
    std::cout << "  bias: " << std::fixed << std::setprecision(4) << old_bias
              << " + " << bias_delta << " -> " << n.bias << "\n";
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::cout << "=== Simple Neuron Learning Trace ===\n";

    // 1. Setup
    Neuron my_neuron =
        init_neuron(2, 0.1f); // Example: 2 inputs, 0.1 learning rate
    print_neuron_state(my_neuron, "Initial neuron state:");

    // 2. Example Inputs (e.g., for an AND gate)
    std::vector<float> inputs = {1.0f, 0.0f};
    float target = 0.0f;
    std::cout << "\nTraining sample:\n";
    std::cout << "  inputs = [" << std::fixed << std::setprecision(1)
              << inputs[0] << ", " << inputs[1] << "]\n";
    std::cout << "  target = " << target << "\n";

    // 3. Process
    std::cout << "\nBefore training:\n";
    float result = feedforward(my_neuron, inputs);
    std::cout << "Prediction before training: " << std::fixed
              << std::setprecision(1) << result << "\n";

    // 4. Learning
    std::cout << "\nApplying one training step...\n";
    train(my_neuron, inputs, target);
    print_neuron_state(my_neuron, "Neuron state after training:");

    std::cout << "\nAfter training:\n";
    float updated_result = feedforward(my_neuron, inputs);
    std::cout << "Prediction after training: " << std::fixed
              << std::setprecision(1) << updated_result << "\n";

    return 0;
}
