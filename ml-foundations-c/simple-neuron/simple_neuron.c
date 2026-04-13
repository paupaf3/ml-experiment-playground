#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    float *weights;
    float bias;
    float learning_rate;
    int input_count;
} Neuron;

void print_neuron_state(Neuron *n, const char *label) {
    printf("%s\n", label);
    for (int i = 0; i < n->input_count; i++) {
        printf("  weight[%d] = %.4f\n", i, n->weights[i]);
    }
    printf("  bias      = %.4f\n", n->bias);
    printf("  lr        = %.4f\n", n->learning_rate);
}

void init_neuron(Neuron *n, int inputs, float lr) {
    n->input_count = inputs;
    n->learning_rate = lr;
    n->weights = (float *)malloc(inputs * sizeof(float));
    for (int i = 0; i < inputs; i++) {
        n->weights[i] = ((float)rand() / RAND_MAX) * 2 -
                        1; // Random weights between -1 and 1
    }
    n->bias =
        ((float)rand() / RAND_MAX) * 2 - 1; // Random bias between -1 and 1
}

void cleanup_neuron(Neuron *n) {
    free(n->weights);
    n->weights = NULL;
    n->input_count = 0;
    n->learning_rate = 0.0;
    n->bias = 0.0;
}

float activation(float sum) {
    return (sum > 0) ? 1.0f : 0.0f;
}

float feedforward(Neuron *n, float inputs[]) {
    float sum = n->bias;
    printf("Feedforward details:\n");
    printf("  start sum (bias): %.4f\n", sum);
    for (int i = 0; i < n->input_count; i++) {
        float contribution = n->weights[i] * inputs[i];
        sum += contribution;
        printf("  w[%d] * x[%d] = %.4f * %.4f = %.4f (running sum: %.4f)\n", i,
               i, n->weights[i], inputs[i], contribution, sum);
    }
    float output = activation(sum);
    printf("  activation(%.4f) -> %.1f\n", sum, output);
    return output;
}

void train(Neuron *n, float inputs[], float target) {
    float output = feedforward(n, inputs);
    float error = target - output;

    printf("Training details:\n");
    printf("  target = %.1f, output = %.1f, error = %.1f\n", target, output,
           error);

    for (int i = 0; i < n->input_count; i++) {
        float old_weight = n->weights[i];
        float delta = n->learning_rate * error * inputs[i];
        n->weights[i] += delta;
        printf("  weight[%d]: %.4f + %.4f -> %.4f\n", i, old_weight, delta,
               n->weights[i]);
    }
    float old_bias = n->bias;
    float bias_delta = n->learning_rate * error;
    n->bias += bias_delta;
    printf("  bias: %.4f + %.4f -> %.4f\n", old_bias, bias_delta, n->bias);
}

int main() {
    srand(time(NULL));
    printf("=== Simple Neuron Learning Trace ===\n");

    // 1. Setup
    Neuron my_neuron;
    init_neuron(&my_neuron, 2, 0.1); // Example: 2 inputs, 0.1 learning rate
    print_neuron_state(&my_neuron, "Initial neuron state:");

    // 2. Example Inputs (e.g., for an AND gate)
    float inputs[2] = {1.0, 0.0};
    float target = 0.0;
    printf("\nTraining sample:\n");
    printf("  inputs = [%.1f, %.1f]\n", inputs[0], inputs[1]);
    printf("  target = %.1f\n", target);

    // 3. Process
    printf("\nBefore training:\n");
    float result = feedforward(&my_neuron, inputs);
    printf("Prediction before training: %.1f\n", result);

    // 4. Learning
    printf("\nApplying one training step...\n");
    train(&my_neuron, inputs, target);
    print_neuron_state(&my_neuron, "Neuron state after training:");

    printf("\nAfter training:\n");
    float updated_result = feedforward(&my_neuron, inputs);
    printf("Prediction after training: %.1f\n", updated_result);

    // 5. Cleanup
    cleanup_neuron(&my_neuron);

    return 0;
}