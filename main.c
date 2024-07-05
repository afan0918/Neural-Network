#include "neural_network.h"

// xor test
int main() {
    int layers[] = {2, 6, 1};
    NeuralNetwork *network = create_network(3, layers, SIGMOID, CROSS_ENTROPY);
    initialize_weights(network, -1.0, 1.0);

    // XOR training data
    double inputs[4][2] = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
    };
    double outputs[4][1] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    int epochs = 10000;
    double learning_rate = 0.1;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < 4; i++) {
            forward_propagation(network, inputs[i]);
            total_loss += compute_error(network->layers[network->num_layers - 1].neurons[0].output, outputs[i][0]);
            backward_propagation(network, outputs[i], learning_rate);
        }
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / 4);
            for (int i = 0; i < 4; i++) {
                forward_propagation(network, inputs[i]);
                printf("Input: [%f, %f], Predicted Output: %f, Expected Output: %f\n",
                       inputs[i][0], inputs[i][1], network->layers[network->num_layers - 1].neurons[0].output, outputs[i][0]);
            }
        }
    }

    // Testing
    printf("\n\nTesting\n\n");
    for (int i = 0; i < 4; i++) {
        forward_propagation(network, inputs[i]);
        printf("Input: [%f, %f], Predicted Output: %f, Expected Output: %f\n",
               inputs[i][0], inputs[i][1], network->layers[network->num_layers - 1].neurons[0].output, outputs[i][0]);
    }

    save_network(network, "my_network.txt");
    NeuralNetwork *loaded_network = load_network("my_network.txt");

    printf("\n\nLoading model\n\n");

    for (int i = 0; i < 4; i++) {
        forward_propagation(loaded_network, inputs[i]);
        printf("Input: [%f, %f], Predicted Output: %f, Expected Output: %f\n",
               inputs[i][0], inputs[i][1], loaded_network->layers[loaded_network->num_layers - 1].neurons[0].output, outputs[i][0]);
    }

    free_network(network);
    return 0;
}
