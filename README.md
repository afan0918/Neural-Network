總之就是用 c 刻的類神經網路，c 比 c++ 刻起來麻煩太多了 = =

It is a lightweight neural network framework implemented in C. It provides functionalities to create, train, and use neural networks for various machine learning tasks.

## Features

- Support for multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Leaky ReLU).
- Different loss functions available (MSE, Cross Entropy, Binary Cross Entropy, MAE).
- Easily customizable network architecture with adjustable layer sizes and configurations.
- Includes functions for training with backpropagation and predicting outputs.

## Usage

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/afan0918/Neural-Network.git
   cd Neural-Network
   ```

2. Compile the framework:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

### Example Usage

```c
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

    return 0;
}
```
