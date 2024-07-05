//
// Created by afan on 2024/7/6.
//

#include "neural_network.h"

NeuralNetwork *create_network(int num_layers, const int *neurons_per_layer, ActivationFunction activation, LossFunction lossFunction) {
    NeuralNetwork *network = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    if (!network) {
        printf("Memory allocation failed.\n");
        return NULL;
    }

    network->num_layers = num_layers;
    network->activation = activation;
    network->loss_function = lossFunction;
    network->layers = (Layer *) malloc(num_layers * sizeof(Layer));
    if (!network->layers) {
        printf("Memory allocation failed.\n");
        free(network);
        return NULL;
    }

    for (int i = 0; i < num_layers; i++) {
        network->layers[i].num_neurons = neurons_per_layer[i];
        network->layers[i].neurons = (Neuron *) malloc(neurons_per_layer[i] * sizeof(Neuron));
        if (!network->layers[i].neurons) {
            printf("Memory allocation failed.\n");
            free_network(network);
            return NULL;
        }
        for (int j = 0; j < neurons_per_layer[i]; j++) {
            network->layers[i].neurons[j].weights = (i == 0) ? NULL : (double *) malloc(
                    neurons_per_layer[i - 1] * sizeof(double));
            if (i > 0 && !network->layers[i].neurons[j].weights) {
                printf("Memory allocation failed.\n");
                free_network(network);
                return NULL;
            }
        }
    }

    return network;
}


void initialize_weights(NeuralNetwork *network, double min, double max) {
    for (int i = 1; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            for (int k = 0; k < network->layers[i - 1].num_neurons; k++) {
                // 根據 activation 選擇初始化ㄟ方式
                if (network->activation == RELU || network->activation == LEAKY_RELU) {
                    // He 初始化
                    network->layers[i].neurons[j].weights[k] = sqrt(2.0 / network->layers[i - 1].num_neurons) * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
                } else {
                    // Xavier 初始化
                    network->layers[i].neurons[j].weights[k] = sqrt(1.0 / network->layers[i - 1].num_neurons) * ((double)rand() / RAND_MAX);
                }
            }
            network->layers[i].neurons[j].bias = min + ((double)rand() / RAND_MAX) * (max - min);
        }
    }
}

double activation_function(double x, ActivationFunction func) {
    switch (func) {
        case RELU:
            return fmax(0, x);
        case SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case TANH:
            return tanh(x);
        case LINEAR:
            return x;
        case LEAKY_RELU:
            return x > 0 ? x : 0.01 * x;
        default:
            return x;
    }
}

double activation_function_derivative(double x, ActivationFunction func) {
    switch (func) {
        case RELU:
            return x > 0 ? 1 : 0;
        case SIGMOID:
            return x * (1 - x);
        case TANH:
            return 1 - x * x;
        case LINEAR:
            return 1;
        case LEAKY_RELU:
            return x > 0 ? 1 : 0.01;
        default:
            return 1;
    }
}

double compute_loss(double *output, double *expected_output, int length, LossFunction loss_function) {
    double loss = 0.0;
    switch (loss_function) {
        case MSE:
            for (int i = 0; i < length; i++) {
                loss += pow(output[i] - expected_output[i], 2);
            }
            return loss / length;
        case CROSS_ENTROPY:
            for (int i = 0; i < length; i++) {
                loss -= expected_output[i] * log(output[i]) + (1 - expected_output[i]) * log(1 - output[i]);
            }
            return loss / length;
        case BINARY_CROSS_ENTROPY:
            for (int i = 0; i < length; i++) {
                loss -= expected_output[i] * log(output[i]) + (1 - expected_output[i]) * log(1 - output[i]);
            }
            return loss / length;
        case MAE:
            for (int i = 0; i < length; i++) {
                loss += fabs(output[i] - expected_output[i]);
            }
            return loss / length;
        default:
            return loss;
    }
}

double compute_loss_derivative(double output, double expected_output, LossFunction loss_function) {
    switch (loss_function) {
        case MSE:
            return output - expected_output;
        case CROSS_ENTROPY:
        case BINARY_CROSS_ENTROPY:
            return (output - expected_output) / (output * (1 - output));
        case MAE:
            return (output > expected_output) ? 1 : -1;
        default:
            return 0;
    }
}

double compute_error(double output, double expected_output) {
    return pow(output - expected_output, 2) / 2;
}

void forward_propagation(NeuralNetwork *network, const double *input) {
    for (int i = 0; i < network->layers[0].num_neurons; i++) {
        network->layers[0].neurons[i].output = input[i];
    }
    for (int i = 1; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            double sum = network->layers[i].neurons[j].bias;
            for (int k = 0; k < network->layers[i - 1].num_neurons; k++) {
                sum += network->layers[i].neurons[j].weights[k] * network->layers[i - 1].neurons[k].output;
            }
            network->layers[i].neurons[j].output = activation_function(sum, network->activation);
        }
    }
}

void backward_propagation(NeuralNetwork *network, double *expected_output, double learning_rate) {
    for (int i = network->num_layers - 1; i > 0; i--) {
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            double error = 0.0;
            if (i == network->num_layers - 1) {
                // 計算輸出層的 delta
                error = compute_loss_derivative(network->layers[i].neurons[j].output, expected_output[j], network->loss_function)
                        * activation_function_derivative(network->layers[i].neurons[j].output, network->activation);
            } else {
                // 計算隱藏層的 delta
                for (int k = 0; k < network->layers[i + 1].num_neurons; k++) {
                    error += network->layers[i + 1].neurons[k].weights[j] * network->layers[i + 1].neurons[k].delta;
                }
                error *= activation_function_derivative(network->layers[i].neurons[j].output, network->activation);
            }
            network->layers[i].neurons[j].delta = error;
        }
    }
    for (int i = 1; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            for (int k = 0; k < network->layers[i - 1].num_neurons; k++) {
                network->layers[i].neurons[j].weights[k] -= learning_rate * network->layers[i].neurons[j].delta * network->layers[i - 1].neurons[k].output;
            }
            network->layers[i].neurons[j].bias -= learning_rate * network->layers[i].neurons[j].delta;
        }
    }
}

void free_network(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            if (network->layers[i].neurons[j].weights != NULL) {
                free(network->layers[i].neurons[j].weights);
            }
        }
        free(network->layers[i].neurons);
    }
    free(network->layers);
    free(network);
}

void save_network(NeuralNetwork *network, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error opening file %s for writing.\n", filename);
        return;
    }

    // 網路結構
    fprintf(fp, "%d ", network->num_layers);
    fprintf(fp, "%d ", network->activation);
    fprintf(fp, "%d\n", network->loss_function);

    for (int i = 0; i < network->num_layers; i++) {
        fprintf(fp, "%d ", network->layers[i].num_neurons);
    }
    fprintf(fp, "\n");

    // 權重
    for (int i = 1; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            for (int k = 0; k < network->layers[i - 1].num_neurons; k++) {
                fprintf(fp, "%lf ", network->layers[i].neurons[j].weights[k]);
            }
            fprintf(fp, "%lf\n", network->layers[i].neurons[j].bias);
        }
    }

    fclose(fp);
}

NeuralNetwork *load_network(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file %s for reading.\n", filename);
        return NULL;
    }

    NeuralNetwork *network = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    if (!network) {
        printf("Memory allocation failed.\n");
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d", &network->num_layers);
    fscanf(fp, "%d", (int *)&network->activation);
    fscanf(fp, "%d", (int *)&network->loss_function);

    network->layers = (Layer *) malloc(network->num_layers * sizeof(Layer));
    if (!network->layers) {
        printf("Memory allocation failed.\n");
        fclose(fp);
        free(network);
        return NULL;
    }

    for (int i = 0; i < network->num_layers; i++) {
        fscanf(fp, "%d", &network->layers[i].num_neurons);
    }

    for (int i = 1; i < network->num_layers; i++) {
        network->layers[i].neurons = (Neuron *) malloc(network->layers[i].num_neurons * sizeof(Neuron));
        if (!network->layers[i].neurons) {
            printf("Memory allocation failed.\n");
            fclose(fp);
            free_network(network);
            return NULL;
        }

        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            network->layers[i].neurons[j].weights = (double *) malloc(network->layers[i - 1].num_neurons * sizeof(double));
            if (!network->layers[i].neurons[j].weights) {
                printf("Memory allocation failed.\n");
                fclose(fp);
                free_network(network);
                return NULL;
            }

            for (int k = 0; k < network->layers[i - 1].num_neurons; k++) {
                fscanf(fp, "%lf", &network->layers[i].neurons[j].weights[k]);
            }
            fscanf(fp, "%lf", &network->layers[i].neurons[j].bias);
        }
    }

    fclose(fp);
    return network;
}
