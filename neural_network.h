//
// Created by afan on 2024/7/6.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef enum {
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    LEAKY_RELU
} ActivationFunction;

typedef enum {
    MSE,
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY,
    MAE
} LossFunction;

typedef struct {
    double *weights;
    double bias;
    double output;
    double delta;
} Neuron;

typedef struct {
    int num_neurons;
    Neuron *neurons;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    LossFunction loss_function;
    ActivationFunction activation;
} NeuralNetwork;

// 函數宣告
NeuralNetwork* create_network(int num_layers, const int *neurons_per_layer, ActivationFunction activation, LossFunction lossFunction);
void free_network(NeuralNetwork *network);
void initialize_weights(NeuralNetwork *network, double min, double max);
void forward_propagation(NeuralNetwork *network, const double *input);
void backward_propagation(NeuralNetwork *network, double *expected_output, double learning_rate);
double activation_function(double x, ActivationFunction func);
double activation_function_derivative(double x, ActivationFunction func);
double compute_loss(double *output, double *expected_output, int length, LossFunction loss_function);
double compute_loss_derivative(double output, double expected_output, LossFunction loss_function);
double compute_error(double output, double expected_output);
void save_network(NeuralNetwork *network, const char *filename);
NeuralNetwork *load_network(const char *filename);

#endif // NEURAL_NETWORK_H
