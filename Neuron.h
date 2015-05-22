#pragma once

// Get a random floating point value in between 0 and 1
float GetRandom()
{
    return float(rand())/float(RAND_MAX);
}

// Neuron class containing data for each node of a neural network
struct Neuron
{
    float bias;             // Bias value
    float value;            // Output value of this node: the activation
    float delta;            // Delta value used in backpropagation training

    std::vector<float> weights; // List of inputs weights from previous layer of neurons

    // Activation function
    // Before this method is called, the value is the weighted sum of inputs plus the bias
    // After this method, the value will be the activation value: the output of the node
    void Activate()
    {
        value = 1/(1 + exp(-value));    // Sigmoid function
    };
};


// Layer class containing a list of neurons
struct Layer
{
    std::vector<Neuron> neurons;
};


