#pragma once

#include "Neuron.h"

// Network consisting of a list of neural layers
class Network
{
public:

    // Create a Neural Network with given learning-rate and given neurons count for its layers
    Network(float learning_rate, const std::vector<size_t>& layersNeuronsCnt)
        : learning_rate(learning_rate)
    {
        // A network is valid after its is succesfully created
        is_valid = false;
        
        // The number of neurons-count provided is the number of layers: neurons-count is provided for each layer
        size_t layersCnt = layersNeuronsCnt.size();
        if (layersCnt < 2)          // There must be at least an input and an output layer
            return;
        layers.resize(layersCnt);

        // Set given neurons-count for the layers and initialize the bias and weights with random values
        for (size_t i=0; i<layers.size(); ++i)
        {
            layers[i].neurons.resize(layersNeuronsCnt[i]);
            for (size_t j=0; j<layers[i].neurons.size(); ++j)
            {
                Neuron& neuron = layers[i].neurons[j];
                // For the first layer, there is no need of weights and bias values
                // For the rest, the number of weights is equal to number of neurons in previous layer
                if (i != 0)
                {
                    neuron.bias = GetRandom();
                    neuron.weights.resize(layersNeuronsCnt[i-1]);
                    for (size_t k=0; k<neuron.weights.size(); ++k)
                        neuron.weights[k] = GetRandom();
                }
            }
        }
        
        // The network is created
        is_valid = true;
    }

    bool IsValid() const { return is_valid; }
    
    // Run a set of inputs through this network
    // Returns true on success
    bool Run(const std::vector<float>& inputs)
    {
        if (!is_valid)
            return false;

        if (inputs.size() != layers[0].neurons.size())      // number of given inputs must be equal to number of neurons in first layer
            return false;
        
        // Set the values of input layer
        for (size_t j=0; j<layers[0].neurons.size(); ++j)
            layers[0].neurons[j].value = inputs[j];
        
        // For each of the rest of the layers, 
        // calculate weighted sum plus the bias for each neuron and pass it through activation function
        for (size_t i=1; i<layers.size(); ++i)
        {
            for (size_t j=0; j<layers[i].neurons.size(); ++j)
            {
                Neuron& neuron = layers[i].neurons[j];
                neuron.value = 0;
                for (size_t k=0; k<neuron.weights.size(); ++k)
                    neuron.value += layers[i-1].neurons[k].value * neuron.weights[k];  // weighted sum
                neuron.value += neuron.bias;                                           // plus the bias
                neuron.Activate();              // Activation
            }
        }
        return true;
    }
    
    // Get the output layer, set of output neurons
    const Layer& GetOutputLayer() const { return layers[layers.size()-1]; }
    

    // Train the network with given set of inputs and outputs : The Backpropagation Algorithm
    // Returns true on success
    bool Supervise(const std::vector<float>& inputs, const std::vector<float>& outputs)
    {
        if (!is_valid)
            return false;

        // number of given inputs and outputs must match number of neurons in respective layers
        if (inputs.size() != layers[0].neurons.size() || outputs.size() != layers[layers.size()-1].neurons.size())
            return false;
        
        // First the run the inputs through the network and obtain the outputs
        if (!Run(inputs))
            return false;
        
        // Next calculate the delta values for the output layer:
        //    delta = obtained_output * (1 - obtained_output) * (given_output - obtained_output)
        for (size_t i=0; i<layers[layers.size()-1].neurons.size(); ++i)
        {
            Neuron& neuron = layers[layers.size()-1].neurons[i];
            neuron.delta = neuron.value * (1 - neuron.value) * (outputs[i] - neuron.value);
        }
        // Then calculate the delta values for rest of the layers:
        //    delta = obtained_output * (1 - obtained_output) * SUM(i+1)[delta * weight]  // SUM(i+1) is sum of neurons of (i+1)-th layer
        for (size_t i=layers.size()-2; i>0; --i)
        {
            for (size_t j=0; j<layers[i].neurons.size(); ++j)
            {
                Neuron& neuron = layers[i].neurons[j];
                float sum = 0.0f;
                for (size_t k=0; k<layers[i+1].neurons.size(); ++k)
                    sum += layers[i+1].neurons[k].delta * layers[i+1].neurons[k].weights[j];
                neuron.delta = neuron.value * (1-neuron.value) * sum;
            }
        }
        
        // Bias and weights are adjusted according to the delta values
        //    bias += learning-rate * delta
        //    weight += SUM(learning-rate * input * delta)  // inputs are outputs of (i-1)-th layer neurons
        for (size_t i=1; i<layers.size(); ++i)
        {
            for (size_t j=0; j<layers[i].neurons.size(); ++j)
            {
                Neuron& neuron = layers[i].neurons[j];
                neuron.bias += learning_rate * neuron.delta;
                for (size_t k=0; k<neuron.weights.size(); ++k)
                    neuron.weights[k] += learning_rate * layers[i-1].neurons[k].value * neuron.delta;
            }
        }
        return true;
    }

    // Display the bias and weights of neurons of each layer
    void Display() const
    {
        if (!is_valid)
            return;

        for (size_t i=0; i<layers.size(); ++i)
        {
            std::cout << "Layer #" << i << std::endl;
            for (size_t j=0; j<layers[i].neurons.size(); ++j)
            {
                const Neuron& neuron = layers[i].neurons[j];
                std::cout << "\tNeuron #" << j << std::endl;
                std::cout << "\t\t Bias: " << neuron.bias << std::endl;
                std::cout << "\t\t Value: " << neuron.value << std::endl;
                
                if (i!=0)
                    std::cout << "\t\t Weights: ";
                for (size_t k=0; k<neuron.weights.size(); ++k)
                    std::cout << neuron.weights[k] << "\t";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    // Display the values of neurons in input and output layers
    void DisplayIO() const
    {
        if (!is_valid)
            return;

        std::cout << "Inputs: ";
        for (size_t i=0; i<layers[0].neurons.size(); ++i)
            std::cout << layers[0].neurons[i].value << " ";

        std::cout << std::endl << "Outputs: ";
        for (size_t i=0; i<layers[layers.size()-1].neurons.size(); ++i)
            std::cout << layers[layers.size()-1].neurons[i].value << " ";

        std::cout << std::endl << std::endl;
    }

private:
    std::vector<Layer> layers;  // List of layers
    float learning_rate;        // learning-rate constant used in Backpropagation algorithm

    bool is_valid;               // has the network been successfully created?
};
