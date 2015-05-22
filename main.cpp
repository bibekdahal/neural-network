#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "Network.h"

int main()
{
    srand((unsigned)time(0));
    
    // Create neural network with input layer(2 neurons), output layer(1 neuron) and a hidden layer(2 neurons)
    // and learning-rate of 0.2
    std::vector<size_t> layersNeuronsCnt = {2, 2, 1};
    float learningRate = 0.2f;
    Network network(learningRate, layersNeuronsCnt);

    // Create two list, one to contain inputs and other to contain output
    std::vector<float> inputs(2);
    std::vector<float> output(1);
    
    // Train the network with sample inputs and output
    // Since there is only 4 inputs-output pair for x-or, repeat same training data for 100000 times
    for (int i=0; i<100000; ++i)
    {
        inputs[0] = 0.0f; inputs[1] = 0.0f; output[0] = 0.0f;
        network.Supervise(inputs, output);
        inputs[0] = 1.0f; inputs[1] = 0.0f; output[0] = 1.0f;
        network.Supervise(inputs, output);
        inputs[0] = 0.0f; inputs[1] = 1.0f; output[0] = 1.0f;
        network.Supervise(inputs, output);
        inputs[0] = 1.0f; inputs[1] = 1.0f; output[0] = 0.0f;
        network.Supervise(inputs, output);
    }
    
    // Now run the network with all 4 possible inputs and display the ouput

    inputs[0] = 1.0f; inputs[1] = 0.0f;
    network.Run(inputs);
    network.DisplayIO();

    inputs[0] = 0.0f; inputs[1] = 1.0f;
    network.Run(inputs);
    network.DisplayIO();

    inputs[0] = 1.0f; inputs[1] = 1.0f;
    network.Run(inputs);
    network.DisplayIO();

    inputs[0] = 0.0f; inputs[1] = 0.0f;
    network.Run(inputs);
    network.DisplayIO();

    return 0;
}
