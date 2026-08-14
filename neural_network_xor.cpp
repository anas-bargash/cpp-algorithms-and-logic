#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Sigmoid activation function & its derivative
double sigmoid(double x) { 
    return 1.0 / (1.0 + std::exp(-x)); 
}

double sigmoidDerivative(double x) { 
    return x * (1.0 - x); 
}

int main() {
    std::srand(std::time(0));

    // Dataset: XOR Gate Inputs (X1, X2) and Target Outputs (Y)
    std::vector<std::vector<double>> inputs = {
        {0, 0}, 
        {0, 1}, 
        {1, 0}, 
        {1, 1}
    };
    std::vector<double> targets = {0, 1, 1, 0};

    // Architecture: 2 Inputs -> 2 Hidden Neurons -> 1 Output Neuron
    double w_hidden[2][2]; // Input to Hidden weights
    double b_hidden[2];    // Hidden bias
    double w_output[2];    // Hidden to Output weights
    double b_output;       // Output bias

    // Random initialization of weights and biases [0, 1]
    for (int i = 0; i < 2; ++i) {
        b_hidden[i] = ((double)rand() / RAND_MAX);
        w_output[i] = ((double)rand() / RAND_MAX);
        for (int j = 0; j < 2; ++j) {
            w_hidden[i][j] = ((double)rand() / RAND_MAX);
        }
    }
    b_output = ((double)rand() / RAND_MAX);

    double learning_rate = 0.5;
    int epochs = 10000;

    std::cout << "--- Training Neural Network on XOR Problem From Scratch ---\n\n";

    // Training Loop (Training from Scratch)
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t sample = 0; sample < inputs.size(); ++sample) {

            // 1. Forward Pass (Hidden Layer)
            double hidden_layer[2];
            for (int i = 0; i < 2; ++i) {
                double sum = b_hidden[i];
                for (int j = 0; j < 2; ++j) {
                    sum += inputs[sample][j] * w_hidden[j][i];
                }
                hidden_layer[i] = sigmoid(sum);
            }

            // Forward Pass (Output Layer)
            double output_sum = b_output;
            for (int i = 0; i < 2; ++i) {
                output_sum += hidden_layer[i] * w_output[i];
            }
            double final_output = sigmoid(output_sum);

            // 2. Backpropagation (Output Layer Error & Gradient)
            double error = targets[sample] - final_output;
            double d_output = error * sigmoidDerivative(final_output);

            // Backpropagation (Hidden Layer Error & Gradient)
            double d_hidden[2];
            for (int i = 0; i < 2; ++i) {
                d_hidden[i] = d_output * w_output[i] * sigmoidDerivative(hidden_layer[i]);
            }

            // 3. Weight & Bias Updates (Gradient Descent)
            for (int i = 0; i < 2; ++i) {
                w_output[i] += learning_rate * d_output * hidden_layer[i];
            }
            b_output += learning_rate * d_output;

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    w_hidden[j][i] += learning_rate * d_hidden[i] * inputs[sample][j];
                }
                b_hidden[i] += learning_rate * d_hidden[i];
            }
        }
    }

    // 4. Test the Trained Model
    std::cout << "Training Complete! Model Evaluation Results:\n";
    std::cout << "-------------------------------------------\n";
    for (size_t sample = 0; sample < inputs.size(); ++sample) {
        double hidden_layer[2];
        for (int i = 0; i < 2; ++i) {
            double sum = b_hidden[i];
            for (int j = 0; j < 2; ++j) {
                sum += inputs[sample][j] * w_hidden[j][i];
            }
            hidden_layer[i] = sigmoid(sum);
        }

        double output_sum = b_output;
        for (int i = 0; i < 2; ++i) {
            output_sum += hidden_layer[i] * w_output[i];
        }
        double final_output = sigmoid(output_sum);

        std::cout << "Input: {" << inputs[sample][0] << ", " << inputs[sample][1] << "}"
                  << " | Expected: " << targets[sample]
                  << " | Predicted: " << final_output
                  << " -> Classified as: " << (final_output > 0.5 ? 1 : 0) << "\n";
    }

    return 0;
}
