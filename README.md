# Neural Network Implementations for Logic Gates
This repository implements neural networks for solving **AND** and **XOR** logic gates using **backpropagation**. 

## How It Works
These neural networks are designed to learn logical operations through supervised training. Each network consists of two hidden neurons and one output neuron, using the **sigmoid activation function** and **gradient descent** with backpropagation to adjust weights and minimize error.

## Key Features
* **Multi-Layer Architecture**: Both implementations use a neural network with:
  * 2 input neurons
  * 2 hidden neurons
  * 1 output neuron
* **Backpropagation Algorithm**: Implements the complete backpropagation process including:
  * Forward pass calculation
  * Error computation
  * Gradient calculation
  * Weight updates
* **Xavier/Glorot Initialization**: Uses optimal weight initialization to improve training convergence
* **Early Stopping**: Implements early stopping when accuracy reaches a specified threshold
* **Performance Metrics**: Tracks and reports:
  * Mean Squared Error (MSE)
  * Accuracy per epoch
  * Final evaluation results

## File Overview
* **and_logic_nn.py**:
  * Implementation of a neural network for the AND logic gate
  * Training data: `[(0,0,0), (0,1,0), (1,0,0), (1,1,1)]`
  * Simpler training task due to linear separability
  
* **xor_logic_nn.py**:
  * Implementation of a neural network for the XOR logic gate
  * Training data: `[(0,0,0), (0,1,1), (1,0,1), (1,1,0)]`
  * More challenging due to non-linear separability

## Neural Network Components
* **Sigmoid Activation**: Uses the sigmoid function `1/(1 + e^(-x))` for non-linear transformation
* **Weight Updates**: Implements full gradient descent across all layers
* **Loss Function**: Uses Mean Squared Error (MSE) for error calculation
* **Learning Rate**: Configurable learning rate (default: 0.1)

## Example Output
```python
Epoch 1000: MSE = 0.0124, Accuracy = 75.00%
Epoch 2000: MSE = 0.0089, Accuracy = 100.00%

Final Evaluation:
Input: [0, 0], Target: 0, Output: 0.031
Input: [0, 1], Target: 1, Output: 0.962
Input: [1, 0], Target: 1, Output: 0.959
Input: [1, 1], Target: 0, Output: 0.042
```

## Future Improvements
* Add additonal code for other logic gates
* Add visualization of the **decision boundaries**
* Implement different activation functions (**ReLU**, **tanh**)
* Add support for **batch training**
* Include **momentum** and **adaptive learning rates**
* Implement **cross-validation** for better generalization

The repository serves as an educational resource for understanding basic neural network operations and can be used as a foundation for more complex deep learning implementations.
