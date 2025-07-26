# 🧠 Pure C++ Neural Network Library - `NeuroLite`

This is a minimal yet fully functional deep learning framework written in C++ from scratch. It includes:

- Tensor class with shape handling, operations, matrix multiplication
- Dense layers with ReLU, Sigmoid, Tanh activations
- Mean Squared Error loss
- Backpropagation + weight updates
- Optimizer abstraction (SGD implemented)
- XOR training example

## 🔧 Build

```bash
g++ -std=c++17 neuralnetpure.cpp -o neuralnet
./neuralnet
