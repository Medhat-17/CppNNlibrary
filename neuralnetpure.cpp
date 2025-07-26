#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // std::accumulate

class Tensor
{
public:
    std::vector<double> data;
    std::vector<size_t> shape;
    size_t size;

    // Constructor for a tensor with a given shape
    Tensor(const std::vector<size_t> &s) : shape(s)
    {
        size = 1;
        for (size_t dim : shape)
        {
            size *= dim;
        }
        data.resize(size, 0.0);
    }

    // Constructor for a scalar tensor
    Tensor(double value = 0.0) : shape({1}), size(1)
    {
        data.push_back(value);
    }

    // Copy constructor
    Tensor(const Tensor &other) : data(other.data), shape(other.shape), size(other.size) {}

    // Assignment operator
    Tensor &operator=(const Tensor &other)
    {
        if (this != &other)
        {
            data = other.data;
            shape = other.shape;
            size = other.size;
        }
        return *this;
    }

    // Element access (simple 1D indexing for now, assumes flattened data)
    double &operator[](size_t index)
    {
        if (index >= size)
        {
            // Basic error handling
            std::cerr << "Error: Tensor index out of bounds." << std::endl;
            exit(1);
        }
        return data[index];
    }

    const double &operator[](size_t index) const
    {
        if (index >= size)
        {
            std::cerr << "Error: Tensor index out of bounds (const)." << std::endl;
            exit(1);
        }
        return data[index];
    }

    // Initialize with random values (for weights)
    void randomize(double min_val, double max_val)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = min_val + (static_cast<double>(rand()) / RAND_MAX) * (max_val - min_val);
        }
    }

    // Fill with a specific value
    void fill(double value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    // Basic matrix multiplication (for 2D tensors only)
    static Tensor matmul(const Tensor &a, const Tensor &b)
    {
        if (a.shape.size() != 2 || b.shape.size() != 2)
        {
            std::cerr << "Error: Matmul only supports 2D tensors." << std::endl;
            exit(1);
        }
        if (a.shape[1] != b.shape[0])
        {
            std::cerr << "Error: Matmul dimensions mismatch: A cols (" << a.shape[1] << ") != B rows (" << b.shape[0] << ")" << std::endl;
            exit(1);
        }

        Tensor result({a.shape[0], b.shape[1]});
        for (size_t i = 0; i < a.shape[0]; ++i)
        {
            for (size_t j = 0; j < b.shape[1]; ++j)
            {
                double sum = 0.0;
                for (size_t k = 0; k < a.shape[1]; ++k)
                {
                    sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
                }
                result.data[i * result.shape[1] + j] = sum;
            }
        }
        return result;
    }

    // Element-wise addition
    Tensor operator+(const Tensor &other) const
    {
        if (shape != other.shape)
        {
            std::cerr << "Error: Tensor addition shape mismatch." << std::endl;
            exit(1);
        }
        Tensor result(shape);
        for (size_t i = 0; i < size; ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Element-wise subtraction
    Tensor operator-(const Tensor &other) const
    {
        if (shape != other.shape)
        {
            std::cerr << "Error: Tensor subtraction shape mismatch." << std::endl;
            exit(1);
        }
        Tensor result(shape);
        for (size_t i = 0; i < size; ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    Tensor operator*(const Tensor &other) const
    {
        if (shape != other.shape)
        {
            std::cerr << "Error: Tensor element-wise multiplication shape mismatch." << std::endl;
            exit(1);
        }
        Tensor result(shape);
        for (size_t i = 0; i < size; ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    // Scalar multiplication
    Tensor operator*(double scalar) const
    {
        Tensor result(shape);
        for (size_t i = 0; i < size; ++i)
        {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Scalar division
    Tensor operator/(double scalar) const
    {
        if (scalar == 0.0)
        {
            std::cerr << "Error: Division by zero." << std::endl;
            exit(1);
        }
        Tensor result(shape);
        for (size_t i = 0; i < size; ++i)
        {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    // Transpose (for 2D tensors only)
    Tensor transpose() const
    {
        if (shape.size() != 2)
        {
            std::cerr << "Error: Transpose only supports 2D tensors." << std::endl;
            exit(1);
        }
        Tensor result({shape[1], shape[0]});
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < shape[1]; ++j)
            {
                result.data[j * result.shape[1] + i] = data[i * shape[1] + j];
            }
        }
        return result;
    }

    // Sum all elements
    double sum() const
    {
        return std::accumulate(data.begin(), data.end(), 0.0);
    }

    // Sum along a specific axis (for 2D tensors, axis 0 = sum columns, axis 1 = sum rows)
    // Returns a 1D tensor representing the sum
    Tensor sum_axis(size_t axis) const
    {
        if (shape.size() != 2)
        {
            std::cerr << "Error: sum_axis only supports 2D tensors." << std::endl;
            exit(1);
        }

        if (axis == 0)
        { // Sum columns (result has shape {1, cols})
            Tensor result({1, shape[1]});
            for (size_t j = 0; j < shape[1]; ++j)
            {
                double col_sum = 0.0;
                for (size_t i = 0; i < shape[0]; ++i)
                {
                    col_sum += data[i * shape[1] + j];
                }
                result.data[j] = col_sum;
            }
            return result;
        }
        else if (axis == 1)
        { // Sum rows (result has shape {rows, 1})
            Tensor result({shape[0], 1});
            for (size_t i = 0; i < shape[0]; ++i)
            {
                double row_sum = 0.0;
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    row_sum += data[i * shape[1] + j];
                }
                result.data[i] = row_sum;
            }
            return result;
        }
        else
        {
            std::cerr << "Error: Invalid axis for sum_axis." << std::endl;
            exit(1);
        }
    }

    // Reshape the tensor (must maintain the same total number of elements)
    Tensor reshape(const std::vector<size_t> &new_shape) const
    {
        size_t new_size = 1;
        for (size_t dim : new_shape)
        {
            new_size *= dim;
        }
        if (new_size != size)
        {
            std::cerr << "Error: Reshape failed. New shape size (" << new_size << ") does not match original size (" << size << ")." << std::endl;
            exit(1);
        }
        Tensor reshaped_tensor(new_shape);
        std::copy(data.begin(), data.end(), reshaped_tensor.data.begin());
        return reshaped_tensor;
    }

    // Print tensor for debugging
    void print() const
    {
        std::cout << "Tensor (Shape: ";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << ", Size: " << size << "):\n";
        // For 2D tensors, print as matrix
        if (shape.size() == 2)
        {
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    std::cout << data[i * shape[1] + j] << "\t";
                }
                std::cout << "\n";
            }
        }
        else
        { // For other shapes, just print flattened data
            for (size_t i = 0; i < size; ++i)
            {
                std::cout << data[i] << " ";
            }
            std::cout << "\n";
        }
    }
};

// --- 2. Activation Functions ---
class Activation
{
public:
    // ReLU
    static Tensor relu(const Tensor &x)
    {
        Tensor result(x.shape);
        for (size_t i = 0; i < x.size; ++i)
        {
            result.data[i] = std::max(0.0, x.data[i]);
        }
        return result;
    }

    static Tensor relu_derivative(const Tensor &x)
    {
        Tensor result(x.shape);
        for (size_t i = 0; i < x.size; ++i)
        {
            result.data[i] = (x.data[i] > 0) ? 1.0 : 0.0;
        }
        return result;
    }

    // Sigmoid
    static Tensor sigmoid(const Tensor &x)
    {
        Tensor result(x.shape);
        for (size_t i = 0; i < x.size; ++i)
        {
            result.data[i] = 1.0 / (1.0 + std::exp(-x.data[i]));
        }
        return result;
    }

    static Tensor sigmoid_derivative(const Tensor &x)
    {
        Tensor sig_x = sigmoid(x);
        Tensor result(x.shape);
        for (size_t i = 0; i < x.size; ++i)
        {
            result.data[i] = sig_x.data[i] * (1.0 - sig_x.data[i]);
        }
        return result;
    }

    // Tanh
    static Tensor tanh(const Tensor &x)
    {
        Tensor result(x.shape);
        for (size_t i = 0; i < x.size; ++i)
        {
            result.data[i] = std::tanh(x.data[i]);
        }
        return result;
    }

    static Tensor tanh_derivative(const Tensor &x)
    {
        Tensor tanh_x = tanh(x);
        Tensor result(x.shape);
        for (size_t i = 0; i < x.size; ++i)
        {
            result.data[i] = 1.0 - (tanh_x.data[i] * tanh_x.data[i]);
        }
        return result;
    }
};

// --- 3. Loss Functions ---
class Loss
{
public:
    // Mean Squared Error (MSE)
    static double mse(const Tensor &predictions, const Tensor &targets)
    {
        if (predictions.shape != targets.shape)
        {
            std::cerr << "Error: MSE input shape mismatch." << std::endl;
            exit(1);
        }
        double sum_sq_error = 0.0;
        for (size_t i = 0; i < predictions.size; ++i)
        {
            sum_sq_error += std::pow(predictions.data[i] - targets.data[i], 2);
        }
        return sum_sq_error / predictions.size;
    }

    // Derivative of MSE
    static Tensor mse_derivative(const Tensor &predictions, const Tensor &targets)
    {
        if (predictions.shape != targets.shape)
        {
            std::cerr << "Error: MSE derivative input shape mismatch." << std::endl;
            exit(1);
        }
        Tensor gradient(predictions.shape);
        for (size_t i = 0; i < predictions.size; ++i)
        {
            gradient.data[i] = 2.0 * (predictions.data[i] - targets.data[i]) / predictions.size;
        }
        return gradient;
    }
};

// --- 4. Layer Base Class ---
class Layer
{
public:
    virtual ~Layer() = default;

    // Forward pass: Computes output for a given input
    virtual Tensor forward(const Tensor &input) = 0;

    virtual Tensor backward(const Tensor &output_gradient) = 0;

    // Update weights (called by optimizer)
    virtual void update_weights(double learning_rate) = 0;
};

class DenseLayer : public Layer
{
public:
    Tensor weights;
    Tensor biases;
    Tensor last_input;
    Tensor last_output;

    Tensor d_weights;
    Tensor d_biases;

    // Activation function type
    enum ActivationType
    {
        RELU,
        SIGMOID,
        TANH,
        NONE
    };
    ActivationType activation_type;

    DenseLayer(size_t input_dim, size_t output_dim, ActivationType act_type = RELU)
        : weights({input_dim, output_dim}),
          biases({1, output_dim}), // Biases are typically row vectors, one for each output neuron
          d_weights({input_dim, output_dim}),
          d_biases({1, output_dim}),
          activation_type(act_type)
    {
        // Initialize weights with small random values (e.g., He initialization for ReLU)
        // For simplicity, using uniform random here.
        weights.randomize(-0.1, 0.1);
        biases.fill(0.0); // Initialize biases to zero
    }

    Tensor forward(const Tensor &input) override
    {
        // Store input for backward pass
        last_input = input;

        // Linear transformation: Z = Input * Weights + Biases
        // Input shape: (batch_size, input_dim)
        // Weights shape: (input_dim, output_dim)
        // Biases shape: (1, output_dim)
        // Z shape: (batch_size, output_dim)

        // Perform matrix multiplication
        Tensor z = Tensor::matmul(input, weights);

        // Add biases (broadcasting needed - add bias row vector to each row of z)
        // Assuming z.shape[0] is batch_size
        Tensor z_plus_bias(z.shape);
        for (size_t i = 0; i < z.shape[0]; ++i)
        { // Iterate over batch size
            for (size_t j = 0; j < z.shape[1]; ++j)
            { // Iterate over output_dim
                z_plus_bias.data[i * z.shape[1] + j] = z.data[i * z.shape[1] + j] + biases.data[j];
            }
        }
        last_output = z_plus_bias; // Store output before activation

        // Apply activation function
        if (activation_type == RELU)
        {
            return Activation::relu(last_output);
        }
        else if (activation_type == SIGMOID)
        {
            return Activation::sigmoid(last_output);
        }
        else if (activation_type == TANH)
        {
            return Activation::tanh(last_output);
        }
        else
        { // NONE
            return last_output;
        }
    }

    Tensor backward(const Tensor &output_gradient) override
    {
        // output_gradient is dL/dY (gradient from next layer or loss function)
        // Y = activation(Z)
        // Z = Input * Weights + Biases

        // 1. Calculate dL/dZ (gradient with respect to Z, before activation)
        Tensor d_activation;
        if (activation_type == RELU)
        {
            d_activation = Activation::relu_derivative(last_output);
        }
        else if (activation_type == SIGMOID)
        {
            d_activation = Activation::sigmoid_derivative(last_output);
        }
        else if (activation_type == TANH)
        {
            d_activation = Activation::tanh_derivative(last_output);
        }
        else
        { // NONE
            d_activation = Tensor(last_output.shape);
            d_activation.fill(1.0); // Derivative of identity is 1
        }

        // dL/dZ = dL/dY * dY/dZ (element-wise multiplication)
        Tensor dz = output_gradient * d_activation;

        // 2. Calculate dL/dWeights (gradient with respect to weights)
        // dL/dW = Input^T * dL/dZ
        // last_input shape: (batch_size, input_dim)
        // dz shape: (batch_size, output_dim)
        // d_weights shape: (input_dim, output_dim)
        d_weights = Tensor::matmul(last_input.transpose(), dz);

        // 3. Calculate dL/dBiases (gradient with respect to biases)
        // dL/dB = sum(dL/dZ, axis=0) (sum along batch dimension)
        // dz shape: (batch_size, output_dim)
        // d_biases shape: (1, output_dim)
        d_biases = dz.sum_axis(0);

        // 4. Calculate dL/dInput (gradient to pass to previous layer)
        // dL/dInput = dL/dZ * Weights^T
        // dz shape: (batch_size, output_dim)
        // weights shape: (input_dim, output_dim)
        // d_input shape: (batch_size, input_dim)
        Tensor d_input = Tensor::matmul(dz, weights.transpose());

        return d_input;
    }

    void update_weights(double learning_rate) override
    {
        // Update weights: W = W - learning_rate * dW
        weights = weights - (d_weights * learning_rate);
        // Update biases: B = B - learning_rate * dB
        biases = biases - (d_biases * learning_rate);
    }
};

// --- 6. Optimizer Base Class ---
class Optimizer
{
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<Layer *> &layers, double learning_rate) = 0;
};

// --- 7. SGD Optimizer ---
class SGD : public Optimizer
{
public:
    void update(std::vector<Layer *> &layers, double learning_rate) override
    {
        for (Layer *layer : layers)
        {
            layer->update_weights(learning_rate);
        }
    }
};

// --- 8. Neural Network Class ---
class NeuralNetwork
{
public:
    std::vector<Layer *> layers;
    Optimizer *optimizer; // Pointer to the optimizer
    double learning_rate;

    NeuralNetwork(double lr = 0.01) : optimizer(nullptr), learning_rate(lr)
    {
        // Seed random number generator
        srand(static_cast<unsigned int>(time(0)));
    }

    ~NeuralNetwork()
    {
        for (Layer *layer : layers)
        {
            delete layer;
        }
        layers.clear();
        if (optimizer)
        {
            delete optimizer;
        }
    }

    void add_layer(Layer *layer)
    {
        layers.push_back(layer);
    }

    void set_optimizer(Optimizer *opt)
    {
        if (optimizer)
        {
            delete optimizer; // Delete previous optimizer if any
        }
        optimizer = opt;
    }

    Tensor predict(const Tensor &input)
    {
        Tensor current_output = input;
        for (Layer *layer : layers)
        {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }

    void train(const Tensor &X_train, const Tensor &Y_train, int epochs, size_t batch_size)
    {
        if (!optimizer)
        {
            std::cerr << "Error: Optimizer not set. Call set_optimizer() before training." << std::endl;
            return;
        }
        if (X_train.shape[0] != Y_train.shape[0])
        {
            std::cerr << "Error: X_train and Y_train must have the same number of samples." << std::endl;
            return;
        }

        size_t num_samples = X_train.shape[0];
        size_t num_batches = (num_samples + batch_size - 1) / batch_size;

        std::cout << "Starting training for " << epochs << " epochs...\n";

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epoch_loss = 0.0;
            // Shuffle data (simple shuffling of indices)
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());

            for (size_t b = 0; b < num_batches; ++b)
            {
                size_t batch_start = b * batch_size;
                size_t batch_end = std::min(batch_start + batch_size, num_samples);
                size_t current_batch_size = batch_end - batch_start;

                // Create batch tensors
                Tensor batch_X({current_batch_size, X_train.shape[1]});
                Tensor batch_Y({current_batch_size, Y_train.shape[1]});

                for (size_t i = 0; i < current_batch_size; ++i)
                {
                    size_t original_index = indices[batch_start + i];
                    for (size_t j = 0; j < X_train.shape[1]; ++j)
                    {
                        batch_X.data[i * X_train.shape[1] + j] = X_train.data[original_index * X_train.shape[1] + j];
                    }
                    for (size_t j = 0; j < Y_train.shape[1]; ++j)
                    {
                        batch_Y.data[i * Y_train.shape[1] + j] = Y_train.data[original_index * Y_train.shape[1] + j];
                    }
                }

                // Forward pass
                Tensor predictions = predict(batch_X);

                // Calculate loss
                double current_loss = Loss::mse(predictions, batch_Y);
                epoch_loss += current_loss;

                // Backward pass (calculate gradients)
                Tensor loss_gradient = Loss::mse_derivative(predictions, batch_Y);
                Tensor current_gradient = loss_gradient;
                // Iterate layers in reverse order for backpropagation
                for (int i = layers.size() - 1; i >= 0; --i)
                {
                    current_gradient = layers[i]->backward(current_gradient);
                }

                // Update weights using optimizer
                optimizer->update(layers, learning_rate);
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss / num_batches << "\n";
        }
        std::cout << "Training finished.\n";
    }
};

// --- Example Usage (XOR Problem) ---
int main()
{
    std::cout << "Building a simple Neural Network for XOR problem...\n";

    // XOR input data (4 samples, 2 features)
    Tensor X_train({4, 2});
    X_train.data = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0};

    // XOR target data (4 samples, 1 output)
    Tensor Y_train({4, 1});
    Y_train.data = {
        0.0,
        1.0,
        1.0,
        0.0};

    // Create Neural Network
    NeuralNetwork nn(0.1); // Learning rate 0.1

    // Add layers
    // Input: 2 neurons
    // Hidden Layer 1: 4 neurons, ReLU activation
    nn.add_layer(new DenseLayer(2, 4, DenseLayer::RELU));
    // Output Layer: 1 neuron, Sigmoid activation (for binary classification-like output)
    nn.add_layer(new DenseLayer(4, 1, DenseLayer::SIGMOID));

    // Set optimizer
    nn.set_optimizer(new SGD());

    // Train the network
    int epochs = 10000;
    size_t batch_size = 4; // Full batch gradient descent for this small dataset
    nn.train(X_train, Y_train, epochs, batch_size);

    std::cout << "\nTesting the trained network:\n";
    // Test predictions
    Tensor test_input({1, 2});

    test_input.data = {0.0, 0.0};
    Tensor pred_00 = nn.predict(test_input);
    std::cout << "Input: [0, 0], Predicted Output: " << pred_00.data[0] << "\n";

    test_input.data = {0.0, 1.0};
    Tensor pred_01 = nn.predict(test_input);
    std::cout << "Input: [0, 1], Predicted Output: " << pred_01.data[0] << "\n";

    test_input.data = {1.0, 0.0};
    Tensor pred_10 = nn.predict(test_input);
    std::cout << "Input: [1, 0], Predicted Output: " << pred_10.data[0] << "\n";

    test_input.data = {1.0, 1.0};
    Tensor pred_11 = nn.predict(test_input);
    std::cout << "Input: [1, 1], Predicted Output: " << pred_11.data[0] << "\n";

    return 0;
}
