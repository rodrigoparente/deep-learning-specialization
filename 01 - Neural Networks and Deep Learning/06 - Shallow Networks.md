# Overview of Neural Networks

 - A neural network is formed by stacking multiple layers, each composed of small activations units (e.g., sigmoid or relu).
 - Each layer $ l $ (for $ l = 1, 2, \ldots, n $) performs the following computations:
    - **Step 1:** Compute $ z^{[l]} = w^{[l]} \cdot a^{[l-1]} + b^{[l]} $.
    - **Step 2:** Compute $ a^{[l]} = \sigma(z^{[l]}) $ using the sigmoid function.

**Layered Structure in Neural Networks**
 
 - The network is composed of $ n $ layers, where:
    - **Layer 1:** Receives the input features $ x $.
    - **Layer $ l $:** Receives the output from the previous layer $ a^{[l-1]} $ and computes new values $ z^{[l]} $ and $ a^{[l]} $.
    - **Final Layer $ n $:** Outputs the final prediction $ \hat{y} = a^{[n]} $.

**Notations Used**

 - **Superscript Square Brackets [ ]:**
    - Used to denote layers, e.g., $ z^{[l]} $, $ a^{[l]} $ for Layer $ l $.
 - **Superscript Round Brackets ( ):**
    - Used to refer to individual training examples, e.g., $ x^{(i)} $ refers to the $ i $-th training example.

# Basic Components of a Neural Network

Imagine the following neural network with:
 - **Input Layer:**
    - Contains input features $ x_1 $, $ x_2 $, $ x_3 $.
    - This layer represents the inputs to the neural network.
 - **Hidden Layer:**
    - Contains 4 nodes (or neurons) that process inputs from the input layer.
    - The term "hidden" refers to the fact that the values in this layer are not observed in the training set.
 - **Output Layer:**
    - Consists of a single node that generates the predicted value $ \hat{y} $.

**Notations in Neural Networks**

 - The input features are denoted by $ \mathbf{X} $ or $ A^{[0]} $, which represent the activations of the input layer.
 - $ A^{[1]} $ represents the activations of the hidden layer and is composed of the individual activation values $ a^{[1]}_1, a^{[1]}_2, \dots, a^{[1]}_4 $ of the layer nodes.
 - The output layer generates an activation value $ A^{[2]} $, which is a real number corresponding to the predicted output $ \hat{y} $.

**Layer Numbering and Conventions**

 - Although the network has an input layer, a hidden layer, and an output layer, it is conventionally called a "two-layer neural network."
 - The layers are counted starting from the hidden layer:
    - Hidden Layer: Layer 1
    - Output Layer: Layer 2
 - The input layer is often referred to as Layer 0, but it is not counted as an official layer.

# Computational Steps

 - Each node in the hidden layer performs two main computations:
    1. **Linear Combination**: Calculate $ z $ using the formula:
       $$  z^{[l]}_i = \mathbf{w}^{[l]}_i \cdot \mathbf{X} + b^{[l]}_i $$
       where $ \mathbf{w}^{[l]}_i $ is the weight vector for node $ i $ in the hidden layer $ l $, $ \mathbf{X} $ is the input feature vector, and $ b^{[l]}_i $ is the bias term for that node.
    2. **Activation Function**: Compute the activation $ a $ using for example, the sigmoid function:
       $$  a^{[l]}_i = \sigma(z^{[l]}_i) $$

**Vectorized Implementation**

 - Instead of computing $ z $ and $ a $ for each node individually, vectorization can be used to make the computations more efficient:
 - Form a weight matrix $ \mathbf{W}^{[l]} $ by stacking the individual weight vectors $ \mathbf{w}^{[l]}_1, \mathbf{w}^{[l]}_2, \dots, \mathbf{w}^{[l]}_n $ as rows.
    - Compute $ \mathbf{Z}^{[l]} $ as:
      $$ \mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \cdot \mathbf{X} + \mathbf{B}^{[l]} $$
      where $ \mathbf{B}^{[l]} $ is the vector of bias terms for the hidden layer.
    - The activations $ \mathbf{A}^{[l]} $ are calculated by applying the sigmoid function element-wise:
      $$ \mathbf{A}^{[l]} = \sigma(\mathbf{Z}^{[l]}) $$

# Activation Functions

When building a neural network, you must decide which activation function to use in the hidden layers and output units. Below there are 3 most common activation functions used while developing deep learning algorithms.

**Sigmoid Function**
  
 - Sigmoid function is expressed by:
   
   $$ a = \frac{1}{1 + e^{-z}} $$
 
 - Outputs values between 0 and 1.
 - Traditionally used in many neural networks for binary classification.

**Tanh (Hyperbolic Tangent) Function**

  - TANH function  is expressed by:
  
   $$ a = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $$
  
  - Outputs values between -1 and +1.
  - Typically superior to the sigmoid function for hidden layers because it makes learning for subsequent layers easier.

**Rectified Linear Unit (ReLU)**

 - ReLU function is expressed by:
  
   $$ a = \max(0, z) $$
  
 - Outputs $ z $ if $ z $ is positive, otherwise outputs 0.
 - For large portions of the input space, the derivative is non-zero, which accelerates learning.

Predicting which activation function will work best for a specific problem can be challenging. The best approach is to remain flexible and adapt based on the specific requirements and characteristics of the problem at hand.

# Importance of Random Initialization

 - If weights are initialized to zero, all hidden units will compute the exact same function.
 - This causes the gradients for these units to be identical during backpropagation.
 - As a result, after every iteration of training, the hidden units remain symmetric, computing the same function, making additional hidden units redundant.
 - Because of that, weights in a neural network must be initialized randomly.

**Impact of Large Weights**

 - If weights are initialized to large values, it can cause issues with the activation functions:
   - For tanh or sigmoid activation functions:
      - Large weights result in large values of $ Z $ (i.e., $ Z = W_1X + B_1 $).
      - This can lead to saturation in the activation functions (where the gradient is very small).
    - Saturation causes gradient descent to be slow, hampering the learning process.
 - For shallow networks, its recommended that the weights are initialized to small random values (e.g., multiply by 0.01) to avoid large $ Z $ values.
 - For deeper networks, it might be necessary to choose a different constant for initialization.