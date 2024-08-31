# Vectorization

 - The process of eliminating explicit for-loops in your code to improve computational efficiency.
 - Especially important in deep learning due to the large datasets typically used, which can otherwise result in long training times.

**Example: Logistic Regression**

 - In logistic regression, you often need to compute $ z = W^T X + b $, where:
    - $ W $ and $ X $ are vectors of size $ n $.
    - $ W^T $ is the transpose of $ W $, resulting in a dot product with $ X $.
 - **Non-Vectorized Code**: uses a for-loop to compute the dot product:
    ```python
    z = 0
    for i in range(n):
        z += W[i] * X[i]
    z += b
    ```
 - **Vectorized Code**: directly computes the dot product and adds $ b $:
    ```python
    z = np.dot(W, X) + b
    ```
 - The vectorized approach is significantly faster due to the elimination of the for-loop.

# Logistic Regression

**Forward Propagation**

 - For $ M $ training examples, making predictions involves computing $ Z $ values and activations ($ \hat{y} $) for each example.
 - Typically, you would compute $ Z $ for each training example sequentially, but vectorization allows processing all examples simultaneously.
 - The calculation can be expressed as:
    $$ Z = W^T X + b $$
  - Here:
    - $ W^T $ is the transpose of the weight vector $ W $, resulting in a row vector.
    - $ X $ is an $ n_X \times m $ matrix, where $ n_X $ is the number of features.
    - $ b $ is a bias term added to each element, which can be treated as a $ 1 \times m $ row vector due to broadcasting in Python.
  - This vectorized expression efficiently computes all $ Z $ values without loops.
 - For a single training example, the activation $ a_i $ is calculated as $ a_i = \sigma(z_i) $ 
 - To compute all activations $ A $, implement a vectorized sigmoid function to operate on the entire matrix $ Z $, as follow:
    $$ A = \sigma(Z) $$
 
**Backward Propagation**

 - For each training example, the gradient $ dZ_i $ is calculated as $ dZ_i = a_i - y_i $.
 - To handle all training examples at once:
    $$ dZ = A - Y $$
  - Here, $ A $ is the vector of predictions, and $ Y $ is the vector of true labels.
 - The traditional approach to implementing $dW$ and $dB$ involved initializing $dW$ as a vector of zeros and $dB$ as zero.
 - A loop was then used to update their values for each training example $ i $.
 - For the vectorized version, $ dW $ is calculated by multiplying the input matrix $ X $ with the transpose of $ dZ $, and then dividing by $ M $:
      $$ dW = \frac{1}{M} X \cdot dZ^T $$
 - Similarly, $ dB $ is calculated by summing all $ dZ $ values and dividing by $ M $:
      $$ dB = \frac{1}{M} \sum_{i=1}^{M} dZ_i $$

**Gradient Descent**

 - Finally, the weights $ W $ are updated by subtracting the product of the learning rate $ \alpha $ and the gradient of the cost function with respect to $ W $ (denoted as $ dW $):
   $$  W := W - \alpha \cdot dW $$
 - Similarly, the bias $ B $ is updated by subtracting the product of the learning rate $ \alpha $ and the gradient of the cost function with respect to $ B $ (denoted as $ dB $):
   $$  B := B - \alpha \cdot dB $$