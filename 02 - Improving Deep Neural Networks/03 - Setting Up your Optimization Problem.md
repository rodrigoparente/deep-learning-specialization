# Normalizing Inputs

 - Unnormalized inputs can create an elongated cost function, making gradient descent harder to converge.
 - This causes gradient descent to oscillate and take many small steps, slowing down the training.
 - Normalized inputs result in a more symmetric cost function, allowing gradient descent to converge more easily.
 - This enables larger steps during gradient descent, accelerating the optimization process.
 - In high-dimensional parameter spaces, normalization ensures the cost function is well-conditioned, improving optimization efficiency.

**Steps for Normalizing Inputs**

1. **Zero-Centering the Data**:
   - **Objective**: Shift the data so that it has zero mean.
   - **Formula**:
     - Compute the mean vector: $ \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} $
     - Subtract the mean from each training example: $ x^{(i)} := x^{(i)} - \mu $
   - **Result**: The training set is moved such that it centers around the origin (zero mean).

2. **Normalizing Variance**:
   - **Objective**: Scale the data so that each feature has a unit variance.
   - **Formula**:
     - Compute the variance vector: $ \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} \left(x^{(i)} - \mu\right)^2 $
     - Normalize each feature by its standard deviation: $ x^{(i)} := \frac{x^{(i)}}{\sigma} $
   - **Result**: Each feature in the training set will have a variance of 1, ensuring they are on a similar scale.

# Vanishing/Exploding Gradient

 - The **vanishing gradient problem** occurs when gradients become too small during backpropagation, leading to minimal learning in the lower layers.
 - This makes it difficult for the model to capture long-term dependencies or complex features.
 - The **exploding gradient problem** is the opposite, where gradients become excessively large.
 - This can cause the weights to change drastically, leading to instability in the training process and preventing the model from converging properly.

**Solution**

 - Proper weight initialization helps mitigate vanishing and exploding gradients, although it doesn’t entirely solve the problem.
 - Different initialization strategies exist for various activation functions.
 - For ReLU, use a variance of $  \frac{2}{n_{l-1}} $:
  $$ W \sim \text{np.random.randn}(\text{shape}) \times \sqrt{\frac{2}{n_{l-1}}} $$
 - For Tanh, use a variance of $  \frac{1}{n_{l-1}} $:
  $$ W \sim \text{np.random.randn}(\text{shape}) \times \sqrt{\frac{1}{n_{l-1}}} $$

# Gradient Checking

 - Gradient Checking is essential for validating backpropagation.
 - It ensures that the computed gradients align with numerical approximations, confirming the accuracy of the neural network's learning process.
 - Steps to Perform Gradient Checking:
   1. Reshape and Concatenate Parameters:
      - Reshape weights ($W$) and biases ($B$) into a single vector, $\theta$.
      - The cost function $J$ becomes a function of $\theta$ (i.e., $J(\theta)$).
  
   2. Reshape and Concatenate Gradients:
      - Reshape the gradients of the cost function, $dW$ and $dB$, into a vector $d\theta$.
      - Ensure $d\theta$ has the same dimension as $\theta$.
  
   3. Compute Approximate Gradients:
      - For each component $i$ of $\theta$, calculate the approximate gradient:
      $$
      d\theta_{\text{approx}}[i] = \frac{J(\theta_1, \theta_2, \dots, \theta_i + \epsilon, \dots) - J(\theta_1, \theta_2, \dots, \theta_i - \epsilon, \dots)}{2\epsilon}
      $$
  
   4. Compare Gradients:
      - Calculate the distance between the approximate gradient $d\theta_{\text{approx}}$ and the analytical gradient $d\theta$ using the $L_2$ norm:
      $$
      \text{Difference} = \frac{\|\ d\theta_{\text{approx}} - d\theta\|_2}{\|\ d\theta_{\text{approx}} + d\theta\|_2}
      $$
      - This normalization accounts for differences in gradient magnitudes.

- Interpreting the Difference:
  - $ < 10^{-7}$: Implementation is likely correct.
  - $\approx 10^{-5}$: Double-check; could be acceptable.
  - $ > 10^{-3}$: Likely indicates a bug in gradient computation.
