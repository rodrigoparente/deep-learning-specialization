# Mini-batch Gradient Descent

 - Training on large datasets can be slow, making efficient optimization algorithms essential for speeding up the process.
 - Steps of Mini-Batch Gradient Descent:

    - **Splitting the Dataset**: The training set is divided into smaller subsets called *mini-batches*:
        - $ X^{\{t\}} $ represents the input data for the $ t $-th mini-batch, with dimensions $ n_x \times k $.
        - $ Y^{\{t\}} $ represents the labels for the $ t $-th mini-batch, with dimensions $ 1 \times k $.
    - **Processing Each Mini-Batch**: Perform forward propagation, back-propagation, and weight updates for each mini-batch.
    
**Size of Mini-Batches**

 - **Batch Gradient Descent**:
     - A mini-batch size equal to the full training set ($ m $) results in batch gradient descent.
    - Suitable for small training sets.
 - **Stochastic Gradient Descent**:
    - A mini-batch size of 1 leads to stochastic gradient descent.
    - Processes one training example at a time, which introduces high noise and prevents the algorithm from fully converging.
 - **Choosing the Mini-Batch Size**:
    - The mini-batch size should be between 1 and $ m $ to balance noise and computational efficiency.
    - Common mini-batch sizes range from 64 to 512, often selected as powers of 2 (e.g., 64, 128, 256, 512) for computational efficiency.
    - Ensure the mini-batch size fits within the CPU/GPU memory to avoid performance degradation.

**Advantages of Mini-Batch Gradient Descent**

 - Faster than batch gradient descent, particularly on large datasets.
 - Enables more frequent updates, leading to quicker convergence.
 - Scales effectively with large datasets, making it the preferred method in deep learning.


# Exponentially Weighted Averages

 - Exponentially Weighted Averages (EWA) is a technique used to compute a smoothed version of a time series or sequence of values.
 - It gives more weight to recent observations while exponentially decreasing the influence of older observations.
 - This method is particularly useful in contexts where recent data is more relevant for predictions or analysis, such as in gradient descent optimization algorithms.

**Mathematical Definition**

Given a sequence of observations $ \theta_1, \theta_2, \dots, \theta_t $, the exponentially weighted average at time $ t $, denoted as $ v_t $, is computed as:

$$
v_t = \beta v_{t-1} + (1 - \beta) \theta_t
$$

where:
- $ v_t $ is the exponentially weighted average at time $ t $.
- $ \beta $ is the smoothing factor (or decay rate), typically $ 0 < \beta < 1 $.
- $ \theta_t $ is the current observation at time $ t $.

The smoothing factor $ \beta $ determines how quickly the influence of older observations decays:
- A larger $ \beta $ (e.g., 0.9) results in slower decay, meaning older observations still have a significant influence.
- A smaller $ \beta $ (e.g., 0.1) results in faster decay, meaning the average is more responsive to recent observations.

**Bias Correction**

 - When $ v_0 $ is initialized to 0, the initial values of $ v_t $ tend to be biased toward zero, especially for small $ t $.
 - This is because the algorithm hasn't had enough time to accumulate the influence of multiple observations.
 - To correct this bias, especially during the initial stages, a bias-corrected estimate $ \hat{v}_t $ is used.
 - The bias-corrected value is given by:

    $$
    \hat{v}_t = \frac{v_t}{1 - \beta^t}
    $$

    where:
    - $ \hat{v}_t $ is the bias-corrected estimate at time $ t $.
    - $ v_t $ is the exponentially weighted average at time $ t $.
    - $ \beta^t $ accounts for the bias that occurs when $ t $ is small.

# Gradient Descent with Momentum

 - For cost functions with elongated elliptical contours, standard gradient descent often oscillates between the sides of the ellipse.
 - These oscillations occur because the algorithm takes small steps in the vertical direction to avoid overshooting, which slows down convergence.
 - Gradient Descent with Momentum addresses this issue by using an exponentially weighted average of past gradients to update the weights instead of relying solely on the current gradient.
 - This averaging smooths out vertical oscillations, leading to more stable updates in that direction.
 - Since horizontal gradients are generally consistent, their average remains significant, allowing for quicker movement toward the minimum.
 - As a result, this method speeds up convergence by minimizing oscillations and enabling a more direct path to the optimal solution.

**Momentum-Based Gradient Descent Implementation**

 - Compute the usual derivatives: $ dW $ and $ db $ (gradients for weights and biases).
 - Calculate exponentially weighted averages of the gradients:
    $$ v_{dW} = \beta \cdot v_{dW} + (1 - \beta) \cdot dW $$
    $$ v_{db} = \beta \cdot v_{db} + (1 - \beta) \cdot db $$
 - Update the parameters using these averages:
    $$ W = W - \alpha \cdot v_{dW} $$
    $$ b = b - \alpha \cdot v_{db} $$
    where:
     - $ \alpha $ is the learning rate.
     - $ \beta $ is a hyperparameter controlling the momentum.

# RMSprop

 - RMSprop, which stands for Root Mean Square Propagation, is designed to address issues with gradient descent, particularly oscillations and slow convergence.
 - To manage this, RMSprop computes the squared gradients for each parameter and maintains an exponentially weighted average of these squared gradients.

**RMSprop Implementation**

 - Calculate the derivative $ dW $ and $ db $ for the current mini-batch.
 - Update $ S_{dW} $ and $ S_{db} $ as follows:
    $$ S_{dW} = \beta S_{dW} + (1 - \beta) \cdot dW^2 $$
    $$ S_{db} = \beta S_{db} + (1 - \beta) \cdot db^2 $$
    where:
     - $ \beta $ is a smoothing constant. 
     - The squaring is element-wise operation.
- Then, update the weights $ W $ and biases $ b $ using:
    $$ W \gets W - \frac{\alpha \cdot dW}{\sqrt{S_{dW}} + \epsilon} $$
    $$ b \gets b - \frac{\alpha \cdot db}{\sqrt{S_{db}} + \epsilon} $$
    where:
     - $ \alpha $ is the learning rate.
     - $ \epsilon $ is a small constant added for numerical stability.

# Adam

 - Adam is an optimization algorithm that stands for Adaptive Moment Estimation.
 - It integrates the benefits of momentum and RMSprop to enhance gradient descent.
 - Adam has proven effective across a wide range of architectures, combining momentum and RMSprop approaches.

**Adam Implementation**

 - Initialize $ V_{dw} = 0 $, $ S_{dw} = 0 $, $ V_{db} = 0 $, and $ S_{db} = 0 $.
 - Compute the gradients $ dW $ and $ db $ from the current mini-batch.
 - Update the momentum values:
    $$ V_{dw} = \beta_1 \cdot V_{dw} + (1 - \beta_1) \cdot dW $$
    $$ V_{db} = \beta_1 \cdot V_{db} + (1 - \beta_1) \cdot db $$
 - Updated RMSprop values:
    $$ S_{dw} = \beta_2 \cdot S_{dw} + (1 - \beta_2) \cdot dW^2 $$
    $$ S_{db} = \beta_2 \cdot S_{db} + (1 - \beta_2) \cdot db^2 $$
 - Correct the bias values:
    $$ V_{dw}^{\text{corrected}} = \frac{V_{dw}}{1 - \beta_1^t} $$
    $$ V_{db}^{\text{corrected}} = \frac{V_{db}}{1 - \beta_1^t} $$
    $$ S_{dw}^{\text{corrected}} = \frac{S_{dw}}{1 - \beta_2^t} $$
    $$ S_{db}^{\text{corrected}} = \frac{S_{db}}{1 - \beta_2^t} $$
 - Updated $ W $ and $ b $ parameters:
    $$ W \gets W - \frac{\alpha \cdot V_{dw}^{\text{corrected}}}{\sqrt{S_{dw}^{\text{corrected}}} + \epsilon} $$
    $$ b \gets b - \frac{\alpha \cdot V_{db}^{\text{corrected}}}{\sqrt{S_{db}^{\text{corrected}}} + \epsilon} $$

**Hyperparameters**

 - Learning rate $ \alpha $ needs tuning.
 - $ \beta_1 $ (momentum term) is usually 0.9.
 - $ \beta_2 $ (RMSprop term) is usually 0.999.
 - $ \epsilon $ is typically set to $10^{-8}$ for stability.

# Learning Rate Decay

 - In mini-batch gradient descent with a fixed learning rate, noise in mini-batches can cause the algorithm to oscillate or wander around the minimum.
 - Gradually reducing the learning rate ($ \alpha $) over time can improve convergence.
 - Smaller updates as training progresses help the algorithm oscillate closer to the global minimum.

**Learning Rate Decay Implementation**
  - Set an initial learning rate $ \alpha_0 $.
  - After each iteration, update the learning rate using:
    $$ \alpha = \frac{\alpha_0}{1 + \text{decay rate} \times \text{epoch number}} $$