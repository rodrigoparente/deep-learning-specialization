# Binary Classification

 - Imagine the problem were the user wants to classify an image as either a cat ($y = 1$) or not a cat ($y = 0$).
 - For this problem, we want a classifier that predicts the label $ y $ (1 or 0) based on the feature vector $ \mathbf{x} $. 

**Feature Vector Creation**

 - Images are stored as three separate matrices for red, green, and blue (RGB) color channels.
 - For a $64 \times 64$ image, each color channel matrix is $64 \times 64$ in size.
 - The pixel values from the three RGB matrices are unrolled into a single feature vector $ \mathbf{x} $.
 - For a $64 \times 64$ image, the resulting feature vector $ \mathbf{x} $ will have a dimension of 12,288 ($64 \times 64 \times 3$).

**Notation for Training Data**

  - A single training example is represented by a pair $(\mathbf{x}, y)$, where $ \mathbf{x} $ is the feature vector and $ y $ is the label.
  - The training set consists of $m$ training examples: $(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots, (\mathbf{x}^{(m)}, y^{(m)})$.

**Matrix Representation of Training Data**

 - The input Matrix $ \mathbf{X} $ is created by stacking the feature vectors of all training examples in columns: 
    $$ \mathbf{X} = \begin{bmatrix} \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \dots & \mathbf{x}^{(m)} \end{bmatrix} $$
    - **Dimension:** $ n_x \times m $ (where $ n_x $ is the number of features and $ m $ is the number of training examples).
 - The output Matrix $ \mathbf{Y} $ is created by stacking the labels in columns:
    $$ \mathbf{Y} = \begin{bmatrix} y^{(1)} & y^{(2)} & \dots & y^{(m)} \end{bmatrix} $$
    - **Dimension:** $ 1 \times m $.

Stack data from different training examples in columns (both for $ \mathbf{X} $ and $ \mathbf{Y} $) to simplify neural network implementation. This convention helps streamline coding in Python, particularly when using matrix operations.

# Logistic Regression

 - Logistic regression is a learning algorithm used for binary classification problems, where output labels $ Y $ are either 0 or 1.
 - Given an input feature vector $ \mathbf{X} $ (e.g., an image), the goal is to predict a label $ \hat{Y} $, which is an estimate of $ Y $.
 - More formally, $ \hat{Y} $ should represent the probability that $ Y = 1 $ given the input features $ \mathbf{X} $: 
    $$ \hat{Y} = P(Y = 1 \mid \mathbf{X}) $$
 - A straightforward approach might suggest using:
    $$ \hat{Y} = \mathbf{W}^T \mathbf{X} + b $$
    where:
    - $ \hat{Y} $ is a linear function of $ \mathbf{X} $.
    - $ \mathbf{W} $ is an $ n_x $-dimensional vector (same dimension as $ \mathbf{X} $).
    - $ b $ is a scalar (real number).
 - However, this is not suitable for binary classification, as $ \hat{Y} $ should be a probability, and therefore must be between 0 and 1.
 - The linear function $ \mathbf{W}^T \mathbf{X} + b $ can produce values outside this range, which isn't valid for probabilities.
 - To address this problem, logistic regression uses the **sigmoid function** to ensure the output $ \hat{Y} $ stays between 0 and 1:
    $$ \hat{Y} = \sigma(z) = \frac{1}{1 + e^{-z}} $$
    where $ z = \mathbf{W}^T \mathbf{X} + b $.
 - When $ z $ is large, $ e^{-z} $ approaches 0, so:
      $$ \sigma(z) \approx \frac{1}{1 + 0} = 1 $$
 - When $ z $ is small or negative, $ e^{-z} $ becomes a very large number, so:
      $$ \sigma(z) \approx \frac{1}{1 + \text{(large number)}} \approx 0 $$

The goal of the algorithm is to learn the parameters $ \mathbf{W} $ and $ b $ such that $ \hat{Y} $ is a good estimate of the probability that $ Y = 1 $.

# Loss Function 

 - The loss function $ L(\hat{Y}, Y) $ measures the error for a single training example and is given by:
    $$ L(\hat{Y}, Y) = -\left[ Y \log(\hat{Y}) + (1 - Y) \log(1 - \hat{Y}) \right] $$
 - The intuition behind it is:
    - If $ Y = 1 $, the loss function reduces to:
        $$ L(\hat{Y}, 1) = -\log(\hat{Y}) $$
    - In that case, to minimize the loss, $ \hat{Y} $ should be as large as possible, ideally close to 1.
    - If $ Y = 0 $, the loss function reduces to:
        $$ L(\hat{Y}, 0) = -\log(1 - \hat{Y}) $$
    - In that scenario, to minimize the loss, $ \hat{Y} $ should be as small as possible, ideally close to 0.
  
# Cost Function

 - The cost function $ J(\mathbf{W}, b) $ is defined as the average loss over the entire training set:
    $$ J(\mathbf{W}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ Y^{(i)} \log(\hat{Y}^{(i)}) + (1 - Y^{(i)}) \log(1 - \hat{Y}^{(i)}) \right] $$
 - Here, $ m $ is the number of training examples.
 - The cost function measures how well the logistic regression model performs on the entire training set.
 - The goal of training the logistic regression model is to find the parameters $ \mathbf{W} $ and $ b $ that minimize the cost function $ J(\mathbf{W}, b) $.

# Gradient Descent

  - The objective of Gradient Descent is to minimize the cost function $ J(W, b) $ by adjusting $ W $ and $ b $.
  - If we plot the function $ J(W, b) $, its surface will resemble a convex "bowl".
  - The algorithm starts at an initial point in the surface and iteratively takes steps downhill towards the global minimum.
  - The direction of the step is determined by the gradient, which points in the direction of the steepest ascent; however, the algorithm subtracts this gradient, effectively moving in the direction of the steepest descent.

**Gradient Descent Algorithm**

 - **Initialization**: Start with initial values for $ W $ and $ b $, which often are initialized to zeros.
 - **Iteration**: Repeatedly update $ W $ and $ b $ to move towards the minimum of $ J(W, b) $.
 - **Update Rule**:
    - For the parameter $ W $:
    $$ W := W - \alpha \frac{\partial J(W, b)}{\partial W} $$
    - For the parameter $ b $:
    $$ b := b - \alpha \frac{\partial J(W, b)}{\partial b}  $$

    $ \alpha $ is the learning rate, which controls the size of the step taken towards the minimum.

**Code Implementation**
    
 - In coding, the derivative of the cost function with respect to $ W $ and $ b $ is typically stored in variables named `dW` and `db`, respectively.
 - The update rules in code might look like:
    ```python
    W = W - alpha * dW
    b = b - alpha * db
    ```