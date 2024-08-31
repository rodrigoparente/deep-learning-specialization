# Tuning Process

- Deep learning models have many hyperparameters, and selecting the right ones is crucial for optimal performance.
- Here are tips on prioritizing important hyperparameters and finding their best values.

**Prioritization of Hyperparameters**

 - **Most Important**: Learning rate ($\alpha$) is generally the most crucial hyperparameter.
 - **Second Priority**: Momentum term ($\beta$), mini-batch size, and number of hidden units.
 - **Less Critical**: Number of layers, learning rate decay, and parameters for Adam ($\beta_1$, $\beta_2$, $\epsilon$).
 - Default values for Adam parameters are often used: $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.

**Hyperparameter Search Strategies**

 - **Grid Search:**
    - Traditional method involves exploring a predefined grid of hyperparameter values.
    - Effective with a small number of hyperparameters but becomes impractical with more parameters due to exponential growth in combinations.
 - **Random Sampling:**
    - More efficient for deep learning; involves randomly selecting hyperparameter values.
    - Allows for a richer exploration of the hyperparameter space, focusing more on critical parameters like learning rate.
 - **Coarse-to-Fine Sampling:**
    - Start with a broad search over a large hyperparameter space.
    - Identify promising regions and then conduct a finer search within those regions to refine hyperparameter values.

# Importance of Proper Scaling in Hyperparameter Sampling

 - Sampling uniformly at random across the range of hyperparameter values may not always be effective.
 - For hyperparameters spanning several orders of magnitude, logarithmic scale sampling can be more effective.

**Example of How to Sampling the Learning Rate ($ \alpha $)**

 - Suppose $ \alpha $ ranges from 0.0001 to 1.
 - Uniform sampling on a linear scale will disproportionately favor values closer to 1 (e.g., 0.1 to 1) over smaller values (e.g., 0.0001 to 0.1).
 - To solve this problem, you can use a logarithmic scale to sample more effectively across the entire range.
 - Here is an example of how to do using Python code:
    ```python
    import numpy as np

    # Define the bounds for the learning rate
    lower_bound = 0.0001
    upper_bound = 1

    # Calculate the log base 10 of the bounds
    a = np.log10(lower_bound)  # -4
    b = np.log10(upper_bound)  # 0

    # Sample a random value from the log scale
    r = np.random.uniform(a, b)  # Random value between -4 and 0

    # Convert the sampled value back to the linear scale
    alpha = 10 ** r
    
    print(f"Sampled learning rate (alpha): {alpha}")
    ```

# Hyperparameters Tuning in Practice

 - Deep learning ideas from one domain (e.g., computer vision) often transfer successfully to other domains (e.g., speech, NLP).
 - Researchers frequently adapt techniques across different fields, enhancing the effectiveness of hyperparameter settings.
 - There are two Major Approaches to Hyperparameter Search:

**Panda Approach (Babysitting)**
       
 - Focus on training and fine-tuning one model at a time.
 - Gradually adjust hyperparameters based on performance metrics (e.g., learning curve, cost function).
 - Initialize the model and monitor its performance.
 - Incrementally adjust parameters like learning rate or momentum based on observed results.
 - Requires daily monitoring and adjustments, akin to caring for one baby panda.
 - Best for scenarios with limited computational resources or when training only a few models is feasible.

**Caviar Approach (Batch Testing)**
    
 - Train multiple models in parallel with different hyperparameter settings.
 - Compare the performance of these models based on their learning curves or other metrics.
 - Simultaneously train multiple models with different settings.
 - Evaluate and select the best-performing model based on performance metrics.
 - Similar to how fish lay many eggs and rely on a few succeeding.
 - Ideal when sufficient computational resources are available to train many models simultaneously.
