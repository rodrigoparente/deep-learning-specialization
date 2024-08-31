# Batch Normalization

 - Batch Normalization aims to make neural networks more robust and simplifies the hyperparameter search.
 - It does that by applying normalization to activations in hidden layers.
 - The normalization process is done:
 
    1. For a given layer, compute the mean ($ \mu $) and variance ($ \sigma^2 $) of the activations $ z_i $ (pre-activation values).
    2. Normalize $z_i$ to have zero mean and unit variance:
        $$ z_i^{\text{norm}} = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
        Where:
        - $\epsilon$ is a small constant added for numerical stability.
    3. After normalization, apply a linear transformation to allow the network to learn different distributions:
        $$ \tilde{z}_i = \gamma z_i^{\text{norm}} + \beta $$
        Where:
        - $\gamma$ (scale) and $\beta$ (shift) are learnable parameters.
        - These parameters enable the network to learn the optimal mean and variance for the activations.
    4. Replace the original $ z_i $ in computations with the normalized values $ \tilde{z}_i $.
  
Normalizing input features improves training efficiency, normalizing hidden layer activations can accelerate training.

# Why Batch Norm Works

 - Covariate shift occurs when the distribution of input features $ X $ changes, even if the mapping from $ X $ to $ Y $ remains unchanged.
 - In deep networks, parameter changes in earlier layers can shift the distributions of values in later layers, causing covariate shift.
 - Batch Norm minimizes these shifts by keeping the mean and variance of hidden unit values consistent.
 - This consistency stabilizes the learning process in deeper layers.
 - Additionally, by stabilizing input distributions, Batch Norm allows each layer to learn more independently, reducing inter-layer dependency.
 - This independence speeds up learning across the entire network. 
